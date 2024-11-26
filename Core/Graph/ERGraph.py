import re
import asyncio
import json
from collections import defaultdict, Counter
from typing import Union, Any
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Common.Utils import (
    clean_str,
    split_string_by_multi_markers,
    is_float_regex,
    list_to_quoted_csv_string,
    prase_json_from_response,
    truncate_list_by_token_size,
    mdhash_id,
    processing_phrases,
    min_max_normalize
)
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import EntityPrompt, QueryPrompt
from Core.Schema.EntityRelation import Entity, Relationship
from Core.Storage.JsonKVStorage import JsonKVStorage
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Common.Constants import (
    GRAPH_FIELD_SEP, 
    DEFAULT_ENTITY_TYPES, 
    DEFAULT_RECORD_DELIMITER, 
    DEFAULT_COMPLETION_DELIMITER,
    DEFAULT_TUPLE_DELIMITER
)
from Core.Common.Memory import Memory
from Core.Community.ClusterFactory import get_community_instance
from Core.Common.QueryConfig import QueryConfig
from Core.Common.QueryConfig import query_config
from metagpt.provider.llm_provider_registry import create_llm_instance
from pydantic import model_validator
from Core.Index import (
    get_rag_embedding
)    
from Core.Index.Schema import (
     FAISSIndexConfig,
     ColBertIndexConfig
)
from Core.Index.VectorIndex import VectorIndex
class ERGraph(BaseGraph):
   
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    
    @model_validator(mode="after")
    def _init_vectordb(cls, data):
        # index_config = FAISSIndexConfig(persist_path="./storage", embed_model = get_rag_embedding())
        index_config = ColBertIndexConfig(persist_path="./storage/colbert_index", index_name="nbits_2", model_name=cls.config.colbert_checkpoint_path, nbits=2)
        cls.entity_vdb = VectorIndex(index_config)
      

        return data

    async def _construct_graph(self, chunks: dict[str, TextChunk]):
        try:
            filtered_keys = await self.text_chunks.filter_keys(list(chunks.keys()))
            inserting_chunks = {key: value for key, value in chunks.items() if key in filtered_keys}
            ordered_chunks = list(inserting_chunks.items())
       
            async def extract_openie_from_triples(chunk_key_dp: tuple[str, TextChunk]):
                chunk_key, chunk_dp = chunk_key_dp
                entities = await self.named_entity_recognition(chunk_dp)
                triples =  await self._openie_post_ner_extract(chunk_dp, entities)
                return await self._organize_records(entities, triples, chunk_key)
            #TODO: support persist for the extracted enetities and tirples
            results = await asyncio.gather(*[extract_openie_from_triples(c) for c in ordered_chunks])

            # Build graph based on the extracted entities and triples
            await self._add_to_graph(results)
            
            # # ---------- commit upsertings and indexing
            # await self.full_docs.upsert(new_docs)
            await self.text_chunks.upsert(inserting_chunks)
    
        finally:
            logger.info("Consturcting graph finisihed")

    

   
    async def named_entity_recognition(self, passage: str):
        ner_messages = EntityPrompt.NER.format(user_input=passage)

        response_content = await self.llm.aask(ner_messages)
        entities = prase_json_from_response(response_content)
        
        if 'named_entities' not in entities:
            entities = []
        else:
            entities = entities['named_entities']



        return entities

    async def _openie_post_ner_extract(self, chunk, entities):
        named_entity_json = {"named_entities": entities}
        openie_messages = EntityPrompt.OPENIE_POST_NET.format(passage=chunk, named_entity_json=json.dumps(named_entity_json))
        response_content = await self.llm.aask(openie_messages)
        triples = prase_json_from_response(response_content)
        try:
            triples = triples["triples"]
        except:
           return []

        return triples



    async def _organize_records(self, entities, triples, chunk_key: str):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for entity in entities :
            entity_name =  processing_phrases(entity)
            maybe_nodes[entity_name].append({
                "source_id": chunk_key,
            })

        for triple in triples: 
            relationship = Relationship(src_id = processing_phrases(triple[0]), 
                                        tgt_id = processing_phrases(triple[2]), 
                                        weight=1.0, source_id=chunk_key, relation_name = processing_phrases(triple[1]))
            maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

   

    async def _add_to_graph(self, results: list):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
    
        entities = await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
 
        # If there is a vectordb, we need to upsert the entities
        if hasattr(self, "entity_vdb") and self.entity_vdb is not None:
            if not isinstance(self.entity_vdb.config, ColBertIndexConfig):
                data_for_vdb = {
                    mdhash_id(dp["entity_name"], prefix="ent-"): {
                        "content": dp["entity_name"],
                        "entity_name": dp["entity_name"],
                    }
                    for dp in entities
                }
            else:
                # If the colbert indexer is set, we need to upsert the entities
                data_for_vdb = [dp['entity_name'] for dp in entities]

            await self.entity_vdb.upsert(data_for_vdb)
            

        #Augment the graph by ann searching & Store the similarity objects for indexing. 
        if self.config.enable_graph_augmentation:
            data_for_aug =  { mdhash_id(dp["entity_name"], prefix="ent-"): dp["entity_name"] for dp in entities}
            maybe_edges_aug = await self._augment_graph(queries = data_for_aug)

        # Merge the edges
        for k, v in maybe_edges_aug.items():
            maybe_edges[tuple(sorted(k))].extend(v)

        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _augment_graph(self, queries, similarity_threshold = 0.8, similarity_top_k = 100, duplicate = True):
        """
        For each entity in the graph, get its synonyms from the knowledge base
        queries: list of enetity names
        """
        ranking = await self.entity_vdb.retrieve_batch(queries, top_k = similarity_top_k) 
        entity_names = list(queries.values())
        kb_similarity = {}
        for key, entity_name in queries.items():
            rank = ranking.data[key]
            filtered_rank = rank[1:] if duplicate else rank
            kb_similarity[entity_name] = (
                [entity_names[r[0]] for r in filtered_rank],
                [r[2] / rank[0][2] for r in filtered_rank]
            )

        maybe_edges = defaultdict(list)
        # Refactored second part using dictionary iteration and enumerate
        for src_id, nns in kb_similarity.items():
            processed_nns = [processing_phrases(nn) for nn in nns[0]]
            for idx, (nn, score) in enumerate(zip(processed_nns, nns[1])):
                if score < similarity_threshold or idx >= similarity_top_k:
                    break
                if nn == src_id:
                    continue
                tgt_id = nn

                # No need source_id for this type of edges
                relationship = Relationship(src_id = processing_phrases(src_id), 
                                        tgt_id = processing_phrases(tgt_id), 
                                        source_id = "N/A",
                                        weight= self.config.similarity_max * score, relation_name = "similarity")
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return maybe_edges
              
    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: list[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        existing_data = [[]] if existing_node is None else [
            existing_node["source_id"],
        ]

        source_id = GRAPH_FIELD_SEP.join(
            set(dp["source_id"] for dp in nodes_data) | set(existing_data[0])
        )

        node_data = dict(source_id = source_id)
 
        await self.er_graph.upsert_node(entity_name, node_data=node_data)
        return {**node_data, "entity_name": entity_name}


    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: list[Relationship]):
        existing_edge_data = {}

        if await self.er_graph.has_edge(src_id, tgt_id):
            existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id)

        #NOTE: For the nano-rag, it supports DSpy 
        weight = sum(dp.weight for dp in edges_data) + existing_edge_data.get("weight", 0)

        source_id = GRAPH_FIELD_SEP.join(
            set([dp.source_id for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("source_id", ""), [GRAPH_FIELD_SEP])
        ))
        relation_name = GRAPH_FIELD_SEP.join(
            sorted(set(dp.relation_name for dp in edges_data) | {existing_edge_data.get("relation_name", "")})
        )
        for need_insert_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(need_insert_id):
                await self.er_graph.upsert_node(
                    need_insert_id,
                    node_data=dict(source_id=source_id)
                )
        edge_data = dict(weight=weight, source_id=source_id, relation_name = relation_name)
        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data = edge_data)
  
        

    def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
        merged_dict = {}

        # Iterate through each element in the list
        for element in elements:
            # Split the element into lines and get the first line
            lines = element.split('\n')
            first_line = lines[0]

            # Check if the first line is already a key in the dictionary
            if first_line in merged_dict:
                # Append the current element to the existing value
                merged_dict[first_line] += "\n" + element.split(first_line, 1)[1].strip('\n')
            else:
                # Add the current element as a new entry in the dictionary
                merged_dict[first_line] = prefix + element

        # Extract the merged elements from the dictionary
        merged_elements = list(merged_dict.values())
        return merged_elements


    def merge_elements_with_same_first_line(self, elements, prefix='Wikipedia Title: '):
        merged_dict = defaultdict(str)
        for element in elements:
            lines = element.split('\n', 1)
            first_line = lines[0]
            content = lines[1] if len(lines) > 1 else ''
            merged_dict[first_line] += '\n' + content
        merged_elements = [prefix + first_line + merged_dict[first_line] for first_line in merged_dict]
        return merged_elements
    def reason_step(self, dataset, few_shot: list, query: str, passages: list, thoughts: list, client):
        """
        Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
        :return: next thought
        """
        prompt_demo = ''
        for sample in few_shot:
            prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

        prompt_user = ''
        if dataset in ['hotpotqa', 'hotpotqa_train']:
            passages = self.merge_elements_with_same_first_line(passages)
        for passage in passages:
            prompt_user += f'{passage}\n\n'
        prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)

        messages = ChatPromptTemplate.from_messages([SystemMessage(ircot_reason_instruction + '\n\n' + prompt_demo),
                                                    HumanMessage(prompt_user)]).format_prompt()

        try:
            chat_completion = client.invoke(messages.to_messages())
            response_content = chat_completion.content
        except Exception as e:
            print(e)
            return ''
    

    def link_nodes(self, query_ner_list):
        phrase_ids = []
        max_scores = []

        for query in query_ner_list:
            queries = Queries(path=None, data={0: query})

            queries_ = [query]
            encoded_query = self.phrase_searcher.encode(queries_, full_length_search=False)

            max_score = self.get_colbert_max_score(query)

            ranking = self.phrase_searcher.search_all(queries, k=1)
            for phrase_id, rank, score in ranking.data[0]:
                phrase = self.phrases[phrase_id]
                phrases_ = [phrase]
                encoded_doc = self.phrase_searcher.checkpoint.docFromText(phrases_).float()
                real_score = encoded_query[0].matmul(encoded_doc[0].T).max(dim=1).values.sum().detach().cpu().numpy()

                phrase_ids.append(phrase_id)
                max_scores.append(real_score / max_score)

        # create a vector (num_doc) with 1s at the indices of the retrieved documents and 0s elsewhere
        top_phrase_vec = np.zeros(len(self.phrases))

        for phrase_id in phrase_ids:
            if self.node_specificity:
                if self.phrase_to_num_doc[phrase_id] == 0:
                    weight = 1
                else:
                    weight = 1 / self.phrase_to_num_doc[phrase_id]
                top_phrase_vec[phrase_id] = weight
            else:
                top_phrase_vec[phrase_id] = 1.0

        return top_phrase_vec, {(query, self.phrases[phrase_id]): max_score for phrase_id, max_score, query in zip(phrase_ids, max_scores, query_ner_list)}
    async def rank_docs(self, query: str, top_k=10):
        """
        Rank documents based on the query using ColBERTv2 and PPR.
        @param query: the input phrase
        @param top_k: the number of documents to return
        @return: the ranked document ids and their scores
        """
        
        assert isinstance(query, str), 'Query must be a string'
        query_entities = await self._extract_eneity_from_query(query)
    
        # Use ColBERTv2 for retrieval
        queries = Queries(path=None, data={0: query})
        
        # Use PPR for graph algorithm
        if len(query_entities) > 0:
            all_phrase_weights, linking_score_map = self.link_nodes(query_entities)
            ppr_phrase_probs = self.run_pagerank_igraph_chunk([all_phrase_weights])[0]
        else:  # no entities found
            logger.warning('No entities found in query')
            ppr_doc_prob = np.ones(len(self.extracted_triples)) / len(self.extracted_triples)
        # Combine scores using PPR
        fact_prob = self.facts_to_phrases_mat.dot(ppr_phrase_probs)
        ppr_doc_prob = self.docs_to_facts_mat.dot(fact_prob)
        ppr_doc_prob = min_max_normalize(ppr_doc_prob)
        
        # Final document probability
        doc_prob = ppr_doc_prob
        
        # Return top k documents
        sorted_doc_ids = np.argsort(doc_prob, kind='mergesort')[::-1]
        sorted_scores = doc_prob[sorted_doc_ids]
        
        # Logging for debugging
        if len(query_ner_list) > 0:
            phrase_one_hop_triples = []
            for phrase_id in np.where(all_phrase_weights > 0)[0]:
                for t in list(self.kg_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1]])
                for t in list(self.kg_inverse_adj_list[phrase_id].items())[:20]:
                    phrase_one_hop_triples.append([self.phrases[t[0]], t[1], 'inv'])
            
            nodes_in_retrieved_doc = []
            for doc_id in sorted_doc_ids[:5]:
                node_id_in_doc = list(np.where(self.doc_to_phrases_mat[[doc_id], :].toarray()[0] > 0)[0])
                nodes_in_retrieved_doc.append([self.phrases[node_id] for node_id in node_id_in_doc])
            
            top_pagerank_phrase_ids = np.argsort(ppr_phrase_probs, kind='mergesort')[::-1][:20]
            top_ranked_nodes = [self.phrases[phrase_id] for phrase_id in top_pagerank_phrase_ids]
            
            logs = {
                'named_entities': query_ner_list,
                'linked_node_scores': [list(k) + [float(v)] for k, v in linking_score_map.items()],
                '1-hop_graph_for_linked_nodes': phrase_one_hop_triples,
                'top_ranked_nodes': top_ranked_nodes,
                'nodes_in_retrieved_doc': nodes_in_retrieved_doc
            }
        else:
            logs = {}
    
        return sorted_doc_ids.tolist()[:top_k], sorted_scores.tolist()[:top_k], logs
    async def retrieve_step(self, query: str, corpus, top_k: int, dataset: str):
        ranks, scores, logs = await self.rank_docs(query, top_k=top_k)
        if dataset in ['hotpotqa', 'hotpotqa_train']:
            retrieved_passages = []
            for rank in ranks:
                key = list(corpus.keys())[rank]
                retrieved_passages.append(key + '\n' + ''.join(corpus[key]))
        else:
            retrieved_passages = [corpus[rank]['title'] + '\n' + corpus[rank]['text'] for rank in ranks]
        return retrieved_passages, scores, logs
    
    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass
    

    def _extract_eneity_from_query(self, query):
        entities = []
        try:
            if query in self.named_entity_cache:
                query_ner_list = self.named_entity_cache[query]['named_entities']
            else:
                query_ner_json = await self.named_entity_recognition(query)
                query_ner_list = eval(query_ner_json)['named_entities']

            query_ner_list = [processing_phrases(p) for p in query_ner_list]
        except:
            self.logger.error('Error in Query NER')
            query_ner_list = []
        return query_ner_list

    async def query(self, query: str, param: QueryConfig):
        
       # Initial retrieval step
        initial_passages, initial_scores, initial_logs = await self.retrieve_step(
            query, self.chunks, param.top_k, dataset="hotpotqa"
        )
        iteration = 1
        all_logs[iteration] = initial_logs

        thoughts = []
        passage_scores = {
            passage: score for passage, score in zip(initial_passages, initial_scores)
        }

        # Iterative refinement loop
        for iteration in range(2, param.max_ir_steps + 1):
            # Generate a new thought based on current passages and thoughts
            new_thought = self.reason_step(
                args.dataset, few_shot_samples, query, initial_passages[:args.top_k], thoughts, client
            )
            thoughts.append(new_thought)
            
            # Check if the thought contains the answer
            if 'So the answer is:' in new_thought:
                break
            
            # Retrieve new passages based on the new thought
            new_passages, new_scores, retrieval_logs = self.retrieve_step(
                new_thought, corpus, args.top_k, rag, args.dataset
            )
            all_logs[iteration] = retrieval_logs
            
            # Update passage scores
            for passage, score in zip(new_passages, new_scores):
                if passage in passage_scores:
                    passage_scores[passage] = max(passage_scores[passage], score)
                else:
                    passage_scores[passage] = score
            
            # Sort passages by score in descending order
            sorted_passages = sorted(
                passage_scores.items(), key=lambda item: item[1], reverse=True
            )
            initial_passages, initial_scores = zip(*sorted_passages)