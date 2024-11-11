import re
import asyncio
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
    truncate_list_by_token_size
)
from Core.Schema.ChunkSchema import TextChunk
from Core.Schema.Message import Message
from Core.Prompt import EntityPrompt
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
from Core.Prompt import QueryPrompt
from Core.Common.QueryConfig import query_config

class ERGraph(BaseGraph):
   
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    


    async def _construct_graph(self, chunks: dict[str, TextChunk]):
        try:
            filtered_keys = await self.text_chunks.filter_keys(list(chunks.keys()))
            inserting_chunks = {key: value for key, value in chunks.items() if key in filtered_keys}
            ordered_chunks = list(inserting_chunks.items())
            async def _process_single_content(chunk_key_dp: tuple[str, TextChunk]):
                chunk_key, chunk_dp = chunk_key_dp
                records = await self._extract_records_from_chunk(chunk_dp)
                return await self._organize_records(records, chunk_key)

            results = await asyncio.gather(*[_process_single_content(c) for c in ordered_chunks])
            await self._update_graph(results)
            if self.config.use_community:
                # ---------- update clusterings of graph
                logger.info("Starting [Community Report]")
                community_ins = get_community_instance(self.config.graph_cluster_algorithm, enforce_sub_communities = self.config.enforce_sub_communities, llm = self.llm)
                cluster_node_map = await community_ins._clustering_(self.er_graph.graph, self.config.max_graph_cluster_size, self.config.graph_cluster_seed)
                self.er_graph._cluster_data_to_subgraphs(cluster_node_map)
                await community_ins._generate_community_report_(self.er_graph)
                logger.info("[Community Report]  Finished")
                await self.global_query("who are you", community_ins)
            #TODO: persistent   
            # # ---------- commit upsertings and indexing
            # await self.full_docs.upsert(new_docs)
            # await self.text_chunks.upsert(inserting_chunks)
        finally:
            logger.info("Consturcting graph finisihed")

    
    async def query(self, query: str, param: QueryConfig):
        if param.mode == "local" and not self.config.enable_local:
            raise ValueError("enable_local is False, cannot query in local mode")
        if param.mode == "naive" and not self.config.enable_naive_rag:
            raise ValueError("enable_naive_rag is False, cannot query in naive mode")
        if param.mode == "local":
            response = await self.local_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param
            )
        elif param.mode == "global":
            response = await self.global_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.community_reports,
                self.text_chunks,
                param
            )
        else:
            raise ValueError(f"Unknown mode {param.mode}")
        # await self._query_done()
        return response
    async def local_query():
        pass
    async def _map_global_communities(
            self,
            query: str,
            communities_data
        ):
            
            community_groups = []
            while len(communities_data):
                this_group = truncate_list_by_token_size(
                    communities_data,
                    key=lambda x: x["report_string"],
                    max_token_size=self.query_config.global_max_token_for_community_report,
                )
                community_groups.append(this_group)
                communities_data = communities_data[len(this_group) :]

            async def _process(community_truncated_datas: list[Any]) -> dict:
                communities_section_list = [["id", "content", "rating", "importance"]]
                for i, c in enumerate(community_truncated_datas):
                    communities_section_list.append(
                        [
                            i,
                            c["report_string"],
                            c["report_json"].get("rating", 0),
                            c['community_info'].occurrence,
                        ]
                    )
                community_context = list_to_quoted_csv_string(communities_section_list)
                sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
                sys_prompt = sys_prompt_temp.format(context_data=community_context)
       
                response = await self.llm.aask(
                    query,
                    system_msgs = [sys_prompt]
                )

                data = prase_json_from_response(response)
                return data.get("points", [])

            logger.info(f"Grouping to {len(community_groups)} groups for global search")
            responses = await asyncio.gather(*[_process(c) for c in community_groups])
            return responses

    async def global_query(
            self,
            query,
            community_instance,
        ) -> str:
            self.query_config = query_config
            community_schema = community_instance.community_schema
            community_schema = {
                k: v for k, v in community_schema.items() if v.level <= self.query_config.level
            }
            if not len(community_schema):
                return QueryPrompt.FAIL_RESPONSE

            sorted_community_schemas = sorted(
                community_schema.items(),
                key=lambda x: x[1].occurrence,
                reverse=True,
            )
     
            sorted_community_schemas = sorted_community_schemas[
                : self.query_config.global_max_consider_community
            ]
            community_datas = await community_instance.community_reports.get_by_ids( ###
                [k[0] for k in sorted_community_schemas]
            )
      
            community_datas = [c for c in community_datas if c is not None]
            community_datas = [
                c
                for c in community_datas
                if c["report_json"].get("rating", 0) >= self.query_config.global_min_community_rating
            ]
            community_datas = sorted(
                community_datas,
                key=lambda x: (x['community_info'].occurrence, x["report_json"].get("rating", 0)),
                reverse=True,
            )
            logger.info(f"Revtrieved {len(community_datas)} communities")

            map_communities_points = await self._map_global_communities(
                query, community_datas
            )
            final_support_points = []
            for i, mc in enumerate(map_communities_points):
                for point in mc:
                    if "description" not in point:
                        continue
                    final_support_points.append(
                        {
                            "analyst": i,
                            "answer": point["description"],
                            "score": point.get("score", 1),
                        }
                    )
            final_support_points = [p for p in final_support_points if p["score"] > 0]
            if not len(final_support_points):
                return QueryPrompt.FAIL_RESPONSE
            final_support_points = sorted(
                final_support_points, key=lambda x: x["score"], reverse=True
            )
            final_support_points = truncate_list_by_token_size(
                final_support_points,
                key=lambda x: x["answer"],
                max_token_size=self.query_config.global_max_token_for_community_report,
            )
            points_context = []
            for dp in final_support_points:
                points_context.append(
                    f"""----Analyst {dp['analyst']}----
        Importance Score: {dp['score']}
        {dp['answer']}
        """
                )
            points_context = "\n".join(points_context)
            if self.query_config.only_need_context:
                return points_context
            sys_prompt_temp = QueryPrompt.GLOBAL_REDUCE_RAG_RESPONSE
            response = await self.llm.aask(
                query,
                system_msgs= [sys_prompt_temp.format(
                    report_data=points_context, response_type=self.query_config.response_type
                )],
            )                
            return response
    async def _extract_records_from_chunk(self, chunk_dp: TextChunk):
        context = self._build_context_for_entity_extraction(chunk_dp["content"])
        prompt_template = None
        if self.config.use_keywords:
            prompt_template = EntityPrompt.ENTITY_EXTRACTION_KEYWORD
        else:
            prompt_template = EntityPrompt.ENTITY_EXTRACTION
        prompt = prompt_template.format(**context)
        
        working_memory = Memory()

        working_memory.add(Message(content = prompt, role = "user"))
        final_result = await self.llm.aask(prompt)
        working_memory.add(Message(content = final_result, role = "assistant"))

        for glean_idx in range(self.config.max_gleaning):
            working_memory.add(Message(content=EntityPrompt.ENTITY_CONTINUE_EXTRACTION, role="user"))
            glean_result = await self.llm.aask(working_memory.get())
            working_memory.add(Message(content=glean_result, role="assistant"))
            final_result += glean_result

            if glean_idx == self.config.max_gleaning - 1:
                break

            working_memory.add(Message(content=EntityPrompt.ENTITY_IF_LOOP_EXTRACTION, role="user"))
            if_loop_result = await self.llm.aask(working_memory.get())
            if if_loop_result.strip().strip('"').strip("'").lower() != "yes":
                break
        working_memory.clear()
        return split_string_by_multi_markers(final_result, [
            DEFAULT_RECORD_DELIMITER, DEFAULT_COMPLETION_DELIMITER
        ])

    async def _organize_records(self, records: list[str], chunk_key: str):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)

        for record in records:
            match = re.search(r"\((.*)\)", record)
            if match is None:
                continue

            record_attributes = split_string_by_multi_markers(match.group(1), [DEFAULT_TUPLE_DELIMITER])
            entity = await self._handle_single_entity_extraction(record_attributes, chunk_key)

            if entity is not None:
                maybe_nodes[entity.entity_name].append(entity)
                continue
       
            relationship = await self._handle_single_relationship_extraction(record_attributes, chunk_key)
         
            if relationship is not None:
                maybe_edges[(relationship.src_id, relationship.tgt_id)].append(relationship)

        return dict(maybe_nodes), dict(maybe_edges)

    async def _handle_single_entity_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Entity, None]:
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None

        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None

        return Entity(
            entity_name = entity_name,
            entity_type = clean_str(record_attributes[2].upper()),
            description = clean_str(record_attributes[3]),
            source_id = chunk_key
        )

    async def _handle_single_relationship_extraction(self, record_attributes: list[str], chunk_key: str) -> Union[Relationship, None]:
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None

        return Relationship(
            src_id=clean_str(record_attributes[1].upper()),
            tgt_id=clean_str(record_attributes[2].upper()),
            weight=float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0,
            description=clean_str(record_attributes[3]),
            source_id=chunk_key,
            keywords = clean_str(record_attributes[4]) if self.config.use_keywords else None
        )

    async def _update_graph(self, results: list, use_keywords: bool = False):
        maybe_nodes, maybe_edges = defaultdict(list), defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)

        await asyncio.gather(*[self._merge_nodes_then_upsert(k, v) for k, v in maybe_nodes.items()])
       
        await asyncio.gather(*[self._merge_edges_then_upsert(k[0], k[1], v) for k, v in maybe_edges.items()])

    async def _merge_nodes_then_upsert(self, entity_name: str, nodes_data: list[Entity]):
        existing_node = await self.er_graph.get_node(entity_name)
        existing_data = [[],[],[]] if existing_node is None else [
            existing_node["entity_type"],
            split_string_by_multi_markers(existing_node["source_id"], [GRAPH_FIELD_SEP]),
            existing_node["description"]
        ]

        entity_type = sorted(
            Counter(dp.entity_type for dp in nodes_data + existing_data[0]).items(),
            key=lambda x: x[1], reverse=True
        )[0][0]

        description = GRAPH_FIELD_SEP.join(
            sorted(set(dp.description for dp in nodes_data) | set(existing_data[2]))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set(dp.source_id for dp in nodes_data) | set(existing_data[1])
        )

        description = await self._handle_entity_relation_summary(entity_name, description)

        node_data = dict(entity_type=entity_type, description=description, source_id=source_id)
 
        await self.er_graph.upsert_node(entity_name, node_data=node_data)
        return {**node_data, "entity_name": entity_name}


    async def _merge_edges_then_upsert(self, src_id: str, tgt_id: str, edges_data: list[Relationship]):
        existing_edge_data = {}

        if await self.er_graph.has_edge(src_id, tgt_id):
            existing_edge_data = await self.er_graph.get_edge(src_id, tgt_id)

        #NOTE: For the nano-rag, it supports DSpy 
        weight = sum(dp.weight for dp in edges_data) + existing_edge_data.get("weight", 0)

        description = GRAPH_FIELD_SEP.join(
            sorted(set(dp.description for dp in edges_data) | {existing_edge_data.get("description", "")})
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp.source_id for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("source_id", ""), [GRAPH_FIELD_SEP])
        ))
        if self.config.use_keywords:
            keywords = GRAPH_FIELD_SEP.join(
                sorted(set([dp.keywords for dp in edges_data] + split_string_by_multi_markers(existing_edge_data.get("keywords", ""), [GRAPH_FIELD_SEP])
            )))
            
        for need_insert_id in (src_id, tgt_id):
            if not await self.er_graph.has_node(need_insert_id):
                await self.er_graph.upsert_node(
                    need_insert_id,
                    node_data=dict(source_id=source_id, description=description, entity_type='"UNKNOWN"')
                )

        description = await self._handle_entity_relation_summary((src_id, tgt_id), description)

        await self.er_graph.upsert_edge(src_id, tgt_id, edge_data=dict(weight=weight, description=description, source_id=source_id, keywords = keywords))

    async def _handle_entity_relation_summary(self, entity_or_relation_name: str, description: str) -> str:

        tokens = self.ENCODER.encode(description)
        if len(tokens) < self.config.summary_max_tokens:
            return description

        use_description = self.ENCODER.decode(tokens[:self.llm.get_maxtokens()])
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP)
        )
        use_prompt = EntityPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        return await self.llm.aask(use_prompt, max_tokens = self.config.summary_max_tokens)
        

    def _build_context_for_entity_extraction(self, content: str) -> dict:
        return dict(
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,
            record_delimiter = DEFAULT_RECORD_DELIMITER,
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(DEFAULT_ENTITY_TYPES),
            input_text=content
        )

    # 
    def _extract_node(self):
        pass

    def _extract_relationship(self):
        pass

    def _exist_graph(self):
        pass


    