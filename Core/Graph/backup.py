from Core.Graph.BaseGraph import BaseGraph
from Core.Common.LLM import BaseLLM
from pydantic import model_validator
from Core.Common.Logger import logger
from Core.Common.Utils import clean_str, split_string_by_multi_markers, is_float_regex, encode_string_by_tiktoken, decode_tokens_by_tiktoken
from Core.Schema.ChunkSchema import TextChunk
from GraphRAG.Core.Community.LeidenCommunity import LeidenCommunity
from Core.Storage.BaseGraphStorage import BaseGraphStorage
from Core.Storage.BaseKVStorage import BaseKVStorage
from Core.Storage.JsonKVStorage import JsonKVStorage
import re
import asyncio
from typing import TypedDict, Union, Type
from Core.Common.Constants import GRAPH_FIELD_SEP, Process_tickers
from Core.Schema.Message import Message
import Core.Prompt.EntityPrompt as EntityPrompt
from Core.Common.Memory import Memory
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from Core.Storage.NetworkXStorage import NetworkXStorage
async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    ):
        if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
            return None
        # add this record as a node in the G
        entity_name = clean_str(record_attributes[1].upper())
        if not entity_name.strip():
            return None
        entity_type = clean_str(record_attributes[2].upper())
        entity_description = clean_str(record_attributes[3])
        entity_source_id = chunk_key
        return dict(
            entity_name=entity_name,
            entity_type=entity_type,
            description=entity_description,
            source_id=entity_source_id,
        )
        
    
async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
    ):
        if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
            return None
        # add this record as edge
        source = clean_str(record_attributes[1].upper())
        target = clean_str(record_attributes[2].upper())
        edge_description = clean_str(record_attributes[3])
        edge_source_id = chunk_key
        weight = (
            float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
        )
        return dict(
            src_id=source,
            tgt_id=target,
            weight=weight,
            description=edge_description,
            source_id=edge_source_id,
        )
async def _handle_entity_relation_summary(
        entity_or_relation_name: str,
        description: str,
        global_config: dict,
    ) -> str:
        use_llm_func: callable = global_config["cheap_model_func"]
        llm_max_tokens = global_config["cheap_model_max_token_size"]
        tiktoken_model_name = global_config["tiktoken_model_name"]
        summary_max_tokens = global_config["entity_summary_to_max_tokens"]
        tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
        if len(tokens) < summary_max_tokens:  # No need for summary
            return description
        prompt_template = EntityPrompt.SUMMARIZE_ENTITY_DESCRIPTIONS
        use_description = decode_tokens_by_tiktoken(
            tokens[:llm_max_tokens], model_name=tiktoken_model_name
        )
        context_base = dict(
            entity_name=entity_or_relation_name,
            description_list=use_description.split(GRAPH_FIELD_SEP),
        )
        use_prompt = prompt_template.format(**context_base)
        logger.debug(f"Trigger summary: {entity_or_relation_name}")
        summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
        return summary
    
class ERGraph(BaseGraph):
    text_chunks: JsonKVStorage = JsonKVStorage()
    er_graph: NetworkXStorage = NetworkXStorage()
    async def _merge_nodes_then_upsert(
        self,
        entity_name: str,
        nodes_data: list[dict]
    ):
        already_entitiy_types = []
        already_source_ids = []
        already_description = []
        already_node = await self.er_graph.get_node(entity_name)
        if already_node is not None:
            already_entitiy_types.append(already_node["entity_type"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_node["description"])
        entity_type = sorted(
            Counter(
                [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
            ).items(),
            key=lambda x: x[1],
            reverse=True,
        )[0][0]
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in nodes_data] + already_description))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp["source_id"] for dp in nodes_data] + already_source_ids)
        )
        description = await _handle_entity_relation_summary(
            entity_name, description, {}
        )
        node_data = dict(
            entity_type=entity_type,
            description=description,
            source_id=source_id,
        )
        await self.er_graph.upsert_node(
            entity_name,
            node_data=node_data,
        )
        node_data["entity_name"] = entity_name
        return node_data
    
    async def _merge_edges_then_upsert(
        self,
        src_id: str,
        tgt_id: str,
        edges_data: list[dict],
    ):
        already_weights = []
        already_source_ids = []
        already_description = []
        already_order = []
        if await self.er_graph.has_edge(src_id, tgt_id):
            already_edge = await self.er_graph.get_edge(src_id, tgt_id)
            already_weights.append(already_edge["weight"])
            already_source_ids.extend(
                split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
            )
            already_description.append(already_edge["description"])
            already_order.append(already_edge.get("order", 1))
        # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
        order = min([dp.get("order", 1) for dp in edges_data] + already_order)
        weight = sum([dp["weight"] for dp in edges_data] + already_weights)
        description = GRAPH_FIELD_SEP.join(
            sorted(set([dp["description"] for dp in edges_data] + already_description))
        )
        source_id = GRAPH_FIELD_SEP.join(
            set([dp["source_id"] for dp in edges_data] + already_source_ids)
        )
        for need_insert_id in [src_id, tgt_id]:
            if not (await self.er_graph.has_node(need_insert_id)):
                await self.er_graph.upsert_node(
                    need_insert_id,
                    node_data={
                        "source_id": source_id,
                        "description": description,
                        "entity_type": '"UNKNOWN"',
                    },
                )
        description = await _handle_entity_relation_summary(
            (src_id, tgt_id), description, {}
        )
        await self.er_graph.upsert_edge(
            src_id,
            tgt_id,
            edge_data=dict(
                weight=weight, description=description, source_id=source_id, order=order
            ),
        )
    async def extract_entities(
        self,
        chunks: dict[str, TextChunk],
        knwoledge_graph_inst: BaseGraphStorage,
  
        global_config: dict,
    ) -> Union[BaseGraphStorage, None]:
        
        # use_llm_func: callable = global_config["best_model_func"]
        entity_extract_max_gleaning = 1
        ordered_chunks = list(chunks.items())
        entity_extract_prompt = EntityPrompt.ENTITY_EXTRACTION
        context_base = dict(
            tuple_delimiter = EntityPrompt.DEFAULT_TUPLE_DELIMITER,
            record_delimiter = EntityPrompt.DEFAULT_RECORD_DELIMITER,
            completion_delimiter = EntityPrompt.DEFAULT_COMPLETION_DELIMITER,
            entity_types=",".join(EntityPrompt.DEFAULT_ENTITY_TYPES),
        )
        continue_prompt = EntityPrompt.ENTITY_CONTINUE_EXTRACTION
        if_loop_prompt = EntityPrompt.ENTITY_IF_LOOP_EXTRACTION
        already_processed = 0
        already_entities = 0
        already_relations = 0
        async def _process_single_content(chunk_key_dp: tuple[str, TextChunk]):
            nonlocal already_processed, already_entities, already_relations
            chunk_key = chunk_key_dp[0]
            chunk_dp = chunk_key_dp[1]
            content = chunk_dp["content"]
            hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
            working_memory = Memory()
    
            working_memory.add(Message(content= hint_prompt, role = "user"))
            
            final_result = await self.llm.aask(hint_prompt)
            final_msg = Message(content= final_result, role = "assistant")
            working_memory.add(final_msg)
     
       
            for now_glean_index in range(1):
                working_memory.add(Message(content = continue_prompt, role = "user"))
       
                glean_result = await self.llm.aask(working_memory.get())
             
                working_memory.add(Message(content = glean_result, role = "assistant"))
             
                final_result += glean_result
                if now_glean_index == entity_extract_max_gleaning - 1:
                    break
                working_memory.add(Message(content = if_loop_prompt, role = "user"))
                if_loop_result: str = await self.llm.aask(self.working_memory.get())
                if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
                if if_loop_result != "yes":
                    break
            records = split_string_by_multi_markers(
                final_result,
                [context_base["record_delimiter"], context_base["completion_delimiter"]],
            )
            maybe_nodes = defaultdict(list)
            maybe_edges = defaultdict(list)
            for record in records:
                record = re.search(r"\((.*)\)", record)
                if record is None:
                    continue
                record = record.group(1)
                record_attributes = split_string_by_multi_markers(
                    record, [context_base["tuple_delimiter"]]
                )
                if_entities = await _handle_single_entity_extraction(
                    record_attributes, chunk_key
                )
                if if_entities is not None:
                    maybe_nodes[if_entities["entity_name"]].append(if_entities)
                    continue
                if_relation = await _handle_single_relationship_extraction(
                    record_attributes, chunk_key
                )
                if if_relation is not None:
                    maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                        if_relation
                    )
            already_processed += 1
            already_entities += len(maybe_nodes)
            already_relations += len(maybe_edges)
            now_ticks = Process_tickers[
                already_processed % len(Process_tickers)
            ]
            print(
                f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
                end="",
                flush=True,
            )
            return dict(maybe_nodes), dict(maybe_edges)
        # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
        results = await asyncio.gather(
            *[_process_single_content(c) for c in ordered_chunks]
        )
        import pdb
        pdb.set_trace()
        print()  # clear the progress bar
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                # it's undirected graph
                maybe_edges[tuple(sorted(k))].extend(v)
        all_entities_data = await asyncio.gather(
            *[
                self._merge_nodes_then_upsert(k, v, knwoledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        )
        await asyncio.gather(
            *[
                self._merge_edges_then_upsert(k[0], k[1], v)
                for k, v in maybe_edges.items()
            ]
        )
        if not len(all_entities_data):
            logger.warning("Didn't extract any entities, maybe your LLM is not working")
            return None
      
        # return knwoledge_graph_inst
    def _extract_node(self):
        pass
    def _extract_relationship(self):
        pass
    def _exist_graph(self):
        pass
    def _construct_graph(self):
        pass
   
    async def _build_graph_(self, docs):
        #TODO: 1. 修改key_string_value_json_storage_cls
        #TODO: 2. 支持load json data to avoid re-extract
        document_keys = list(docs.keys())
        filtered_keys = await self.text_chunks.filter_keys(document_keys)
        inserting_chunks = {key: value for key, value in docs.items() if key in filtered_keys}
        await self.extract_entities(inserting_chunks) 