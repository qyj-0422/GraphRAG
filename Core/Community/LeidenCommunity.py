from Core.Community.BaseCommunity import BaseCommunity
import json
from collections import defaultdict
import networkx as nx
from graspologic.partition import hierarchical_leiden
import asyncio
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Common.Logger import logger
from Core.Common.Utils import (
    community_report_from_json,
    list_to_quoted_csv_string,
    prase_json_from_response,
    encode_string_by_tiktoken,
    truncate_list_by_token_size
)
from Core.Common.Logger import logger
import asyncio
from Core.Schema.CommunitySchema import CommunityReportsResult, LeidonInfo
from Core.Prompt import CommunityPrompt
from Core.Community.ClusterFactory import register_community
from Core.Storage.NetworkXStorage import NetworkXStorage
from Core.Storage.BaseGraphStorage import BaseGraphStorage
from Core.Storage.JsonKVStorage import JsonKVStorage

@register_community(name = "leiden")
class LeidenCommunity(BaseCommunity):
    
    schemas: dict[str, LeidonInfo] = defaultdict(LeidonInfo)
    _community_reports: JsonKVStorage = JsonKVStorage()



    @property
    def community_reports(self):
        """Getter method for community_reports."""
        return self._community_reports

 

    async def _clustering_(self, graph: nx.Graph, max_cluster_size: int, random_seed: int):
        return await self._leiden_clustering(graph, max_cluster_size, random_seed)


    async def _leiden_clustering(self, graph, max_cluster_size: int, random_seed: int):

        graph = NetworkXStorage.stable_largest_connected_component(graph)
        community_mapping = hierarchical_leiden(
            graph,
            max_cluster_size = max_cluster_size,
            random_seed = random_seed,
        )

        node_communities: dict[str, list[dict[str, str]]] = defaultdict(list)
        __levels = defaultdict(set)
        for partition in community_mapping:
            level_key = partition.level
            cluster_id = partition.cluster
            node_communities[partition.node].append(
                {"level": level_key, "cluster": cluster_id}
            )
            __levels[level_key].add(cluster_id)
        node_communities = dict(node_communities)
        __levels = {k: len(v) for k, v in __levels.items()}
        logger.info(f"Each level has communities: {dict(__levels)}")
        return node_communities
        # self._cluster_data_to_subgraphs(node_communities)

    async def _community_schema_(self, graph: nx.Graph):

        max_num_ids = 0
        levels = defaultdict(set)
     
        for node_id, node_data in graph.nodes(data=True):
            if "clusters" not in node_data:
                continue
            clusters = json.loads(node_data["clusters"])
            this_node_edges = graph.edges(node_id)

            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                levels[level].add(cluster_key)
                self.schemas[cluster_key].level = level
                self.schemas[cluster_key].title = f"Cluster {cluster_key}"
                self.schemas[cluster_key].nodes.add(node_id)
                self.schemas[cluster_key].edges.update(
                    [tuple(sorted(e)) for e in this_node_edges]
                )
                self.schemas[cluster_key].chunk_ids.update(
                    node_data["source_id"].split(GRAPH_FIELD_SEP)
                )
                max_num_ids = max(max_num_ids, len(self.schemas[cluster_key].chunk_ids))

        ordered_levels = sorted(levels.keys())
        for i, curr_level in enumerate(ordered_levels[:-1]):
            next_level = ordered_levels[i + 1]
            this_level_comms = levels[curr_level]
            next_level_comms = levels[next_level]
            # compute the sub-communities by nodes intersection
            for comm in this_level_comms:
                 self.schemas[comm].sub_communities = [
                    c
                    for c in next_level_comms
                    if  self.schemas[c].nodes.issubset( self.schemas[comm].nodes)
                ]

        for _, v in self.schemas.items():
            v.edges = list(v.edges)
            v.edges = [list(e) for e in v.edges]
            v.nodes = list(v.nodes)
            v.chunk_ids = list(v.chunk_ids)
            v.occurrence = len(v.chunk_ids) / max_num_ids
        return  self.schemas

    @property
    def community_schema(self):
        return self.schemas
    
    async def _generate_community_report_(self, er_graph : BaseGraphStorage):
        # Fetch community schema
        communities_schema = await self._community_schema_(er_graph.graph)

        community_keys, community_values = list(communities_schema.keys()), list(communities_schema.values())
        # Generate reports by community levels
        levels = sorted(set([c.level for c in community_values]), reverse=True)
        logger.info(f"Generating by levels: {levels}")
        community_datas = {}
        
        for level in levels:
            this_level_community_keys, this_level_community_values = zip(
                *[(k, v) for k, v in zip(community_keys, community_values) if v.level == level]
            )
            this_level_communities_reports = await asyncio.gather(
                *[self._form_single_community_report(er_graph, c, community_datas) for c in this_level_community_values]
            )
          
            community_datas.update(
                {
                    k: {
                        "report_string": community_report_from_json(r),
                        "report_json": r,
                        "community_info": v
                    }
                    for k, r, v in zip(this_level_community_keys, this_level_communities_reports, this_level_community_values)
                }
            )

        await self._community_reports.upsert(community_datas)
        
    

    async def _form_single_community_report(self, er_graph, community, already_reports: dict[str, CommunityReportsResult]) -> dict:

        describe = await self._pack_single_community_describe(er_graph, community, already_reports=already_reports)
        prompt = CommunityPrompt.COMMUNITY_REPORT.format(input_text=describe)

        response = await self.llm.aask(prompt)
        data = prase_json_from_response(response)
        
        return data

    async def _pack_single_community_by_sub_communities(
        self,
        community,
        max_token_size: int,
        already_reports: dict[str, CommunityReportsResult],
    ) -> tuple[str, int]:
        
        """Pack a single community by summarizing its sub-communities."""
        all_sub_communities = [
            already_reports[k] for k in community.sub_communities if k in already_reports
        ]
        all_sub_communities = sorted(
            all_sub_communities, key=lambda x: x.occurrence, reverse=True
        )
        truncated_sub_communities = truncate_list_by_token_size(
            all_sub_communities,
            key=lambda x: x["report_string"],
            max_token_size=max_token_size,
        )
        sub_fields = ["id", "report", "rating", "importance"]
        sub_communities_describe = list_to_quoted_csv_string(
            [sub_fields]
            + [
                [
                    i,
                    c["report_string"],
                    c["report_json"].get("rating", -1),
                    c["occurrence"],
                ]
                for i, c in enumerate(truncated_sub_communities)
            ]
        )
        already_nodes = set()
        already_edges = set()
        for c in truncated_sub_communities:
            already_nodes.update(c.nodes)
            already_edges.update([tuple(e) for e in c.edges])
        return (
            sub_communities_describe,
            len(encode_string_by_tiktoken(sub_communities_describe)),
            already_nodes,
            already_edges,
        )

    async def _pack_single_community_describe(
        self, er_graph: BaseGraphStorage, community: dict, max_token_size: int = 12000, already_reports: dict = {}
    ) -> str:
        """Generate a detailed description of the community based on its attributes and existing reports."""
        nodes_in_order = sorted(community.nodes)
        edges_in_order = sorted(community.edges, key=lambda x: x[0] + x[1])

        nodes_data = await asyncio.gather(*[er_graph.get_node(n) for n in nodes_in_order])
        edges_data = await asyncio.gather(*[er_graph.get_edge(src, tgt) for src, tgt in edges_in_order])

        node_fields = ["id", "entity", "type", "description", "degree"]
        edge_fields = ["id", "source", "target", "description", "rank"]

        nodes_list_data = [
            [
                i,
                node_name,
                node_data.get("entity_type", "UNKNOWN"),
                node_data.get("description", "UNKNOWN"),
                await er_graph.node_degree(node_name),
            ]
            for i, (node_name, node_data) in enumerate(zip(nodes_in_order, nodes_data))
        ]
        nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
        nodes_may_truncate_list_data = truncate_list_by_token_size(
            nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
        )

        edges_list_data = [
            [
                i,
                edge_name[0],
                edge_name[1],
                edge_data.get("description", "UNKNOWN"),
                await er_graph.edge_degree(*edge_name),
            ]
            for i, (edge_name, edge_data) in enumerate(zip(edges_in_order, edges_data))
        ]
        edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
        edges_may_truncate_list_data = truncate_list_by_token_size(
            edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
        )

        truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(edges_list_data) > len(edges_may_truncate_list_data)
        report_describe = ""
        need_to_use_sub_communities = truncated and len(community.sub_communities) and len(already_reports)

        if need_to_use_sub_communities or self.enforce_sub_communities:
            logger.info(f"Community {community.title} exceeds the limit or force_to_use_sub_communities is True, using sub-communities")
            report_describe, report_size, contain_nodes, contain_edges = await self._pack_single_community_by_sub_communities(
                community, max_token_size, already_reports
            )

            report_exclude_nodes_list_data = [n for n in nodes_list_data if n[1] not in contain_nodes]
            report_include_nodes_list_data = [n for n in nodes_list_data if n[1] in contain_nodes]
            report_exclude_edges_list_data = [e for e in edges_list_data if (e[1], e[2]) not in contain_edges]
            report_include_edges_list_data = [e for e in edges_list_data if (e[1], e[2]) in contain_edges]

            nodes_may_truncate_list_data = truncate_list_by_token_size(
                report_exclude_nodes_list_data + report_include_nodes_list_data,
                key=lambda x: x[3],
                max_token_size=(max_token_size - report_size) // 2,
            )
            edges_may_truncate_list_data = truncate_list_by_token_size(
                report_exclude_edges_list_data + report_include_edges_list_data,
                key=lambda x: x[3],
                max_token_size=(max_token_size - report_size) // 2,
            )

        nodes_describe = list_to_quoted_csv_string([node_fields] + nodes_may_truncate_list_data)
        edges_describe = list_to_quoted_csv_string([edge_fields] + edges_may_truncate_list_data)

        return f"""-----Reports-----
            ```csv
            {report_describe}
            ```
            -----Entities-----
            ```csv
            {nodes_describe}
            ```
            -----Relationships-----
            ```csv
            {edges_describe}
        ```"""
