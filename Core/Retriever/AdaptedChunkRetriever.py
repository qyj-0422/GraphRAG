# AdaptedChunkRetriever.py
from Core.Retriever.ChunkRetriever import ChunkRetriever


class AdaptedChunkRetriever(ChunkRetriever):
    """继承ChunkRetriever，优化已有图数据的检索逻辑"""

    @register_retriever_method(type="chunk", method_name="ppr")
    async def _find_relevant_chunks_by_ppr(self, query, seed_entities: list[dict], link_entity=False):
        if self.config.use_existing_graph:
            # 已有图数据的检索逻辑
            node_ppr_matrix = await self._run_personalized_pagerank(query, seed_entities)
            relevant_nodes = self.graph.get_top_nodes(node_ppr_matrix, top_k=self.config.top_k)
            context = [self._format_node_context(node) for node in relevant_nodes]
            return context, [node.score for node in relevant_nodes]
        else:
            # 保持原有逻辑
            return await super()._find_relevant_chunks_by_ppr(query, seed_entities, link_entity)

    def _format_node_context(self, node):
        """从图节点提取上下文信息（适配已有图结构）"""
        # 根据实际图结构调整提取逻辑
        node_data = self.graph._graph.nodes[node.id]
        neighbors = list(self.graph._graph.neighbors(node.id))
        relations = [
            f"{node.id} -{self.graph._graph[node.id][neighbor].get('label', 'rel')}-> {neighbor}"
            for neighbor in neighbors
        ]
        return {
            "node_id": node.id,
            "attributes": node_data,
            "relations": relations
        }