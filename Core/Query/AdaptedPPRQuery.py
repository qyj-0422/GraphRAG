# AdaptedPPRQuery.py
from Core.Query.PPRQuery import PPRQuery


class AdaptedPPRQuery(PPRQuery):
    """继承PPRQuery，优化已有图数据的查询逻辑"""

    async def generation_qa(self, query, context):
        if self.config.use_existing_graph:
            # 已有图数据的上下文格式化
            graph_context = "\n".join([
                f"节点: {ctx['node_id']}; 属性: {ctx['attributes']}; 关系: {', '.join(ctx['relations'])}"
                for ctx in context
            ])
            msg = QueryPrompt.GENERATE_RESPONSE_WITH_GRAPH.format(
                query=query, graph_context=graph_context
            )
            return await self.llm.aask(msg=msg)
        else:
            # 保持原有逻辑
            return await super().generation_qa(query, context)