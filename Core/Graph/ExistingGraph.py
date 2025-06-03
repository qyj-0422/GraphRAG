# Core/Graph/ExistingGraph.py
from Core.Graph.BaseGraph import BaseGraph
import networkx as nx
import json


class ExistingGraph(BaseGraph):
    """从外部文件加载已有图结构数据"""

    def __init__(self, config, llm, encoder):
        super().__init__(config, llm, encoder)
        self.graph_path = './yago/er_graph/nx_data.graphml'
        self._graph = self._load_from_file(self.graph_path)

    def _load_from_file(self, graph_path):
        """根据文件格式加载对应类型的图数据"""
        if graph_path.endswith(('.gml', '.graphml')):
            return nx.read_graphml(graph_path)
        elif graph_path.endswith('.json'):
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            return self._parse_json_graph(graph_data)
        elif graph_path.endswith(('.ttl', '.rdf')):
            # 支持RDF格式，需安装rdflib
            from rdflib import Graph as RDFGraph
            rdf_graph = RDFGraph()
            rdf_graph.parse(graph_path)
            return self._convert_rdf_to_networkx(rdf_graph)
        else:
            raise ValueError(f"不支持的图文件格式: {graph_path}")

    def _parse_json_graph(self, graph_data):
        """将JSON格式的图数据转换为NetworkX图"""
        G = nx.DiGraph()
        for node in graph_data.get('nodes', []):
            G.add_node(node['id'], **node.get('attributes', {}))
        for edge in graph_data.get('edges', []):
            G.add_edge(edge['source'], edge['target'], **edge.get('attributes', {}))
        return G

    def _convert_rdf_to_networkx(self, rdf_graph):
        """将RDF图转换为NetworkX图（简化版）"""
        G = nx.DiGraph()
        for s, p, o in rdf_graph:
            G.add_node(str(s), type="entity")
            G.add_node(str(o), type="entity" if isinstance(o, URIRef) else "literal")
            G.add_edge(str(s), str(o), label=str(p))
        return G

    # 重写无需LLM提取的方法
    async def _named_entity_recognition(self, passage: str):
        return []  # 已有图无需NER

    async def _build_graph(self, chunks, force=False):
        """已有图无需构建，直接返回"""
        return

    def _extract_entity_relationship(self):
        return

    @property
    def node_num(self):
        return self._graph.number_of_nodes()