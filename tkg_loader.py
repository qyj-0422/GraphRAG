import json
import os
import pickle as pkl

from Core.Graph.ERGraph import ERGraph
from Core.Schema.GraphSchema import ERGraphSchema
from Core.Schema.EntityRelation import Entity, Relationship
import networkx as nx
from Core.Storage.NetworkXStorage import NetworkXStorage

def read_entity2id(file_path):
    """
    读取 entity2id.json 文件，返回实体名到 ID 的映射
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_relation2id(file_path):
    """
    读取 relation2id.json 文件，返回关系名到 ID 的映射
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_edges(file_paths, entity2id, relation2id):
    """
    读取多个边集文件，根据映射将边集中的 ID 转换为实体名和关系名
    """
    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    graph = nx.Graph()

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src_id, relation_id, tgt_id, _, _ = line.strip().split()
                src_entity = id2entity[int(src_id)]
                tgt_entity = id2entity[int(tgt_id)]
                relation_name = id2relation[int(relation_id)]

                # 添加节点
                if not graph.has_node(src_entity):
                    graph.add_node(src_entity)
                if not graph.has_node(tgt_entity):
                    graph.add_node(tgt_entity)
                # 添加边
                graph.add_edge(src_entity, tgt_entity, relation_name=relation_name)
    return graph

def save_graph_to_graphml(graph, namespace):
    """
    使用 NetworkXStorage 类将图保存为 .graphml 文件
    """
    storage = NetworkXStorage()
    storage.namespace = namespace
    storage._graph = graph
    import asyncio
    asyncio.run(storage.persist(force=True))

# 示例调用
data_path = "./Data/yago/"
edge_file_paths = ["train.txt", "test.txt", "valid.txt"]  # 可以添加更多边集文件路径
for i in range(len(edge_file_paths)):
    edge_file_paths[i] = data_path + edge_file_paths[i]
entity2id_path = data_path + "entity2id.json"
relation2id_path = data_path + "relation2id.json"

# 定义保存路径（建议与目标目录同级或自定义）
tgt_path = "./yago"
output_path = os.path.join(tgt_path, "yago_nx_data.graphml")

# 读取映射文件
entity2id = read_entity2id(entity2id_path)
relation2id = read_relation2id(relation2id_path)

# 读取边集文件并构建图
graph = read_edges(edge_file_paths, entity2id, relation2id)

# 假设 namespace 是一个具有 get_save_path 方法的对象
class MockNamespace:
    def get_save_path(self, name=output_path):
        return os.path.join(os.getcwd(), name)

namespace = MockNamespace()

# 保存图为 .graphml 文件
save_graph_to_graphml(graph, namespace)