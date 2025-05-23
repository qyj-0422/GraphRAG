import json
import os
import pickle as pkl

from Core.Graph.ERGraph import ERGraph
from Core.Schema.GraphSchema import ERGraphSchema
from Core.Schema.EntityRelation import Entity, Relationship

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
    entities = set()
    relationships = []

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                src_id, relation_id, tgt_id, _, _ = line.strip().split()
                src_entity = id2entity[int(src_id)]
                tgt_entity = id2entity[int(tgt_id)]
                relation_name = id2relation[int(relation_id)]
                entities.add(src_entity)
                entities.add(tgt_entity)
                relationship = Relationship(
                    src_id=src_entity,
                    tgt_id=tgt_entity,
                    source_id="None",
                    relation_name=relation_name,
                    weight=1.0
                )
                relationships.append(relationship)
    return entities, relationships

def create_er_graph_schema(edge_file_paths, entity2id_path, relation2id_path):
    """
    从多个边集文件、entity2id.json 和 relation2id.json 中读取数据，创建 ERGraphSchema 对象
    """
    entity2id = read_entity2id(entity2id_path)
    relation2id = read_relation2id(relation2id_path)
    entities, relationships = read_edges(edge_file_paths, entity2id, relation2id)
    node_list = [Entity(entity_name=entity, source_id="combined_edge_files") for entity in entities]
    ERG = ERGraphSchema()
    ERG.nodes = node_list
    ERG.edges = relationships
    return ERG

# 示例调用
data_path = "./Data/yago/"
edge_file_paths = ["train.txt", "test.txt", "valid.txt"]  # 可以添加更多边集文件路径
for i in range(len(edge_file_paths)):
    edge_file_paths[i] = data_path + edge_file_paths[i]
entity2id_path = data_path + "entity2id.json"
relation2id_path = data_path + "relation2id.json"
er_graph_schema = create_er_graph_schema(edge_file_paths, entity2id_path, relation2id_path)

# 定义保存路径（建议与目标目录同级或自定义）
tgt_path = "./yago"
output_path = os.path.join(tgt_path, "er_graph_schema.pkl")

# 保存为 pkl 文件
with open(output_path, "wb") as f:
    pkl.dump(er_graph_schema, f)

print(f"ERGraphSchema 对象已保存至：{output_path}")