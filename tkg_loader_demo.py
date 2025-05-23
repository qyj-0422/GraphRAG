import pickle as pkl
from Core.Schema.GraphSchema import ERGraphSchema
from Core.Schema.EntityRelation import Entity, Relationship

# 示例节点数据
node_data = [
    {
        "entity_name": "Node1",
        "source_id": "source1",
        "entity_type": "Type1",
        "description": "Description of Node1"
    },
    {
        "entity_name": "Node2",
        "source_id": "source2",
        "entity_type": "Type2",
        "description": "Description of Node2"
    }
]

# 示例边数据
edge_data = [
    {
        "src_id": "Node1",
        "tgt_id": "Node2",
        "source_id": "source1",
        "relation_name": "RelatedTo",
        "weight": 1.0,
        "description": "Relation between Node1 and Node2",
        "keywords": "",
        "rank": 0
    }
]

# 将节点数据转换为 Entity 对象列表
nodes = [Entity(**node) for node in node_data]

# 将边数据转换为 Relationship 对象列表
edges = [Relationship(**edge) for edge in edge_data]

# 创建 ERGraphSchema 对象
er_graph_schema = ERGraphSchema(nodes=nodes, edges=edges)

# 打印 ERGraphSchema 对象的节点和边
print("Nodes:", [node.as_dict for node in er_graph_schema.nodes])
print("Edges:", [edge.as_dict for edge in er_graph_schema.edges])