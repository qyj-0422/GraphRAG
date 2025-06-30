# 此处想要实现一个图转图的算法
# 输入图：由yago数据建立的简略图，见yago/er_graph/nx_data.graphml
# 输入数据：被chunk后的数据，其实也是前面的输入图构建的

# 输出图：带chunkid的输出图，格式类似于./working_dir/mix/er_graph/graph_storage_nx_data.graphml


from Core.Chunk.DocChunk import DocChunk
from Core.Storage.NetworkXStorage import NetworkXStorage
import networkx as nx
import asyncio
import pickle as pkl
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Storage.NameSpace import Workspace
from Core.Common.Logger import logger

async def convert_simple_to_detailed_graph(
    simple_graph_path: str,
    default_weight: float = 1.0
) -> NetworkXStorage:
    """
    Convert a simple graph to detailed graph by enriching with chunk information
    
    Args:
        simple_graph_path: Path to the simple .graphml file
        doc_chunk: Initialized DocChunk instance with loaded chunks
        default_weight: Default edge weight if not specified
    
    Returns:
        NetworkXStorage instance with detailed graph
    """
    # 1. Load simple graph
    try:
        simple_graph = nx.read_graphml(simple_graph_path)
    except Exception as e:
        raise NameError(f"Failed to load graph from {simple_graph_path}: {e}")
    
    # 2. Initialize detailed graph storage
    detailed_storage = NetworkXStorage()

    workspace = Workspace('./working_dir', 'yago')
    namespace = workspace.make_for("graph_storage")

    detailed_storage.namespace = namespace
    
    # 3. Process nodes
    
    relation2chunk_path = "./Data/yago/relation2chunk.pkl"
    node2chunk_path = "./Data/yago/node2chunk.pkl"

    with open(relation2chunk_path, 'rb') as f:
        relation2chunk = pkl.load(f)

    with open(node2chunk_path, 'rb') as f:
        node2chunk = pkl.load(f)
    
    for node_id in simple_graph.nodes():
        # Get or generate chunk ID for each node
        chunk_ids = node2chunk.get(node_id) # 此处得到的应该是一个list类型
        # TODO: 将list转变为正常str，然后作为source_id -- 参考BaseGraph中的MergeEntity
        if not chunk_ids:
            raise ValueError(f"No chunk found for node {node_id}")
        
        # Add node with detailed attributes
        await detailed_storage.upsert_node(node_id, {
            "entity_name": node_id,
            "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
            "entity_type": "",
            "description": ""
        })
    
    # 4. Process edges
    for src, tgt, data in simple_graph.edges(data=True):
        relation_name = data.get('relation_name', 'related_to')
        
        # Generate chunk ID for edge
        chunk_ids = relation2chunk.get(relation_name)
        # TODO: 将list转变为正常str，然后作为source_id -- 参考BaseGraph中的MergeEntity
        if not chunk_ids:
            raise ValueError(f"No chunk found for edge {relation_name}")
        
        # Add edge with detailed attributes
        await detailed_storage.upsert_edge(
            src, tgt, {
                "src_id": src,
                "tgt_id": tgt,
                "source_id": GRAPH_FIELD_SEP.join(chunk_ids),
                "weight": default_weight,
                "relation_name": relation_name,
                "keywords": "",
                "description": ""
            }
        )
    
    # 5. Persist the detailed graph
    await detailed_storage.persist(force=True)
    return detailed_storage

# Example usage
async def build_detailed_graph():
    
    # Convert graph
    detailed_storage = await convert_simple_to_detailed_graph(
        "./yago/er_graph/nx_data.graphml"
    )
    logger.info(f"Detailed graph saved to: {detailed_storage.graphml_xml_file}")

# asyncio.run(main())
