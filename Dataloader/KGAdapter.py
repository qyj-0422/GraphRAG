import os
import pickle as pkl
import networkx as nx
from Core.Common.Utils import mdhash_id
from collections import defaultdict
from Core.Common.Logger import logger
from Core.Common.Constants import GRAPH_FIELD_SEP
from Core.Storage.NameSpace import Workspace
from Core.Storage.NetworkXStorage import NetworkXStorage


class KGAdapter:
    # 输入为nxGraph
    def __init__(self):
        self.kg_name = 'yago'
        self.data_path = './Data'
        self.working_dir = './working_dir'
        self.origin_graph_path = os.path.join(self.data_path, self.kg_name, 'nx_data.graphml')
        self.corpus_save_path = os.path.join(self.data_path, self.kg_name, 'Corpus.pkl')
        self.middle_stage_data_path = os.path.join(self.data_path, self.kg_name, 'middle_stage_data')
        self.default_weight = 1.0

    def kg2doc(self, force=False):
        # 将知识图谱转换为./Data/{kg_name}/Corpus.pkl文件，供QueryDataset直接使用，
        # 同时存储./Data/middlnode2doc和relation2doc
        node2doc_path = os.path.join(self.middle_stage_data_path,'node2doc.pkl')
        relation2doc_path = os.path.join(self.middle_stage_data_path,'relation2doc.pkl')
        if not force and os.path.exists(self.corpus_save_path) and os.path.exists(node2doc_path) and os.path.exists(relation2doc_path):
            logger.info('files already exists')
            return
        
        # 1. 读取.graphml文件
        G = nx.read_graphml(self.origin_graph_path)  # 替换为实际文件路径

        # 2. 遍历所有边并存储为一个文档
        corpus_list = []
        node2doc = {}
        relation2doc = {}
        for i, (source, target, data) in enumerate(G.edges(data=True)):
            # Get node names (handling possible encoding issues)
            source_name = G.nodes[source].get('id', source)  # 优先使用node的id属性
            target_name = G.nodes[target].get('id', target)
            doc = {
                    "title": data['relation_name'],
                    "content": source_name + " " + data['relation_name'] + " " + target_name,
                    "doc_id": i,
                    }
            corpus_list.append(doc)          
            doc_hash_id = mdhash_id(doc["content"].strip(), prefix="doc-")

            if source_name not in node2doc:
                node2doc[source_name] = []
            node2doc[source_name].append(doc_hash_id)

            if target_name not in node2doc:
                node2doc[target_name] = []
            node2doc[target_name].append(doc_hash_id)

            relation2doc[data['relation_name']] = doc_hash_id

        with open(self.corpus_save_path, 'wb') as f:
            pkl.dump(corpus_list, f)
        
        with open(node2doc_path, 'wb') as f:
            pkl.dump(node2doc, f)
        
        with open(relation2doc_path, 'wb') as f:
            pkl.dump(relation2doc, f)
        logger.info('Corpus.pkl generated')

    def kg2chunk(self, force=False):
        # 读取node2doc、relation2doc和doc2chunk，输出node2chunk和relation2chunk
        node2chunk_path = os.path.join(self.middle_stage_data_path,'node2chunk.pkl')
        relation2chunk_path = os.path.join(self.middle_stage_data_path,'relation2chunk.pkl')
        if not force and os.path.exists(node2chunk_path) and os.path.exists(relation2chunk_path):
            logger.info('files already exists')
            return

        with open(os.path.join(self.middle_stage_data_path,'node2doc.pkl'), 'rb') as f:
            node2doc = pkl.load(f)
        with open(os.path.join(self.middle_stage_data_path,'relation2doc.pkl'), 'rb') as f:
            relation2doc = pkl.load(f)
        with open(os.path.join(self.middle_stage_data_path,'doc2chunk.pkl'), 'rb') as f:
            doc2chunk = pkl.load(f)

        node2chunk = defaultdict(list)
        relation2chunk = defaultdict(list)
        # Build node2chunk mapping
        for node_id, doc_ids in node2doc.items():
            for doc_id in doc_ids:
                if doc_id in doc2chunk:
                    node2chunk[node_id].extend(doc2chunk[doc_id])
            # Remove duplicates while preserving order
            node2chunk[node_id] = list(dict.fromkeys(node2chunk[node_id]))
        
        # Build relation2chunk mapping
        for relation_id, doc_id in relation2doc.items():
            if doc_id in doc2chunk:
                relation2chunk[relation_id].extend(doc2chunk[doc_id])
            # Remove duplicates while preserving order
            relation2chunk[relation_id] = list(dict.fromkeys(relation2chunk[relation_id]))
        
        # Save the mappings
        with open(node2chunk_path, 'wb') as f:
            pkl.dump(node2chunk, f)
            logger.info('node2chunk.pkl generated')
        with open(relation2chunk_path, 'wb') as f:
            pkl.dump(relation2chunk, f)
            logger.info('relation2chunk.pkl generated')

    def doc2chunk(self, chunks):
        # 在建立chunk过程中调用，输入建立好的chunks，输出doc2chunk文件
        doc2chunk = {}
        for chunk in chunks:
            doc_id = chunk['doc_id']
            chunk_id = chunk['chunk_id'] # 注意，此处的chunk-id有"chunk-"的前缀
            if doc_id not in doc2chunk:
                doc2chunk[doc_id] = []
            doc2chunk[doc_id].append(chunk_id)
            doc2chunk_path = os.path.join(self.middle_stage_data_path,'doc2chunk.pkl')
            with open(doc2chunk_path, 'wb') as f:
                pkl.dump(doc2chunk, f)
    
    async def build_detailed_graph(self):
        """
        输入relation2chunk和node2chunk，输出./working_dir/{kg_name}/graph_storage_nx_data.graphml
        """
        # 1. Load simple graph
        try:
            simple_graph = nx.read_graphml(self.origin_graph_path)
        except Exception as e:
            raise NameError(f"Failed to load graph from {self.origin_graph_path}: {e}")
        
        # 2. Initialize detailed graph storage
        detailed_storage = NetworkXStorage()


        workspace = Workspace(self.working_dir, self.kg_name)
        namespace = workspace.make_for("graph_storage")

        detailed_storage.namespace = namespace
        
        # 3. Process nodes
        
        relation2chunk_path = os.path.join(self.middle_stage_data_path,'relation2chunk.pkl')
        node2chunk_path = os.path.join(self.middle_stage_data_path,'node2chunk.pkl')

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
                        "weight": self.default_weight,
                        "relation_name": relation_name,
                        "keywords": "",
                        "description": ""
                    }
                )
            
            # 5. Persist the detailed graph
            await detailed_storage.persist(force=True)
            logger.info(f"Detailed graph built and persisted")

            return detailed_storage
    
