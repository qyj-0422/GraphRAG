# 目前已经有三个pkl文件，都在/Data/yago/目录下，一个是node2doc、relation2doc，还有一个doc2chunk文件
# 输入：以上三个文件
# 输出：node2chunk、relation2chunk
import os
import pickle as pkl
from collections import defaultdict

def read_pkl_file(file_path):
    """Read a pickle file and return its content"""
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    

def build_mappings():
    node2doc =read_pkl_file('./Data/yago/node2doc.pkl')
    relation2doc = read_pkl_file('./Data/yago/relation2doc.pkl')
    doc2chunk = read_pkl_file('./Data/yago/doc2chunk.pkl')

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
    with open('./Data/yago/node2chunk.pkl', 'wb') as f:
        pkl.dump(node2chunk, f)
    with open('./Data/yago/relation2chunk.pkl', 'wb') as f:
        pkl.dump(relation2chunk, f)
    
if __name__ == '__main__':
    build_mappings()