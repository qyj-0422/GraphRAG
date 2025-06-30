import networkx as nx
import pickle as pkl
from Core.Common.Utils import mdhash_id
# 将图变为一个list，其中每一个项都是
# {
#     "title": corpus.iloc[i]["title"],
#     "content": corpus.iloc[i]["context"],
#     "doc_id": i,
# }
# 
# 
# 使其的类型与QueryDataset中的corpus_list一致

def graph2doc(graph_path = './yago/er_graph/nx_data.graphml'):
    # 将图转化为文档格式

    # 1. 读取.graphml文件
    G = nx.read_graphml(graph_path)  # 替换为实际文件路径

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
        # 输出结果
    file_path = './Data/yago/Corpus.pkl'
    with open(file_path, 'wb') as f:
        pkl.dump(corpus_list, f)

    node2doc_path = './Data/yago/node2doc.pkl'
    with open(node2doc_path, 'wb') as f:
        pkl.dump(node2doc, f)
    
    relation2doc_path = './Data/yago/relation2doc.pkl'
    with open(relation2doc_path, 'wb') as f:
        pkl.dump(relation2doc, f)
    print('done')

if __name__ == "__main__":
    graph2doc()