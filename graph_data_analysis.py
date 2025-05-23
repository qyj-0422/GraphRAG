import pickle as pkl
import pdb

with open('./cs/tree_graph_balanced/graph_storage_tree_data.pkl', 'rb') as f:
    data = pkl.load(f)
    pdb.set_trace()
    print('load successfully')