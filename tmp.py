import json
import sys
from collections import defaultdict
import pickle as pkl

def analyze_json(data, parent_key='', stats=None):
    """
    Recursively analyze JSON data and count key-value pairs
    
    Args:
        data: JSON data to analyze
        parent_key: Current key path (for nested structures)
        stats: Dictionary to store statistics
    
    Returns:
        Dictionary containing analysis results
    """
    if stats is None:
        stats = defaultdict(int)
    
    if isinstance(data, dict):
        stats['total_dicts'] += 1
        for key, value in data.items():
            current_path = f"{parent_key}.{key}" if parent_key else key
            stats['total_keys'] += 1
            # Calculate size of the key
            stats['keys_size'] += sys.getsizeof(key)
            analyze_json(value, current_path, stats)
    elif isinstance(data, list):
        stats['total_lists'] += 1
        for idx, item in enumerate(data):
            analyze_json(item, f"{parent_key}[{idx}]", stats)
    else:
        # Primitive value (string, number, bool, null)
        stats['total_values'] += 1
        stats['values_size'] += sys.getsizeof(data)
    
    return stats

def print_stats(stats):
    """Print formatted statistics"""
    print("\nJSON Structure Analysis:")
    print(f"Total keys: {stats['total_keys']}")
    print(f"Total values: {stats['total_values']}")
    print(f"Total dictionaries: {stats['total_dicts']}")
    print(f"Total lists: {stats['total_lists']}")
    print("\nMemory Usage:")
    print(f"Keys size: {stats['keys_size'] / 1024:.2f} KB")
    print(f"Values size: {stats['values_size'] / 1024:.2f} KB")
    print(f"Estimated total size: {(stats['keys_size'] + stats['values_size']) / 1024:.2f} KB")

def read_pkl_file(file_path):
    """Read a pickle file and return its content"""
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def check_doc_id_exist(node2doc, doc2chunk):
    docid_list = list(doc2chunk.keys())
    for doc_ids in node2doc.values():
        for doc_id in doc_ids:
            if doc_id not in docid_list:
                print('doc_id not exist')
                return False
    print('全部存在')
    return True
            


# Example usage:
if __name__ == "__main__":
    # node2doc =read_pkl_file('./Data/yago/node2doc.pkl')
    # doc2chunk = read_pkl_file('./Data/yago/doc2chunk.pkl')
    # check_doc_id_exist(node2doc, doc2chunk)

    with open('./Data/yago/node2chunk.pkl', 'rb') as f:
        node2chunk = pkl.load(f)
    with open('./Data/yago/doc2chunk.pkl', 'rb') as f:
        doc2chunk = pkl.load(f)
    print('end')
