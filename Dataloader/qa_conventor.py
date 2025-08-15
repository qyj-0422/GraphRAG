import json
import re

def load_id_mapping(mapping_file):
    """Load Qxxxx to entity name mapping"""
    id2entity = {}
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                id2entity[parts[0]] = parts[1]
    return id2entity

def count_q_ids(text):
    """Count Q IDs in text (returns 0 for non-strings)"""
    return len(re.findall(r'Q\d+', text)) if isinstance(text, str) else 0

def replace_single_q_id(text, id2entity):
    """
    Replace Q ID in text if exactly one exists
    Returns (modified_text, replaced_flag)
    """
    if not isinstance(text, str):
        return text, False
        
    q_ids = re.findall(r'Q\d+', text)
    if len(q_ids) == 1:
        q_id = q_ids[0]
        if q_id in id2entity:
            # Replace all occurrences of this Q ID (though there should be only one)
            return re.sub(re.escape(q_id), id2entity[q_id], text), True
    return text, False

def process_json_lines(input_file, output_file, id2entity):
    """
    Process JSON Lines file with:
    1. Filtering: keep only items with exactly one Q ID in answer
    2. Replacement: replace that Q ID with entity name
    """
    kept_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            total_count += 1
            
            try:
                item = json.loads(line)
                if 'answer' not in item:
                    continue
                    
                # Check and replace in answer field
                answer_text = item['answer']
                new_answer, replaced = replace_single_q_id(answer_text, id2entity)
                
                # Only keep items where:
                # 1. Answer had exactly one Q ID
                # 2. That Q ID was found in our mapping
                if replaced:
                    item['answer'] = new_answer
                    # Also process other fields (without filtering)
                    for field in ['question', 'label']:
                        if field in item:
                            item[field], _ = replace_single_q_id(item[field], id2entity)
                    
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    kept_count += 1
                    
            except json.JSONDecodeError:
                continue
    
    print(f"Processed {total_count} items, kept {kept_count} valid items")
    print(f"Filtered output saved to {output_file}")

# Usage example
if __name__ == "__main__":
    # Load ID mapping
    id2entity = load_id_mapping('./Data/middle_stage_data/wd_id2entity_text.txt')
    
    # Process JSON Lines file
    process_json_lines(
        input_file='./Data/middle_stage_data/Question.json',
        output_file='filtered_and_replaced.jsonl',
        id2entity=id2entity
    )
