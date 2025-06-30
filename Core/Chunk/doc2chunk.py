import pickle as pkl


def build_doc2chunk_mapping(chunks):
    doc2chunk = {}
    for chunk in chunks:
        doc_id = chunk['doc_id']
        chunk_id = chunk['chunk_id'] # 注意，此处的chunk-id有"chunk-"的前缀
        if doc_id not in doc2chunk:
            doc2chunk[doc_id] = []
        doc2chunk[doc_id].append(chunk_id)
        with open('doc2chunk.pkl', 'wb') as f:
            pkl.dump(doc2chunk, f)

