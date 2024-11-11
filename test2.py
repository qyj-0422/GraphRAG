from Core.Index import (
    get_rag_embedding,
    get_index
)    
from Core.Index.Schema import (
    # BM25RetrieverConfig,
    # CohereRerankConfig,
    # ColbertRerankConfig,
    FAISSIndexConfig,
)
from Core.Index.VectorIndex import VectorIndex
# print(get_rag_embedding())
# bb = FAISSRetrieverConfig(dimensions= 128)

aa = FAISSIndexConfig(persist_path = "./storage", embed_model= get_rag_embedding())
ss = VectorIndex(aa)
# bb = get_index(aa)
# print(bb)

import asyncio

async def main():
    aa = FAISSIndexConfig(persist_path="./storage", embed_model=get_rag_embedding())
    ss = VectorIndex(aa)
    # bb = get_index(aa)
    # print(bb)

    ss.build_index()
    res = await ss.retrieval("your mother", top_k=1)
    print(res)

# Call the async function
asyncio.run(main())