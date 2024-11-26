
import pandas as pd
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import pickle 
nns = 100
doc_list = """
1 \t sdsdsdsd \n
2 \t dsdsdsd 
"""
checkpoint_path = "./Tools/Index/colbertv2.0"

from Core.Index.Schema import ColBertIndexConfig
from Core.Index import get_index

data= ['"center for science and law"', '"karl deisseroth"', '"d h  chen foundation professor"', '"legal system with modern neuroscience"', '"integrated optical and genetic strategies"', '"neosensory"', '"neuroscience"', '"department of molecular and cellular physiology"', '"thomas c  sudhof"', '"german american biochemist"', '"medical practices and health systems"', '"dysfunction in neurological and psychiatric disease"', '"normal neural circuit function"', '"april 25  1971"', '"starmap"', '"department of neurology"', '"department of psychiatry and behavioral sciences"', '"optogenetics"', '"study of synaptic transmission"', '"december 22  1955"', '"american scientist"', '"american neuroscientist"', '"november 18  1971"', '"science communicator"', '"author"', '"braincheck"', '"school of medicine"', '"clarity"', '"stanford university"', '"david eagleman"', '"sensory substitution"']
# aa = ColbertIndex(model_name=checkpoint_path, index_name="nbits_2", nbits=2,  ranks=1, doc_maxlen=120, query_maxlen=60, kmeans_niters=4)
print("="*20)
def test_knn():
    # aa._build_index_from_list(data)
    # colbertIndex = aa.load_from_disk("./storage/colbert_index", index_name="nbits_2")
    config = ColBertIndexConfig(persist_path = "./storage/colbert_index", index_name="nbits_3", model_name=checkpoint_path, nbits=2,  ranks=1, doc_maxlen=120, query_maxlen=60, kmeans_niters=4)
    bb = get_index(config)
    print(bb)
    # bb._build_index_from_list(data)
    data = {
    'qid1': 'content1',
    'qid2': 'content2',
    'qid3': 'content3'
}
    import pdb
    pdb.set_trace()
    queries = Queries(data= data)
    import pdb
    pdb.set_trace()
    res = bb.query_batch(queries, top_k = 10)
    print(res)
    # with Run().context(RunConfig(nranks=1, experiment = "dasdsd", root="")):
    #         config = ColBERTConfig(
    #         nbits=1,
    #         root="/home/yingli/GraphRAG/storage/colbert_index",
    #     )
    #         indexer = Indexer(checkpoint=checkpoint_path, config=config)
    #         indexer.index(name="nbits_2", collection=data, overwrite=True)

        # retrieval
    # with Run().context(RunConfig(nranks=1, experiment="colbert", root="")):
    #     colbert_config = ColBERTConfig.load_from_index("/home/yingli/GraphRAG/colbert/indexes/nbits_2")
    #     searcher = Searcher(
    #         index="nbits_2", index_root="/home/yingli/GraphRAG/colbert/indexes", config=colbert_config
    #     )
    # #         config = ColBERTConfig(
    # #             root="data/lm_vectors/colbert",
    # #     )
    # #         searcher = Searcher(index=doc_list, config=config)
  
    #     queries = Queries(data = {"question": "sdsd"})
    #     ranking = searcher.search_all(queries, k=nns)
    #     print(ranking)
if __name__ == "__main__":
    test_knn()