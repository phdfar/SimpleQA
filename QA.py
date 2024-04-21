import argparse
import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers

import os
import pickle


class QAService:
    def __init__(self,modelname,source):

      self.modelname = modelname
      self.source = source
      self.pipe = None

       
    def build(self):
      
      logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
      logging.getLogger("haystack").setLevel(logging.INFO)
      document_store = InMemoryDocumentStore(use_bm25=True)


      files_to_index = [self.source + "/" + f for f in os.listdir(self.source) if '.txt' in f]
      indexing_pipeline = TextIndexingPipeline(document_store)
      indexing_pipeline.run_batch(file_paths=files_to_index)

      retriever = BM25Retriever(document_store=document_store)
      reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)


      self.pipe = ExtractiveQAPipeline(reader, retriever)

      with open(self.modelname, 'wb') as fp:
        pickle.dump(self.pipe, fp)

    def test(self,query,qmode='minimum'):
      
      if self.pipe==None:
        with open(self.modelname, 'rb') as fp:
          self.pipe = pickle.load(fp)
        
      prediction = self.pipe.run(
        query=query, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})

      print_answers(prediction, details=qmode)  ## Choose from `minimum`, `medium`, and `all`


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple QA")


    parser.add_argument("-build", action='store_true')
    parser.add_argument("-test", action='store_true')
    parser.add_argument("--modelname", type=str , help="Path to model")
    parser.add_argument("--source", type=str , help="folder path include .txt files")
    parser.add_argument("--query", type=str , help="your query")
    parser.add_argument("--qmode", type=str , default="minimum" , help="## Choose from `minimum`, `medium`, and `all` ")

    args = parser.parse_args()
    
    QA  = QAService(args.modelname,args.source)

    if args.build:
      QA.build()

    if args.test:
      QA.test(args.query,args.qmode)



    