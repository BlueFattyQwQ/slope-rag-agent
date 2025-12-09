import os
import pickle
import jieba
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from elasticsearch import Elasticsearch
from app.index.base import BaseIndex
from app.ingest.parser import DocumentChunk
from app.core.config import settings
from app.core.logging import logger

class BM25Index(BaseIndex):
    def __init__(self):
        self.use_es = False
        self.es_client = None
        self.bm25_local = None
        self.documents = [] # 本地模式下存储
        
        if settings.ELASTICSEARCH_URL:
            try:
                self.es_client = Elasticsearch(settings.ELASTICSEARCH_URL)
                if self.es_client.ping():
                    self.use_es = True
                    self._init_es_index()
                    logger.info("Using Elasticsearch for BM25.")
                else:
                    logger.warning("Elasticsearch not reachable, falling back to local Rank-BM25.")
            except Exception as e:
                logger.warning(f"Failed to connect to Elasticsearch: {e}, falling back to local Rank-BM25.")
        else:
            logger.info("Elasticsearch URL not set, using local Rank-BM25.")

    def _init_es_index(self):
        if not self.es_client.indices.exists(index="slope_docs"):
            self.es_client.indices.create(index="slope_docs", body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "ik_smart_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard" # 实际应使用 ik_smart，这里简化使用 standard 或 smartcn
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": "standard"}, # 假设 ES 有中文分词插件，否则 standard 效果一般
                        "doc_id": {"type": "keyword"},
                        "page": {"type": "integer"}
                    }
                }
            })

    def _tokenize(self, text: str) -> List[str]:
        return list(jieba.cut_for_search(text))

    def add_documents(self, documents: List[DocumentChunk]):
        if self.use_es:
            for doc in documents:
                self.es_client.index(index="slope_docs", document=doc.to_dict())
            self.es_client.indices.refresh(index="slope_docs")
        else:
            self.documents.extend(documents)
            tokenized_corpus = [self._tokenize(doc.text) for doc in documents]
            # 重新构建 BM25 对象 (Rank-BM25 不支持增量更新，需全量重构)
            # 实际生产中应优化，这里简化
            self.bm25_local = BM25Okapi(tokenized_corpus)
        
        logger.info(f"Added {len(documents)} documents to BM25 index (ES={self.use_es}).")

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.use_es:
            resp = self.es_client.search(index="slope_docs", body={
                "query": {
                    "match": {
                        "text": query
                    }
                },
                "size": k
            })
            results = []
            for hit in resp['hits']['hits']:
                doc_data = hit['_source']
                # 重建 DocumentChunk 对象
                doc = DocumentChunk(**doc_data)
                score = hit['_score']
                results.append((doc, score))
            return results
        else:
            if not self.bm25_local:
                return []
            tokenized_query = self._tokenize(query)
            # get_top_n 只能返回文档，不能返回分数，我们需要分数进行融合
            # 所以手动计算
            scores = self.bm25_local.get_scores(tokenized_query)
            top_n_indices = np.argsort(scores)[::-1][:k]
            
            results = []
            for idx in top_n_indices:
                if scores[idx] > 0: # 过滤掉 0 分
                    results.append((self.documents[idx], float(scores[idx])))
            return results

    def save(self, path: str):
        if not self.use_es:
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "bm25_docs.pkl"), "wb") as f:
                pickle.dump(self.documents, f)
            # BM25 对象本身很难序列化，通常重新构建
            logger.info(f"Saved local BM25 documents to {path}")

    def load(self, path: str):
        if not self.use_es:
            docs_path = os.path.join(path, "bm25_docs.pkl")
            if os.path.exists(docs_path):
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)
                tokenized_corpus = [self._tokenize(doc.text) for doc in self.documents]
                self.bm25_local = BM25Okapi(tokenized_corpus)
                logger.info(f"Loaded local BM25 index from {path}")
