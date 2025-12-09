import os
import pickle
import numpy as np
import faiss
from typing import List, Tuple
from app.index.base import BaseIndex
from app.ingest.parser import DocumentChunk
from app.llm.embedding import embedding_model
from app.core.logging import logger

class FAISSIndex(BaseIndex):
    def __init__(self):
        self.index = None
        self.documents = [] # 存储原始文档数据，FAISS 只存向量
        self.dimension = embedding_model.embedding_dim

    def _init_index(self, num_vectors: int):
        # 使用 IVF+PQ 以支持大规模数据，或者简单使用 FlatL2/IP
        # 这里为了演示简单且数据量不大，使用 IndexFlatIP (内积，归一化后等同于余弦相似度)
        # 如果数据量大，可以切换为:
        # quantizer = faiss.IndexFlatIP(self.dimension)
        # self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist=100, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexFlatIP(self.dimension)

    def add_documents(self, documents: List[DocumentChunk]):
        if not documents:
            return
            
        texts = [doc.text for doc in documents]
        embeddings = embedding_model.embed_documents(texts)
        
        if self.index is None:
            self._init_index(len(documents))
            
        # 如果是 IVF 索引，需要 train
        # if not self.index.is_trained:
        #     self.index.train(embeddings)
            
        self.index.add(embeddings)
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to FAISS index.")

    def search(self, query: str, k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_embedding = embedding_model.embed_query(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
                
        return results

    def save(self, path: str):
        if self.index is None:
            return
        
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str):
        index_path = os.path.join(path, "faiss.index")
        docs_path = os.path.join(path, "docs.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
            logger.info(f"Loaded FAISS index from {path}")
        else:
            logger.warning(f"Index files not found in {path}")
