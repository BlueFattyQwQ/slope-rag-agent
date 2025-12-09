from typing import List
from sentence_transformers import CrossEncoder
from app.ingest.parser import DocumentChunk
from app.core.config import settings
from app.core.logging import logger

class Reranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info(f"Loading reranker model: {settings.RERANKER_MODEL_ID}")
        self.model = CrossEncoder(
            settings.RERANKER_MODEL_ID, 
            device=settings.DEVICE,
            max_length=512
        )

    def rerank(self, query: str, documents: List[DocumentChunk], top_n: int = 5) -> List[DocumentChunk]:
        if not documents:
            return []
            
        pairs = [[query, doc.text] for doc in documents]
        scores = self.model.predict(pairs)
        
        # 结合文档和分数
        doc_scores = list(zip(documents, scores))
        
        # 排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 取 Top N
        top_docs = [doc for doc, score in doc_scores[:top_n]]
        
        logger.info(f"Reranked {len(documents)} docs, returning top {top_n}. Top score: {doc_scores[0][1] if doc_scores else 0}")
        return top_docs

reranker = Reranker()
