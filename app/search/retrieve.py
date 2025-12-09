from typing import List, Tuple
from app.index.faiss_index import FAISSIndex
from app.index.bm25 import BM25Index
from app.ingest.parser import DocumentChunk
from app.core.config import settings
from app.core.logging import logger

class HybridRetriever:
    def __init__(self):
        self.vector_index = FAISSIndex()
        self.bm25_index = BM25Index()
        
        # 尝试加载已有索引
        self.vector_index.load(settings.INDEX_DIR)
        self.bm25_index.load(settings.INDEX_DIR)

    def index_documents(self, documents: List[DocumentChunk]):
        self.vector_index.add_documents(documents)
        self.bm25_index.add_documents(documents)
        
        self.vector_index.save(settings.INDEX_DIR)
        self.bm25_index.save(settings.INDEX_DIR)

    def retrieve(self, query: str, k: int = 50) -> List[DocumentChunk]:
        """
        混合检索：向量检索 + BM25，使用 RRF 或 加权融合
        """
        # 1. 获取结果
        vector_results = self.vector_index.search(query, k=k)
        bm25_results = self.bm25_index.search(query, k=k)
        
        # 2. 归一化分数 (Min-Max Normalization)
        def normalize(results):
            if not results:
                return {}
            scores = [r[1] for r in results]
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return {r[0].text: 1.0 for r in results}
            return {r[0].text: (r[1] - min_s) / (max_s - min_s) for r in results}

        vec_norm = normalize(vector_results)
        bm25_norm = normalize(bm25_results)
        
        # 3. 融合 (加权求和: 0.7 Vector + 0.3 BM25)
        combined_scores = {}
        doc_map = {} # text -> DocumentChunk
        
        for doc, _ in vector_results:
            doc_map[doc.text] = doc
            combined_scores[doc.text] = combined_scores.get(doc.text, 0) + 0.7 * vec_norm.get(doc.text, 0)
            
        for doc, _ in bm25_results:
            doc_map[doc.text] = doc
            combined_scores[doc.text] = combined_scores.get(doc.text, 0) + 0.3 * bm25_norm.get(doc.text, 0)
            
        # 4. 排序并返回 Top K
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        final_docs = [doc_map[text] for text, score in sorted_docs[:k]]
        
        logger.info(f"Hybrid retrieval returned {len(final_docs)} docs for query: {query}")
        return final_docs
