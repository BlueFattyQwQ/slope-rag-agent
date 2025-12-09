from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.logging import logger

class EmbeddingModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_ID}")
        self.model = SentenceTransformer(
            settings.EMBEDDING_MODEL_ID, 
            device=settings.DEVICE
        )
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        """
        生成文档嵌入
        """
        if not texts:
            return np.array([])
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        生成查询嵌入 (BGE 模型通常需要为查询添加指令，但 sentence-transformers 封装可能已处理，
        或者 BGE-v1.5 不需要特定指令，视具体模型版本而定。这里按标准处理)
        """
        # BGE v1.5 推荐为查询添加指令: "为这个句子生成表示以用于检索相关文章："
        # 但如果是对称检索或短文本，直接 encode 也可以。这里加上通用指令。
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self.model.encode([instruction + query], normalize_embeddings=True)[0]

embedding_model = EmbeddingModel()
