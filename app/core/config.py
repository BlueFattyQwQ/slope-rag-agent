import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # 模型路径
    SFT_MODEL_ID: str = "./models/sft_model"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-large-zh-v1.5"
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-large"
    
    # OpenAI 兼容接口
    OPENAI_BASE_URL: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # 检索服务
    ELASTICSEARCH_URL: Optional[str] = None
    
    # 工具 API
    WEATHER_API_URL: str = "https://api.weatherapi.com/v1"
    WEATHER_API_KEY: str = "mock_key"
    
    # 运行参数
    DEVICE: str = "cpu"
    MAX_INPUT_TOKENS: int = 2048
    MAX_OUTPUT_TOKENS: int = 1024
    MAX_CTX_TOKENS: int = 1500
    
    # 检索参数
    INDEX_BACKEND: str = "faiss"
    RERANK_TOPN: int = 5
    RETRIEVE_K: int = 50
    
    # 路径配置
    DATA_DIR: str = "data/sample_docs"
    INDEX_DIR: str = "data/index"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

# 确保目录存在
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)
