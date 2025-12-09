import logging
import sys
from typing import Any

def setup_logging(name: str = "slope_rag") -> logging.Logger:
    """
    配置结构化日志
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

logger = setup_logging()

def log_retrieval_metrics(query: str, initial_count: int, reranked_count: int, scores: list[float]):
    """
    记录检索过程的关键指标
    """
    logger.info(f"Query: {query} | Initial Docs: {initial_count} | Reranked Docs: {reranked_count} | Top Scores: {scores[:3]}")
