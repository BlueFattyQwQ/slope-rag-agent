from typing import List, Set

def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k.intersection(relevant_ids))
    return hits / len(relevant_ids)

def calculate_mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    import numpy as np
    dcg = 0.0
    idcg = 0.0
    
    for i in range(min(len(retrieved_ids), k)):
        if retrieved_ids[i] in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
            
    for i in range(min(len(relevant_ids), k)):
        idcg += 1.0 / np.log2(i + 2)
        
    if idcg == 0:
        return 0.0
    return dcg / idcg

# 简化的三元评估 (Mock 实现，真实需要 LLM 判定)
def evaluate_faithfulness(answer: str, context: str) -> float:
    # 简单检查：答案中的关键词是否在上下文中出现
    # 实际应使用 LLM: "Is the answer derived from the context?"
    return 0.8 # Mock

def evaluate_context_precision(relevant_chunks: int, total_chunks: int) -> float:
    if total_chunks == 0: return 0
    return relevant_chunks / total_chunks

def evaluate_answer_relevancy(answer: str, query: str) -> float:
    # 实际应计算 Embedding Cosine Similarity
    return 0.85 # Mock
