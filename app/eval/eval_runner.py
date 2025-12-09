import json
import os
from app.pipeline.rag_pipeline import rag_pipeline
from app.eval.metrics import calculate_recall_at_k, calculate_mrr, calculate_ndcg
from app.core.logging import logger

def run_eval(questions_file: str = "eval/questions.jsonl", output_dir: str = "outputs"):
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    metrics_summary = {
        "recall@1": [], "recall@3": [], "recall@5": [],
        "mrr": [], "ndcg@5": []
    }
    
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line in f:
            q_data = json.loads(line)
            query = q_data['question']
            gold_citations = set([f"{c['doc_id']}:{c['page']}" for c in q_data['answers']])
            
            # 运行 Pipeline
            # 注意：为了评估检索，我们需要 Pipeline 暴露检索结果，或者修改 Pipeline 返回中间结果
            # 这里假设我们只评估最终引用的准确性作为检索指标的近似，或者修改 Pipeline
            # 为了演示，我们直接调用 retriever
            
            retrieved_docs = rag_pipeline.retriever.retrieve(query, k=50)
            reranked_docs = rag_pipeline.retriever.vector_index.search(query, k=50) # 简化，直接用 vector 结果演示
            # 实际上应该用 rerank 后的结果
            from app.search.rerank import reranker
            reranked_docs = reranker.rerank(query, retrieved_docs, top_n=5)
            
            retrieved_ids = [f"{d.doc_id}:{d.page}" for d in reranked_docs]
            
            # 计算指标
            r1 = calculate_recall_at_k(retrieved_ids, gold_citations, 1)
            r3 = calculate_recall_at_k(retrieved_ids, gold_citations, 3)
            r5 = calculate_recall_at_k(retrieved_ids, gold_citations, 5)
            mrr = calculate_mrr(retrieved_ids, gold_citations)
            ndcg = calculate_ndcg(retrieved_ids, gold_citations, 5)
            
            metrics_summary["recall@1"].append(r1)
            metrics_summary["recall@3"].append(r3)
            metrics_summary["recall@5"].append(r5)
            metrics_summary["mrr"].append(mrr)
            metrics_summary["ndcg@5"].append(ndcg)
            
            # 运行完整生成
            response = rag_pipeline.run(query)
            
            results.append({
                "query": query,
                "gold_citations": list(gold_citations),
                "retrieved_ids": retrieved_ids,
                "response": response,
                "metrics": {"r@5": r5, "mrr": mrr, "ndcg": ndcg}
            })
            
            print(f"Query: {query} | R@5: {r5:.2f} | MRR: {mrr:.2f}")

    # 计算平均值
    avg_metrics = {k: sum(v)/len(v) if v else 0 for k, v in metrics_summary.items()}
    
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding='utf-8') as f:
        json.dump({"summary": avg_metrics, "details": results}, f, indent=2, ensure_ascii=False)
        
    print("\nEvaluation Summary:")
    print(json.dumps(avg_metrics, indent=2))

if __name__ == "__main__":
    # 创建示例问题文件如果不存在
    if not os.path.exists("eval/questions.jsonl"):
        with open("eval/questions.jsonl", "w", encoding='utf-8') as f:
            sample = {
                "question": "边坡稳定性分析中，如何考虑降雨的影响？",
                "answers": [{"doc_id": "sample.pdf", "page": 1}]
            }
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            
    run_eval()
