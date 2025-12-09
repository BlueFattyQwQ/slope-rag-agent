import json
import re
from typing import Dict, Any, List
from app.search.retrieve import HybridRetriever
from app.search.rerank import reranker
from app.llm.generator import llm_generator
from app.prompt.prompt_builder import prompt_builder
from app.utils.citations import validate_citations
from app.core.config import settings
from app.core.logging import logger
from app.tools.weather import weather_tool
from app.tools.engineering import engineering_tool

class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()

    def run(self, query: str) -> Dict[str, Any]:
        logger.info(f"Starting RAG pipeline for query: {query}")
        
        # 0. 工具调用检查 (简单关键词触发，实际应由 LLM 决定)
        if "天气" in query or "降雨" in query:
            # 简单提取城市，默认 A区
            weather_info = weather_tool.query("Area A")
            logger.info(f"Tool used: Weather - {weather_info}")
            # 将工具结果拼接到 Query 中
            query += f" (当前天气状况: {json.dumps(weather_info, ensure_ascii=False)})"

        if "计算" in query and "安全系数" in query:
            # 模拟参数提取
            calc_res = engineering_tool.stability_factor(c=20, phi=30, gamma=18, h=10, beta=45)
            logger.info(f"Tool used: Engineering - {calc_res}")
            query += f" (计算参考: {json.dumps(calc_res, ensure_ascii=False)})"

        # 1. 检索
        retrieved_docs = self.retriever.retrieve(query, k=settings.RETRIEVE_K)
        
        # 2. 重排序
        reranked_docs = reranker.rerank(query, retrieved_docs, top_n=settings.RERANK_TOPN)
        
        # 3. 构建 Prompt
        prompt = prompt_builder.build_prompt(query, reranked_docs)
        
        # 4. LLM 生成
        raw_response = llm_generator.generate(prompt)
        
        # 5. 解析 JSON
        try:
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
            else:
                # Fallback if no JSON found
                response_json = {
                    "risk_level": "unknown", 
                    "rationale": raw_response, 
                    "citations": [], 
                    "recommendations": []
                }
        except json.JSONDecodeError:
             response_json = {
                "risk_level": "unknown", 
                "rationale": raw_response, 
                "citations": [], 
                "recommendations": []
            }

        # 6. 引用校验
        final_response = validate_citations(response_json, reranked_docs)
        
        # 添加证据摘要用于前端展示
        final_response["evidence"] = [
            {"doc_id": d.doc_id, "page": d.page, "snippet": d.text[:200] + "..."} 
            for d in reranked_docs
        ]
        
        return final_response

rag_pipeline = RAGPipeline()
