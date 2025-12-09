from typing import List
import json
from app.ingest.parser import DocumentChunk

class PromptBuilder:
    def __init__(self):
        self.system_prompt = """你是一位资深的边坡工程顾问专家。请基于提供的上下文证据回答用户的问题。
请严格遵守以下规则：
1. 仅根据提供的上下文回答，不要编造信息。
2. 如果上下文不足以回答问题，请明确说明。
3. 输出必须是合法的 JSON 格式，包含 risk_level, rationale, citations, recommendations 字段。
4. citations 中的 doc_id 和 page 必须严格来自上下文。
"""

    def build_prompt(self, query: str, context_docs: List[DocumentChunk]) -> str:
        context_str = ""
        for i, doc in enumerate(context_docs):
            context_str += f"Evidence {i+1}:\n"
            context_str += f"Doc ID: {doc.doc_id}\n"
            context_str += f"Page: {doc.page}\n"
            context_str += f"Content: {doc.text}\n\n"
            
        user_prompt = f"""
Context:
{context_str}

Question: {query}

请以 JSON 格式输出回答，格式如下：
{{
    "risk_level": "low|medium|high",
    "rationale": "分析理由...",
    "citations": [{{"doc_id": "...", "page": 1}}],
    "recommendations": ["建议1", "建议2"]
}}
"""
        # 拼接完整 Prompt (适配 Chat 模型格式)
        # 这里简单拼接，实际应使用 tokenizer.apply_chat_template
        full_prompt = f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        return full_prompt

prompt_builder = PromptBuilder()
