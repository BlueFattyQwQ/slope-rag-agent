from typing import List
from app.ingest.parser import DocumentChunk
from app.core.config import settings
import re

class SemanticChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, docs: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        对文档片段进行进一步的语义分块。
        """
        chunked_docs = []
        
        for doc in docs:
            if doc.is_table:
                # 表格通常保持完整，或者按行切分，这里假设表格已经适合作为单独块
                # 如果表格太大，可以考虑截断，但保持结构完整较难
                chunked_docs.append(doc)
                continue
            
            text = doc.text
            # 简单的按字符数/Token数切分
            # 实际生产中应使用 Tokenizer，这里用字符数近似 (中文约 1 char = 0.6-1 token)
            # 假设 1 char ~= 1 token for simplicity in this demo
            
            start = 0
            text_len = len(text)
            
            while start < text_len:
                end = min(start + self.chunk_size, text_len)
                
                # 尝试在句号、换行符处截断，避免切断句子
                if end < text_len:
                    # 向后查找最近的分隔符
                    lookback = text[max(0, end - 50):end]
                    split_chars = ['。', '！', '？', '\n', '.', '!', '?']
                    last_split = -1
                    for char in split_chars:
                        pos = lookback.rfind(char)
                        if pos != -1:
                            last_split = max(last_split, pos)
                    
                    if last_split != -1:
                        end = max(0, end - 50) + last_split + 1
                
                chunk_text = text[start:end].strip()
                if chunk_text:
                    new_doc = DocumentChunk(
                        doc_id=doc.doc_id,
                        page=doc.page,
                        section_path=doc.section_path,
                        text=chunk_text,
                        is_table=False,
                        metadata={"original_start": start, "original_end": end}
                    )
                    chunked_docs.append(new_doc)
                
                start += (self.chunk_size - self.chunk_overlap)
                
        return chunked_docs
