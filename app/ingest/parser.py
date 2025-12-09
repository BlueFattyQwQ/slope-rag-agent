import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import pdfplumber
from app.core.logging import logger

@dataclass
class DocumentChunk:
    doc_id: str
    page: int
    section_path: str
    text: str
    is_table: bool = False
    table_path: Optional[str] = None
    url: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self):
        return asdict(self)

class DocumentParser:
    def __init__(self):
        pass

    def parse(self, file_path: str) -> List[DocumentChunk]:
        """
        解析文件，支持 PDF。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.md', '.txt']:
            return self._parse_text(file_path)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []

    def _parse_pdf(self, file_path: str) -> List[DocumentChunk]:
        chunks = []
        doc_id = os.path.basename(file_path)
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_num = i + 1
                    
                    # 提取表格
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        # 将表格转换为 Markdown 格式文本
                        table_text = self._table_to_markdown(table)
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            page=page_num,
                            section_path=f"Page {page_num} Table {table_idx+1}",
                            text=table_text,
                            is_table=True,
                            table_path=f"table_{page_num}_{table_idx}"
                        ))

                    # 提取正文文本 (简单过滤掉表格区域可能比较复杂，这里简化处理，直接提取全文)
                    # 实际生产中应剔除表格区域
                    text = page.extract_text()
                    if text:
                        # 简单的按段落分割，后续由 Chunker 进一步处理
                        # 这里先作为一个大块返回，或者按换行符粗分
                        chunks.append(DocumentChunk(
                            doc_id=doc_id,
                            page=page_num,
                            section_path=f"Page {page_num} Content",
                            text=text,
                            is_table=False
                        ))
        except Exception as e:
            logger.error(f"Error parsing PDF {file_path}: {e}")
            
        return chunks

    def _parse_text(self, file_path: str) -> List[DocumentChunk]:
        doc_id = os.path.basename(file_path)
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    page=1,
                    section_path="Full Content",
                    text=text,
                    is_table=False
                ))
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
        return chunks

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        if not table:
            return ""
        # 简单处理 None
        table = [['' if cell is None else str(cell).replace('\n', ' ') for cell in row] for row in table]
        
        markdown = "|" + "|".join(table[0]) + "|\n"
        markdown += "|" + "|".join(["---"] * len(table[0])) + "|\n"
        for row in table[1:]:
            markdown += "|" + "|".join(row) + "|\n"
        return markdown
