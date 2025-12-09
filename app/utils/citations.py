from typing import List, Dict, Any
from app.ingest.parser import DocumentChunk

def validate_citations(response_json: Dict[str, Any], context_docs: List[DocumentChunk]) -> Dict[str, Any]:
    """
    校验引用一致性：citations 中的 doc_id/page 必须来自本次检索结果
    """
    valid_citations = []
    
    # 构建有效引用集合
    valid_sources = set()
    for doc in context_docs:
        valid_sources.add((doc.doc_id, doc.page))
        
    raw_citations = response_json.get("citations", [])
    
    for cit in raw_citations:
        doc_id = cit.get("doc_id")
        page = cit.get("page")
        
        # 宽松匹配：如果 doc_id 包含在 context 中，或者 context 包含 doc_id
        is_valid = False
        for v_doc_id, v_page in valid_sources:
            if (doc_id == v_doc_id) and (int(page) == int(v_page)):
                is_valid = True
                break
        
        if is_valid:
            valid_citations.append(cit)
            
    response_json["citations"] = valid_citations
    return response_json
