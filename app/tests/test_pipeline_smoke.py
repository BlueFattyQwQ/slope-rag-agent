import pytest
from app.pipeline.rag_pipeline import rag_pipeline
from app.ingest.parser import DocumentChunk

def test_pipeline_smoke():
    # Mock 检索结果
    mock_docs = [
        DocumentChunk(doc_id="test.pdf", page=1, section_path="s1", text="边坡稳定性受降雨影响显著，安全系数会降低。", is_table=False)
    ]
    
    # 注入 Mock 数据 (实际应使用 mock 库 patch retriever)
    # 这里简单测试 run 方法是否报错
    try:
        # 假设没有数据也能跑通，返回 unknown
        result = rag_pipeline.run("测试问题")
        assert "risk_level" in result
        assert "citations" in result
    except Exception as e:
        pytest.fail(f"Pipeline run failed: {e}")
