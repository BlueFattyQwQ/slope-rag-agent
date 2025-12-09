import os
import glob
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
from app.ingest.parser import DocumentParser
from app.ingest.chunker import SemanticChunker
from app.pipeline.rag_pipeline import rag_pipeline
from app.core.config import settings
from app.core.logging import logger

app = FastAPI(title="Slope RAG Agent")

class IngestResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    risk_level: str
    rationale: str
    citations: List[dict]
    recommendations: List[str]
    evidence: List[dict]

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(background_tasks: BackgroundTasks):
    """
    扫描 data/sample_docs/，解析→分块→建索引
    """
    files = glob.glob(os.path.join(settings.DATA_DIR, "*.*"))
    if not files:
        return {"message": "No files found", "files_processed": 0, "chunks_created": 0}

    parser = DocumentParser()
    chunker = SemanticChunker(chunk_size=512, chunk_overlap=50)
    
    all_chunks = []
    for file_path in files:
        logger.info(f"Processing {file_path}")
        raw_docs = parser.parse(file_path)
        chunks = chunker.chunk_documents(raw_docs)
        all_chunks.extend(chunks)
    
    # 更新索引
    rag_pipeline.retriever.index_documents(all_chunks)
    
    return {
        "message": "Ingestion complete", 
        "files_processed": len(files), 
        "chunks_created": len(all_chunks)
    }

@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    try:
        result = rag_pipeline.run(request.question)
        return result
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Slope RAG Demo</title>
        <style>
            body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .response { background: #f0f0f0; padding: 15px; margin-top: 20px; border-radius: 5px; }
            .evidence { font-size: 0.9em; color: #555; margin-top: 10px; border-left: 3px solid #ccc; padding-left: 10px; }
        </style>
    </head>
    <body>
        <h1>Slope Stability RAG Agent</h1>
        <textarea id="question" rows="4" style="width: 100%" placeholder="输入问题..."></textarea><br><br>
        <button onclick="ask()">提问</button>
        <div id="result"></div>

        <script>
            async function ask() {
                const q = document.getElementById('question').value;
                const resDiv = document.getElementById('result');
                resDiv.innerHTML = "Thinking...";
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: q})
                    });
                    const data = await response.json();
                    
                    let html = `<div class="response">
                        <h3>Risk Level: ${data.risk_level}</h3>
                        <p><strong>Rationale:</strong> ${data.rationale}</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>${data.recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                        <p><strong>Citations:</strong></p>
                        <ul>${data.citations.map(c => `<li>${c.doc_id} (Page ${c.page})</li>`).join('')}</ul>
                    </div>`;
                    
                    html += `<h3>Evidence Used:</h3>`;
                    data.evidence.forEach(e => {
                        html += `<div class="evidence">
                            <strong>${e.doc_id} (Page ${e.page})</strong><br>
                            ${e.snippet}
                        </div>`;
                    });
                    
                    resDiv.innerHTML = html;
                } catch (e) {
                    resDiv.innerHTML = "Error: " + e;
                }
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
