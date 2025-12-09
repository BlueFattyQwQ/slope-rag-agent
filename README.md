# Slope RAG Agent

基于垂直领域专家模型（SFT）构建的边坡工程 RAG 问答系统。支持混合检索、重排序、证据裁剪与结构化输出。

## 功能特性

*   **混合检索**: BM25 (Elasticsearch/Local) + Vector (FAISS/BGE)
*   **重排序**: Cross-Encoder (BGE-Reranker)
*   **结构化输出**: JSON 格式，包含风险等级、理由、引用与建议
*   **引用校验**: 自动校验生成的引用是否来自检索到的上下文
*   **工具调用**: 集成天气查询与工程简算器
*   **评估体系**: Recall@k, MRR, nDCG 及三元评估脚本

## 目录结构

```
slope-rag-agent/
├── app/
│   ├── api/            # FastAPI 服务
│   ├── core/           # 配置与日志
│   ├── ingest/         # 文档解析与分块
│   ├── index/          # 向量与倒排索引
│   ├── search/         # 检索与重排
│   ├── llm/            # 模型推理封装
│   ├── pipeline/       # RAG 主流程
│   └── tools/          # 外部工具
├── data/               # 数据与索引存储
├── docker/             # Docker 配置
└── eval/               # 评测脚本
```

## 快速开始

### 1. 环境准备

确保已安装 Python 3.10+ 和 Poetry。

```bash
# 安装依赖
poetry install
```

### 2. 配置

复制 `.env.sample` 为 `.env` 并按需修改。

```bash
cp .env.sample .env
```

*   **模型路径**: 默认使用 HuggingFace ID 自动下载，如需离线使用请修改 `SFT_MODEL_ID` 等路径。
*   **Elasticsearch**: 默认尝试连接本地 ES，失败则自动回退到内存版 Rank-BM25。

### 3. 运行服务

```bash
# 启动 API 服务
poetry run python app/api/server.py
```

访问 `http://localhost:8000` 查看 Demo 界面。
访问 `http://localhost:8000/docs` 查看 API 文档。

### 4. 数据导入与提问

1.  将 PDF/Markdown 文档放入 `data/sample_docs/`。
2.  调用 Ingest API 构建索引：
    ```bash
    curl -X POST http://localhost:8000/ingest
    ```
3.  提问：
    ```bash
    curl -X POST http://localhost:8000/ask \
      -H "Content-Type: application/json" \
      -d '{"question": "近期强降雨条件下，A区边坡的稳定性风险？"}'
    ```

### Docker 运行

```bash
docker-compose -f docker-compose.yml up --build
```

## 评估

运行离线评估脚本：

```bash
poetry run python app/eval/eval_runner.py
```

结果将输出到 `outputs/eval_results.json`。

## 常见问题

*   **ES 连接失败**: 系统会自动降级使用 `rank-bm25`，仅在内存中构建索引，重启服务后需重新 Ingest。
*   **CUDA OOM**: 请在 `.env` 中调小 `MAX_INPUT_TOKENS` 或启用 4-bit 量化（默认已启用）。
*   **模型下载慢**: 请设置 `HF_ENDPOINT=https://hf-mirror.com` 环境变量。
