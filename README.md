# GraphRAG SaaS（FastAPI + OCR Ingest + Benchmark + Train）

[![CI](https://github.com/monkey1sai/graphrag_saas/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/monkey1sai/graphrag_saas/actions/workflows/ci.yml)

本專案提供一個可部署的 GraphRAG 後端（FastAPI），支援：
- 從資料夾匯入（DOCX/PDF/XLSX/圖片 OCR）→ chunking → 索引落盤
- 查詢 API：以 `HierarchicalRetriever`（TF‑IDF + entity 層級過濾）檢索，交由 `WeightedIntegrator` 組合回答
- Benchmark：自動生成題目並輸出評估報告（JSON/CSV）
- Train：提供最小可跑的 SFT/RL（PPO/DW‑GRPO wiring）與 IRS（Iterative Rejection Sampling，Unsloth 模式）

---

## 專案結構

- `graphrag_saas/`：Docker Compose 與後端專案
  - `graphrag_saas/backend/app.py`：FastAPI 入口（/health、/ingest、/query、/benchmark、/api/train…）
  - `graphrag_saas/backend/graphrag_platform/`：索引、檢索、整合、benchmark、訓練/IRS 等核心模組
  - `graphrag_saas/docker-compose.yml`：CPU baseline（含 OCR tesseract）
  - `graphrag_saas/docker-compose.gpu.yml`：GPU override（可選）
  - `graphrag_saas/docker-compose.unsloth.yml`：Unsloth/IRS 模式（8GB VRAM 取向）
- `RAG_DATA/`：主要資料集（多為圖片，需要 OCR）
- `docs/`：任務追蹤與設計文件（例如 `docs/plans/2026-02-03-irs-design.md`）

### 題庫 / 評測資產

為了方便重現 benchmark / supervised SFT，本 repo 會直接包含小型題庫與 schema（體積約數 MB 等級）：
- `docs/question_v1.json`、`docs/question_v2.json`
- `docs/eval_sets/question_v2_dev20.json`、`docs/eval_sets/question_v2_dev100.json`
- `docs/question.schema.json`

---

## 快速啟動（Docker，建議）

### 先決條件

- 已安裝 Docker Desktop（含 `docker compose`）
- 已安裝 Git LFS（用來下載/推送權重檔）

### Clone 後先拉 LFS 權重

本 repo 內的 `*.safetensors` 權重使用 Git LFS 管理；clone 完請執行：

```powershell
git lfs install
git lfs pull
```

在 repo 根目錄執行：

```powershell
docker compose -f .\graphrag_saas\docker-compose.yml up -d --build
```

預設會：
- 將 `RAG_DATA/` 以唯讀掛載到容器 `/data/RAG_DATA`
- 將索引與報告落在 `graphrag_saas/backend/data/`、`graphrag_saas/backend/reports/`（可持久化）

注意：`RAG_DATA/` 不會跟著 repo 一起發佈（避免把大型原始資料推上 GitHub）。
- 你需要自行在 repo 根目錄準備 `RAG_DATA/`（或改用環境變數覆寫 `DATASET_PATH`）
- 若沒有資料集，`/ingest` 會找不到輸入來源

### 常用檢查

- Swagger UI：`http://localhost:8000/docs`
- 健康檢查：`GET http://localhost:8000/health`

---

## API（常用）

服務預設埠：`http://localhost:8000`

- `GET /health`：健康檢查
- `POST /ingest`：匯入資料（非同步 job）
- `GET /jobs/{job_id}`：查看任務狀態
- `POST /query`：查詢（回傳 `answer` + `sources`）
- `POST /benchmark`：跑 benchmark（輸出到 reports）
- `POST /api/train`：訓練入口（stage=SFT/RL/IRS）
- `GET /api/train/status/{job_id}`：訓練狀態
- `POST /api/train/stop/{job_id}`：請求停止
- `GET /api/train/{job_id}/metrics`、`GET /api/train/{job_id}/samples`：IRS 產物查詢

---

## Docker 參數覆寫（建議用 .env）

`graphrag_saas/docker-compose.yml` 預設 teacher provider 指向內網 Ollama；一般需要依你的環境調整。

在 repo 根目錄新增 `.env`（不建議提交），例如：

```env
TEACHER_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:latest

# 若要用標註題庫做 supervised SFT，可指定：
# SFT_SUPERVISED_QUESTIONS_PATH=/app/eval_docs/question_v2.json
```

然後照原本方式啟動 compose 即會自動讀取 `.env`。

## 本機啟動（非 Docker）

注意：
- OCR 需要本機可執行的 Tesseract（`tesseract.exe` 在 PATH）。
- 這個專案偏向用 Docker 跑（依賴較多）；若要本機跑，請自行建立 venv 並安裝依賴。

```powershell
cd .\graphrag_saas\backend

python -m venv ..\..\.venv
..\..\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

..\..\.venv\Scripts\python.exe -m uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## GPU / IRS（可選）

- GPU 模式（override）：
```powershell
docker compose -f .\graphrag_saas\docker-compose.yml -f .\graphrag_saas\docker-compose.gpu.yml up -d --build
```

- Unsloth / IRS 模式（見設計文件 `docs/plans/2026-02-03-irs-design.md`）：
```powershell
docker compose -f .\graphrag_saas\docker-compose.yml -f .\graphrag_saas\docker-compose.unsloth.yml up -d --build
```

