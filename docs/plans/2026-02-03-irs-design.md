# IRS (Iterative Rejection Sampling) 設計文件

> **Created**: 2026-02-03  
> **Status**: Approved  
> **Hardware Target**: RTX 4060 8GB VRAM + 64GB RAM

---

## 1. 背景與動機

### 1.1 硬體限制

- **GPU**: RTX 4060 8GB VRAM（無法同時載入 policy + reference model 跑 PPO）
- **RAM**: 64GB（巨大優勢，可用於 CPU offload）

### 1.2 核心問題

傳統 DW-GRPO 使用 PPO 訓練，需要同時載入：
- Policy Model（~3-4GB for 1.5B 4-bit）
- Reference Model（~3-4GB）
- Reward Model（可選）

8GB VRAM 無法容納，導致 OOM。

### 1.3 解決方案：Iterative Rejection Sampling (IRS)

將 RL 拆解為「生成 → 評分 → 微調」三個獨立階段，每階段只載入一個模型：

```
┌─────────────────────────────────────────────────────────────┐
│                        IRS Loop                             │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                   │
│  │Generate │ → │ Score   │ → │  SFT    │  ←─ Loop N 次     │
│  │(Unsloth)│   │(CPU/DW) │   │(Unsloth)│                   │
│  │ GPU     │   │ CPU     │   │ GPU     │                   │
│  └─────────┘   └─────────┘   └─────────┘                   │
│       │              │             │                        │
│       ▼              ▼             ▼                        │
│  N candidates   best_sample   adapter.safetensors          │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 系統架構

### 2.1 組件分工

| 組件 | 運行位置 | 配置 |
|------|----------|------|
| LLM 推理 (Generate) | GPU | Unsloth 4-bit 量化 |
| LLM 微調 (SFT) | GPU | Unsloth + LoRA + paged_adamw_8bit |
| Reward Scoring | CPU | sentence-transformers + BERTScore |
| DW-GRPO 權重調整 | CPU | 滑動窗口斜率 + softmax |
| 向量/圖譜資料 | RAM | 64GB in-memory |

### 2.2 核心流程

1. **Generate**：用 Unsloth 4-bit 推理，每題產 N 個候選答案（GPU ~2GB）
2. **Score**：在 CPU 上跑 DW-GRPO 動態加權評分，選出「金牌答案」
3. **SFT**：用金牌答案微調 LoRA adapter（GPU ~6GB）
4. **Loop**：重複 1-3 共 `n_iterations` 輪

---

## 3. DW-GRPO Scorer 設計

### 3.1 獎勵指標

| 指標 | 來源 | 運行位置 | 說明 |
|------|------|----------|------|
| `r_rel` (相關性) | Cross-Encoder | CPU | `bge-reranker-v2-m3` 判斷 Q-A 關聯 |
| `r_faith` (忠實度) | BERTScore/Embedding | CPU | `bge-m3` 計算 A-Context 相似度 |
| `r_conc` (簡潔性) | Heuristic | CPU | `1 - len(A)/len(Context)` |

### 3.2 動態權重更新

```python
# 計算最近 N 輪的斜率
slopes = {k: np.polyfit(x, history[k][-window:], 1)[0] for k in weights}

# 斜率越小 (stagnant)，權重應越大
exps = {k: np.exp(-v / temperature) for k, v in slopes.items()}
total = sum(exps.values())
new_weights = {k: 3.0 * (v / total) for k, v in exps.items()}

# Momentum 平滑更新
weights = {k: 0.8 * weights[k] + 0.2 * new_weights[k] for k in weights}
```

---

## 4. API 設計

### 4.1 Request: `POST /api/train`

```json
{
  "stage": "IRS",
  "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
  "irs_config": {
    "n_iterations": 3,
    "n_candidates": 4,
    "batch_size": 2,
    "learning_rate": 2e-5,
    "max_new_tokens": 256,
    "load_in_4bit": true,
    "dw_config": {
      "history_window": 5,
      "temperature": 2.0,
      "initial_weights": {"rel": 1.0, "faith": 1.0, "conc": 0.5}
    }
  },
  "n_questions": 100,
  "top_k": 3
}
```

### 4.2 Response: `GET /api/train/status/{job_id}`

```json
{
  "job_id": "train_20260203T120000Z",
  "stage": "IRS",
  "state": "running",
  "detail": {
    "current_iteration": 2,
    "total_iterations": 3,
    "current_phase": "SCORING",
    "phase_progress": 45.0,
    "metrics": {
      "avg_reward": 2.45,
      "reward_breakdown": {"rel": 0.90, "faith": 0.85, "conc": 0.70},
      "current_weights": {"rel": 1.05, "faith": 1.20, "conc": 0.42}
    },
    "runtime": {
      "device": "cuda",
      "unsloth_available": true,
      "gpu_mem_used": 6800000000
    }
  }
}
```

### 4.3 新增: `GET /api/train/{job_id}/metrics`

回傳 `metrics.jsonl` 內容（JSON array），供前端繪製 DW-GRPO 趨勢圖。

---

## 5. 檔案與目錄結構

### 5.1 Job Artifacts 目錄

```
/app/data/jobs/
└── train_{job_id}/
    ├── config.json          # 訓練參數
    ├── metrics.jsonl        # 訓練過程指標 (Loss, Rewards, Weights)
    ├── samples.jsonl        # 每輪選出的金牌答案
    ├── checkpoints/
    │   ├── iter_1/          # 第 1 輪 LoRA Adapter
    │   ├── iter_2/          # 第 2 輪 LoRA Adapter
    │   └── final/           # 最終模型
    └── train.log            # 詳細日誌
```

### 5.2 metrics.jsonl 格式

```jsonl
{"ts": "2026-02-03T12:00:01Z", "iter": 1, "phase": "GENERATING", "progress": 10}
{"ts": "2026-02-03T12:05:00Z", "iter": 1, "phase": "SCORING", "progress": 50}
{"ts": "2026-02-03T12:10:00Z", "iter": 1, "phase": "METRICS", "avg_reward": 2.1, "weights": {"rel": 1.0, "faith": 1.0, "conc": 0.5}}
{"ts": "2026-02-03T12:15:00Z", "iter": 1, "phase": "TRAINING", "epoch": 1, "loss": 1.25}
```

---

## 6. Docker 整合

### 6.1 新增 `Dockerfile.unsloth`

```dockerfile
FROM unslothai/unsloth:cu121-torch240-step15

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    sentence-transformers>=2.7 \
    bert-score>=0.3.13

COPY . .

ENV BACKEND_MODE=IRS_UNSLOTH
ENV REWARD_DEVICE=cpu
ENV HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 新增 `docker-compose.unsloth.yml`

```yaml
services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.unsloth
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - BACKEND_MODE=IRS_UNSLOTH
      - REWARD_DEVICE=cpu
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./backend/data:/app/data
      - ./backend/reports:/app/reports
      - hf_hub_cache:/root/.cache/huggingface
```

### 6.3 啟動指令

```powershell
# IRS/Unsloth 模式（RTX 4060 環境）
docker compose -f docker-compose.yml -f docker-compose.unsloth.yml up -d --build backend

# 標準 RL/PPO 模式（未來 A100 環境）
$env:INSTALL_RL_DEPS=1; docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build backend
```

---

## 7. 程式碼模組結構

### 7.1 新增檔案

| 檔案路徑 | 說明 |
|----------|------|
| `backend/Dockerfile.unsloth` | Unsloth 官方映像為基底 |
| `docker-compose.unsloth.yml` | IRS 模式 compose override |
| `backend/graphrag_platform/irs/__init__.py` | 模組初始化 |
| `backend/graphrag_platform/irs/irs_runner.py` | IRS 迴圈主邏輯 |
| `backend/graphrag_platform/irs/dw_grpo_scorer.py` | DW-GRPO 動態權重 scorer |

### 7.2 修改檔案

| 檔案路徑 | 說明 |
|----------|------|
| `backend/app.py` | 新增 `stage=IRS` 分派 + `/metrics` API |
| `docs/current_task.md` | 更新 Phase 4 進度 |

---

## 8. 實作優先順序

1. **Phase 1**：建立 `Dockerfile.unsloth` + `docker-compose.unsloth.yml`，驗證 Unsloth 環境可用
2. **Phase 2**：實作 `dw_grpo_scorer.py`，單元測試 scoring 邏輯
3. **Phase 3**：實作 `irs_runner.py`，整合 Generate → Score → SFT 迴圈
4. **Phase 4**：修改 `app.py` 新增 `stage=IRS` 分派
5. **Phase 5**：端到端 smoke test（n_iterations=1, n_questions=8）

---

## 9. 風險與緩解

| 風險 | 緩解措施 |
|------|----------|
| Unsloth 官方映像版本更新導致不相容 | Pin 到特定 tag `cu121-torch240-step15` |
| CPU Scoring 太慢 | 支援 fallback 到 `HeuristicScorer` |
| 模型下載卡住 | 複用現有 `hf_hub_cache` volume |
| VRAM OOM（邊緣情況） | 支援 `max_new_tokens` 限制 + `torch.cuda.empty_cache()` |

---

## 10. 驗收標準

- [ ] `docker compose -f docker-compose.yml -f docker-compose.unsloth.yml up -d --build backend` 成功啟動
- [ ] `POST /api/train {stage: "IRS", n_iterations: 1, n_questions: 8}` 回傳 job_id
- [ ] `GET /api/train/status/{job_id}` 顯示 phase 進度與 DW-GRPO 權重
- [ ] `/app/data/jobs/train_{job_id}/metrics.jsonl` 有結構化日誌
- [ ] `/app/data/jobs/train_{job_id}/checkpoints/iter_1/` 有 LoRA adapter
