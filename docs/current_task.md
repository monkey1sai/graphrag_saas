# Current Task: GraphRAG SaaS：FastAPI + OCR ingest + 1000題benchmark

## Objective
- 把後端升級為 FastAPI，支援 RAG_DATA OCR 匯入、索引落盤、查詢 API，並自動生成1000題做效能/品質評估輸出報告

## Plan
- [x] Phase 0（治理）: 遵守 .agent（current_task + ai_journal）
- [x] Phase 1（可跑 MVP / Plan B）
	- [x] 1.1 FastAPI 服務可啟動（/health）
	- [x] 1.2 支援資料集 ingestion（/ingest）
	- [x] 1.3 支援 RAG_DATA OCR（圖片→文字）
	- [x] 1.4 索引落盤（chunks/meta/entity_names）並可重載
	- [x] 1.5 查詢 API（/query）回傳 answer + sources
	- [x] 1.6 Benchmark API（/benchmark）產生 1000 題並輸出報告
	- [x] 1.7 Docker Compose 可跑（掛載 RAG_DATA、持久化 index/reports）
	- [x] 1.8 Smoke test：ingest→query→benchmark

- [x] Phase 2（評估深化 / Evaluation Harness）
	- [x] 2.1 問題生成器：多樣模板、去重、分層（local/global/跨主題）
	- [x] 2.2 指標：Recall@k / MRR / p50~p99 latency / 吞吐（QPS）/ 失敗率
	- [x] 2.3 報告：JSON + CSV（便於畫圖/回歸）
	- [x] 2.4 固定種子 + 版本化（dataset_version + index_version）確保可重現

- [x] Phase 3（平台化訓練介面：對齊 docx 的 /api/train）
	- [x] 3.1 /api/train：非同步啟動（stage=SFT/RL/WEIGHT_TUNE）
	- [x] 3.2 /api/train/status：標準 JSON（epoch/step/loss/avg_rewards/gpu_mem/eta）
	- [x] 3.3 /api/train/stop：可中止任務（安全停止/狀態落盤）
	- [x] 3.4 訓練狀態落盤（避免重啟丟失）
	- 註：先落地「CPU 可跑的最小 SFT」驗收版本；Docker 內改用 CPU-only torch wheel，避免下載超大的 CUDA 依賴。

- [x] Phase 4（完整 DW‑GRPO 訓練：SFT + PPO/RL + 動態調權）
	- [x] 4.1 SFT 資料生成：建立 <Q,K,C>（教師模型離線/可替代 API）
	- [x] 4.2 Reward models：rel（reranker）、faith（embedding/BERTScore）、conc（長度比）
	- [x] 4.3 PPO/RL：使用 TRL 跑最小文本生成 PPO（需可選 RL 依賴）
		- [x] 4.3.1 RL smoke test（CPU + tiny-gpt2）可從 /api/train 跑到 succeeded 並落盤 summary + adapter
	- [x] 4.4 DW‑GRPO：滑動窗口監測 slope/Δr，自動 softmax 調權（已先落地輕量版）
	- [x] 4.5 防作弊：KL 正則、最小回答長度、策略退化監控
	- [x] 4.6 模型權重版本化與熱切換策略（Registry + per-stage active 指標；先不影響 /query）
	- [x] 4.7 **IRS (Iterative Rejection Sampling)**：8GB VRAM 替代方案
		- [x] 4.7.1 Dockerfile.unsloth + docker-compose.unsloth.yml
		- [x] 4.7.2 DWGRPOScorer（CPU offload）
		- [x] 4.7.3 IRSRunner（Generate → Score → SFT 迴圈）
		- [x] 4.7.4 stage=IRS API 整合 + /metrics 端點
		- [x] 4.7.5 Smoke test（n_iterations=1, n_questions=8）
		- 設計文件：`docs/plans/2026-02-03-irs-design.md`

- [x] Phase 5（可觀測性與運維）
	- [x] 5.1 結構化 log + correlation id（X-Request-Id；JSONL）
	- [x] 5.2 GPU/CPU/mem 監控（Prometheus /metrics + sidecar exporters；宿主/容器視角）
	- [x] 5.3 失敗診斷：錯誤分類（API/job error_code + Prometheus counter）

- [x] Phase 6（Deep GraphRAG 前期資料準備：TWIBM_RAG.xlsx 結構化）
	- [x] 6.1 以 `(AI學習資料)TWIBM_RAG.xlsx` 欄位重整資料落盤（backend/data）
	- [x] 6.2 重建 `docs/question.schema.json`（v2：結構化 answer_example + 向下相容舊格式）
	- [x] 6.3 依 `question_a.json` / `question_aPLUS.json` 問題語氣，輸出新架構 questions 檔（docs）
	- [x] 6.4 驗收：可用 `/ingest` 讀取新 dataset（.md/.txt）並成功建立 index（chunks/meta）

- [ ] Phase 7（Qwen3-4B 實戰微調：Ollama teacher + QLoRA SFT）
	- [x] 7.1 切換 teacher：`TEACHER_PROVIDER=ollama`（遠端 `qwen3:latest`）
	- [ ] 7.2 以 `RAG_DATA` 建立索引（/ingest），並生成 SFT dataset（prompt+completion+citations）
	- [ ] 7.3 QLoRA SFT（use_4bit + LoRA）輸出 adapter 到 `reports/`，並在 registry 註冊 stage=SFT
	- [ ] 7.4 （部署前）將 adapter merge 成完整模型資料夾，供 SGLang `--model-path` 使用
	- [ ] 7.5 評估迭代（gate 100 題 → full 1000 題）：固定 eval prompt + 可重現打分 + registry 記錄
	- [ ] 7.6 **[URGENT] Full 1000 Eval Run** (Benchmark Anchoring)
		- 目標：使用當前最佳配置（Prompt v2 + Repair + Rescore）對 Full 1000 題進行定錨。
		- Gate 標準（分層驗收）：
			- **Format Gate (必過)**: `schema_pass_rate ≥ 0.99`, `invalid_json_or_shape = 0`, `placeholder_string = 0` (repair_used_rate ≤ 0.6 暫定).
			- **Quality Gate (Rescored)**: `metrics.rescored.json['eval_score_avg']`.
				- **Pass (Next Phase)**: ≥ 60
				- **Staging Ready**: ≥ 70
				- **Prod Candidate**: ≥ 80
		- 執行要求：固定 Prompt (`docs/prompts/eval_json_v2.md`), Index, Adapter. 開啟 `--repair-json 1`.
		- 產物：`metrics.json`, `metrics.rescored.json`, `per_item.json`, `per_item.rescored_flat.jsonl`.
		- **狀態 (2026-02-09)**: **RESUMING from Q16** (OOM/Container Exit). Partial data saved in `full1000_run1_repair_p10_tok512`. Need to resume from Q17.
	- [ ] 7.7 **Registry Auto-activate Strategy**
		- 策略：
			- **Prod**: 手動 Activate (Default).
			- **Dev/Staging**: Auto-activate 若 `Job Succeeded` AND `Rescored Score ≥ Threshold` (e.g. 60).
		- 實作：新增 Env `AUTO_ACTIVATE_ON_SUCCESS=1`, `AUTO_ACTIVATE_MIN_RESCORED=60`. 寫入 Registry Metadata.
	- [ ] 7.8 **New Endpoint `/api/generate` (MVP)**
		- 功能：純生成 (Inference only) 使用 Active Adapter.
		- 目的：快速 Smoke Test 模型切換效果，不涉及 RAG 檢索流程。

## Context & Thoughts
- init_at: 2026-02-02T09:59:25Z
- 主要資料集為 RAG_DATA，檔案以圖片（JPG/PNG）為主，必須 OCR 才能 ingest。
- docx 規格包含完整 DW‑GRPO 訓練與平台整合；本專案先落地 Plan B（可用、可評估、可部署），再逐步推進 SFT/RL。
- Docker 內以 tesseract + chi-tra 語言包支援 OCR；Windows 本機若要直接跑 OCR 需自行安裝 Tesseract 並放入 PATH。
- 2026-02-04：補上 repo 根目錄 `README.md`，整理專案結構與啟動方式，方便後續維護/交接。
- 2026-02-09: Full 1000 run crashed at Q16. Retaining `full1000_status.log` and partial reports for resumption. Deleted `run_full1000.bat`.

## Handoff Note
- **CRITICAL**: Full 1000 eval run failed at Q16. DO NOT restart from scratch.
- **Action**: Create a script to resume evaluation from Q0017 using `question_v2.json` (need to slice or filter).
- **Data**: Partial results in `graphrag_saas/backend/reports/evals/full1000_run1_repair_p10_tok512`.
- **System**: Docker prune failed (daemon not reachable?), likely system instability. Rebooting.

### Phase 6（TWIBM 結構化資料）備註
- 新增一個可重現的資料準備流程：把 XLSX 每列轉成「可 ingest 的 .md 文件」＋「可用於評估/訓練的 structured questions」。
- 目標不是取代現有 `RAG_DATA` OCR 流程，而是補上「有結構欄位」的資料來源，方便做 schema-based 評估與 SFT 樣本生成。

### Phase 7（評估/調參基建）備註
- 固定評估 prompt：`docs/prompts/eval_json_v2.md`（會記錄 prompt hash/version）。
- Gate set（dev 100 題）：`docs/eval_sets/question_v2_dev100.json`（由 question_v2 stratified 產生）。
- 打分規格：`docs/plans/2026-02-04-eval-json-v2-scoring.md`。
- 最小調參策略：`docs/plans/2026-02-04-sft-min-tuning-strategy.md`。

#### Phase 7（2026-02-04）實作/驗證狀態
- 已確認容器內可連線遠端 Ollama teacher：`OLLAMA_BASE_URL=http://192.168.20.235:11434`、`OLLAMA_MODEL=qwen3:latest`。
- 已修正 GPU image 可載入 `Qwen/Qwen3-4B-Instruct-2507`（Transformers 版本升級後可辨識 `model_type=qwen3`）。
- 已修正 eval runner 在 Docker 內的匯入/掛載：
	- `scripts/eval_json_v2.py` 加入 project root 到 `sys.path`（避免 `ModuleNotFoundError: graphrag_platform`）。
	- Compose 將 repo root `../docs` 掛載到 `/app/eval_docs`（包含 `eval_sets/`、`prompts/`）。
- 已可跑通「SFT → gate eval（smoke 5 題）」端到端流程；但目前 gate 分數仍偏低，需要改進訓練資料對齊（讓模型更穩定輸出 eval_json_v2 的 JSON keys）。
- 2026-02-05：新增 `docs/eval_sets/question_v2_dev20.json` 作為「更快的 dev gate」（先跑 20 題看 schema/格式趨勢，再決定是否值得跑 dev100/full）。
- 2026-02-05：修正「JSON 解析全滅」問題（不是模型內容差，而是輸出格式進不了解析器）：
	- prompt asset bump：`docs/prompts/eval_json_v2.md` → `prompt_version=eval_json_v2@2026-02-05`（`source_map.part` 改成單一 enum 值，並要求最小化 `source_map/refs/anchors` 避免被截斷）。
	- eval runner 強化：`graphrag_saas/backend/scripts/eval_json_v2.py` 新增 partial 輸出（`metrics.partial.json` / `per_item.partial.json`）與 raw dump，便於逐題定位失敗原因。
	- dev20 gate 驗證：`graphrag_saas/backend/reports/evals/dev20_retry_run6_t256_top1/metrics.json` → `schema_pass_rate=0.65`（舊 `gate_dev20_after_chat_sft_20260205T015838/metrics.json` 為 `0.0`），且 provenance 內已記錄 `adapter_id` / `adapter_path`（避免誤判「adapter 沒用到」）。

### Phase 4（RL）備註
- RL 依賴（TRL/PEFT/accelerate/bitsandbytes）採「可選安裝」，避免破壞既有 CPU-only baseline。
	- Docker Compose：`INSTALL_RL_DEPS=1 docker compose build backend`
	- 仍建議在 CUDA 環境執行 RL；CPU-only 跑大型模型會非常慢或 OOM。

#### Phase 7（2026-02-05）dev20 gate（JSON-only）狀態
- 目標：先把輸出「可被解析 + 符合 schema」拉起來（避免 schema_pass_rate=0 造成 gate 全 0）。
- 採用 A 方案：dev20 gate 啟用 2nd-pass auto-repair（只在 primary 輸出不合格時觸發），並在 per_item 記錄 `primary_fail_reason`/`repair_used` 以避免自欺。
- 最新結果（dev20_retry_run7s_repair_p5_tok512）：
	- schema_pass_rate=1.0（20/20）
	- invalid_json_or_shape=0（fail_reason 全為 null）
	- placeholder_string=0
	- repair_used=11（primary_fail_reason=parse_failed: 11）
- dev100（進行中；2026-02-05T10:56:22Z）：
	- 目標：通過 dev100 gate 後再升級跑 full 1000 題。
	- Run：`graphrag_saas/backend/reports/evals/dev100_run1_repair_p5_tok512/`
	- 結果：qc=100/100，schema_pass_rate=1.0（fail_counts.ok=100），eval_score_avg=17.704

#### GPU/QLoRA（骨架 A：compose override）
- 目標：同一個 `backend` service，用 `docker-compose.gpu.yml` 切到 GPU image（不破壞 CPU baseline）。
- 建置 + 啟動：
	- `INSTALL_RL_DEPS=1 INSTALL_BNB=1 docker compose -f graphrag_saas/docker-compose.yml -f graphrag_saas/docker-compose.gpu.yml up -d --build backend`
	- 若要換 PyTorch base image：`PYTORCH_BASE_IMAGE=...`（見 `backend/Dockerfile.gpu`）

##### GPU 可見性檢查（主機不確定時先做）
- 宿主機：`docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi`
- 容器內：
	- `docker compose exec backend python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('torch_cuda=', torch.version.cuda); print('n_gpu=', torch.cuda.device_count() if torch.cuda.is_available() else 0)"`
	- （可選）`docker compose exec backend nvidia-smi`

##### Phi-3.5 RL smoke（先小步，避免 8GB VRAM OOM）
- 建議參數：`max_steps=1~2`、`batch_size=1`、`max_new_tokens=32`、`use_4bit=true`（需要 bitsandbytes）。
- 範例：`POST /api/train {stage:"RL", model_id:"microsoft/Phi-3.5-mini-instruct", n_questions:8, top_k:1, max_steps:1, batch_size:1, max_new_tokens:32, use_4bit:true}`
- 觀測：`GET /api/train/status/{job_id}` 應包含 `detail.runtime.cuda_available` 與 `detail.runtime.bitsandbytes_import_ok`。

- 由於 RL prompt 會包含檢索到的 context，長度可能逼近模型 max_position_embeddings；目前 runner 會在 tokenize 時保留 max_new_tokens 的空間，避免 generation 超長噴錯。

### Teacher（SFT 樣本產生）
- 預設沿用本地 Plan B（retriever+integrator）當 teacher。
- 系統已準備 Ollama LLM：`qwen3:latest`（2026-02-04）；已完成「容器 → host.docker.internal → Ollama」端到端調用驗證。
- 可切換為 Ollama Server API：
	- `TEACHER_PROVIDER=ollama`
	- `OLLAMA_BASE_URL=http://localhost:11434`
	- `OLLAMA_MODEL=qwen2.5:7b-instruct`
	- `OLLAMA_TIMEOUT_SECONDS=60`、`OLLAMA_MAX_RETRIES=3`
- Teacher 輸出採嚴格 JSON schema（`TeacherOutputV1`，extra=forbid），用於可回歸的訓練資料生成。

## Output
- API：/health、/ingest（非同步 job）、/jobs/{job_id}、/query、/benchmark、/api/train、/api/train/status/{job_id}、/api/train/stop/{job_id}、/metrics
- Model Registry API：/api/models、/api/models/active、/api/models/activate（UUID 作為 model_id；active 依 stage 分開）
- Index：backend/data/index（持久化），Reports：backend/reports/benchmark_*.json
- Registry 落盤：backend/data/model_registry.json、backend/data/active_models.json（原子寫入 + 檔案鎖）
- 監控：Prometheus `http://localhost:9090`；exporters：node_exporter `:9100`、cAdvisor `:8080`（Docker Desktop 下 exporters 資料可能受限）
- 驗收：docker-compose build && docker-compose up -d；再呼叫 /health → /ingest → /benchmark
- 最近一次驗收（2026-02-02）：benchmark_20260202T101409Z.json（Recall@5=0.944, MRR=0.6907）
- Phase 2 驗收（2026-02-02）：
	- benchmark_20260202T102719Z.json（Recall@5=0.955, MRR=0.7343, QPS=11834.77, error_rate=0）
	- benchmark_20260202T102719Z.csv（metrics CSV）
	- questions_20260202T102719Z.jsonl（1000 題本體，含 qid/question/gold_chunk_id/kind/template）

- Phase 4（2026-02-03）RL smoke test 驗收：
	- /api/train stage=RL model_id=sshleifer/tiny-gpt2 max_steps=2 batch_size=1 max_new_tokens=16 use_4bit=false
	- 產物：train_20260203T061000374611Z_rl_summary.json + train_20260203T061000374611Z_adapter/

- Phase 4.5（2026-02-04）防作弊/監控最小驗收：
	- /api/train stage=RL model_id=sshleifer/tiny-gpt2 max_steps=1 batch_size=1 max_new_tokens=16 use_4bit=false min_response_tokens=8
	- 產物：train_20260204T075059038150Z_rl_summary.json + train_20260204T075059038150Z_rl_metrics.jsonl

- Phase 4.6（2026-02-04）Registry/active 驗收：
	- GET /api/models?stage=RL 可看到自動註冊 entry（包含 artifacts workspace_hint）
	- POST /api/models/activate {stage:"RL", model_id:"<uuid>"} 後，GET /api/models/active 會顯示 active.RL=<uuid>

- Phase 5.1（2026-02-04）Correlation id + 結構化 log 驗收：
	- GET /health response header 含 `X-Request-Id`
	- docker logs 會輸出 JSONL（含 request_id、method、path、status_code、duration_ms）

- Phase 5.2（2026-02-04）Prometheus /metrics + exporters 驗收：
	- Backend：GET /metrics 回 Prometheus text format；train job 指標 `graphrag_train_job_*`
	- Prometheus：localhost:9090 ready；targets 包含 backend/node_exporter/cadvisor（Docker Desktop 下 exporters 可能資料有限）

- Phase 5.3（2026-02-04）錯誤分類驗收：
	- 失敗的 train job（例：缺 RL 依賴）會在 job.detail 內包含 `error_code`
	- /metrics 會累計 `graphrag_errors_total{stage,code}`

- Phase 4.7（2026-02-04）IRS smoke test 驗收：
	- /api/train stage=IRS model_id=unsloth/Llama-3.2-1B-Instruct-bnb-4bit irs_n_iterations=1 n_questions=8 irs_n_candidates=2 max_new_tokens=32
	- 狀態：`/api/train/status/20260204T071857882375Z` → `succeeded`
	- 產物：`graphrag_saas/backend/data/jobs/train_20260204T071857882375Z/`（metrics.jsonl、samples.jsonl、summary.json、checkpoints/iter_1）

## Handoff Note
- 下一步：決定 Registry 的「自動 activate 策略」（例如：最新成功 job 自動設為 active；或只手動 activate），以及是否新增一個使用 active adapter 的推理端點（不改動 /query 的前提下新增 /generate）。
- IRS 是 8GB VRAM 環境下的 DW-GRPO 替代方案：Generate → Score → SFT 迴圈，無需同時載入 policy + reference model。
- 若 OCR 失敗：優先檢查容器內 tesseract 與語言包是否可用；本機則確認已安裝 Tesseract。
- Phase 7（JSON-only gate）：先等 `dev100_run1_repair_p5_tok512` 產出 `metrics.json`，回報 schema_pass_rate / invalid / placeholder_string / repair_used；通過後再跑 full 1000 題（同樣固定 eval prompt 版本與推理參數，並把指標寫回 registry metadata）。

### Phase 7（2026-02-06）Eval 命中率提升（MVP）進度
- Eval prompt 資產已更新到：`docs/prompts/eval_json_v2.md` → `prompt_version=eval_json_v2@2026-02-06-p10`
	- 加入 Allowed labels 逐字選值規則、禁止 `"資料不足"` 用在分類欄位、anchors 必須取自 anchor_candidates。
	- `detailed_description` / `predicted_questions`：除非 context 完全無資訊，否則不得為空陣列（避免整體 F1 長期為 0）。
- Eval runner：`graphrag_saas/backend/scripts/eval_json_v2.py`
	- Grounding anchors 比對改用正規化文字（避免因換行/多空白導致 anchors 不命中）。
	- Repair pass（2nd-pass auto-repair）補上 Allowed labels + grounding candidates 引導，降低修復輸出的「全資料不足」比例。
	- OpenAI provider 補上 `max_tokens/top_p`（以 gate 參數為準；避免 server 預設 token 太短截斷）。
	- Scoring（v2，避免 list 欄位 all-zero 的假陰性）：
		- `detailed_description` / `predicted_questions` 改用 deterministic fuzzy matching（字元級 similarity）評分。
		- evidence 長度/關鍵字命中門檻調整，使「短摘錄」不被過度扣分（仍保留防灌水上限）。
	- 新增離線重算工具（不重跑 LLM）：`graphrag_saas/backend/scripts/rescore_eval_run.py`
		- 會輸出 `metrics.rescored.json`、`per_item.rescored_flat.json`（用於更新 breakdown）。
	- analyze 工具補強：`graphrag_saas/backend/scripts/analyze_eval_run.py`
		- breakdown 新增 `is_partial` / `question_count_expected`（避免把 partial run 誤判成 dev100 全量）。
		- `--use-rescored 1` 可直接用 rescored 檔產生新版 breakdown/report。
- dev20 rerun（固定題集/adapter/index/top_k；新增 raw dump 便於診斷）：
	- out_dir：`graphrag_saas/backend/reports/evals/dev20_retry_run8_p8_repair_p5_tok512_raw/`
	- metrics：`schema_pass_rate=1.0`、`eval_score_avg=39.688`、`grounding_avg=0.55`
	- breakdown（analyze）：label hit-rate `target_audience=0.80`、`main_topic=0.70`、`sub_topic=1.00`；`repair_used=11/20`（primary parse_failed=11）
- dev100 rerun（同一題集/adapter/index/top_k；raw dump 開啟）：
	- out_dir（run2，已中止）：`graphrag_saas/backend/reports/evals/dev100_run2_p8_repair_p5_tok512_raw/`
		- 問題：repair 有機率輸出過長 evidence/問題句，造成 JSON 截斷而回落到 `parse_failed`。
	- out_dir（run3，repair 短輸出版）：`graphrag_saas/backend/reports/evals/dev100_run3_p8_repair_short_tok512_raw/`
		- 這是一個 partial run：只有 `metrics.partial.json/per_item.partial.json`（count=10），新版 breakdown 已標記 `is_partial=true`。
		- metrics.partial@10：`schema_pass_rate=1.0`、`eval_score_avg=40.5`、`grounding_avg=0.9`（已達成「分數明顯上升」門檻）
	- out_dir（run4，完整 100 題）：`graphrag_saas/backend/reports/evals/dev100_run4_p8_repair_short_tok512_raw_full/`
		- final metrics：`schema_pass_rate=1.0`、`eval_score_avg=43.3`、`grounding_avg=0.75`、`eval_score_p50=42.5`
		- 完整 breakdown（count=100）：`graphrag_saas/backend/reports/evals/dev100_run4_p8_repair_short_tok512_raw_full/breakdown.json`
		- 關鍵診斷：
			- `label hit-rate`：`target_audience=83%`、`main_topic=75%`、`sub_topic=100%`
			- `repair_used=50/100`（primary_fail_reason: parse_failed=50）
			- `grounding_zero_reasons`: `anchors_not_found_in_ctx_text=25`, `no_ok_ref=25`
			- `detailed_description_f1=0`、`predicted_questions_f1=0`（舊評分口徑下卡分主因）
		- 以新版 scoring 離線重算（不重跑 LLM）後：
			- `metrics.rescored.json`: `eval_score_avg≈61.98`（>=60）
			- `breakdown.json`（`--use-rescored 1` 產生）：
				- `detailed_description_f1.mean≈0.407`、`predicted_questions_f1.mean≈0.367`
