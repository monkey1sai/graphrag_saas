# Qwen3-4B QLoRA SFT — 最小參數搜尋策略（2~4 組/輪）

目標：以 `docs/prompts/eval_json_v2.md` 的固定 JSON 輸出格式，在 `docs/eval_sets/question_v2_dev100.json` gate set 上達到：

- `eval_score_avg >= 95`
- `schema_pass_rate >= 0.98`

然後才跑 full 1000 題（`docs/question_v2.json`）。

> 前提：使用相同 index（`index_version` 不變）與相同 prompt（`prompt_sha256` 不變），避免把檢索變動誤當模型進步。

## 1) 資料切分建議（避免 leakage）

- Train：`docs/question_v1.json`
- Gate eval：`docs/eval_sets/question_v2_dev100.json`
- Full eval：`docs/question_v2.json`

若 v1/v2 高度重疊，至少要固定 gate set 不變，確保迭代方向一致。

## 2) 每輪只跑 2~4 組（先粗後細）

### Round 0（Smoke，確認 pipeline）

目的：確認能產出 adapter 且 schema_pass_rate 足夠高（模型能穩定輸出 JSON）。

- `max_steps=20`
- `batch_size=1`
- `max_length=1024`
- `lr=2e-4`
- `lora_r=16, lora_alpha=32, lora_dropout=0.05`
- `temperature=0.0`（eval 時固定）

Gate 不要求 95，但要求 `schema_pass_rate >= 0.9`，否則優先修 prompt/輸出約束，而不是調參。

### Round 1（主要提升：學會結構、降低亂答）

建議只測 3 組（同時控制變因）：

1. **Base**：`lr=2e-4, steps=200, max_length=1024`
2. **更保守**：`lr=1e-4, steps=300, max_length=1024`
3. **更強學習**：`lr=3e-4, steps=150, max_length=1024`

選擇標準（以 gate set）：
- 先比 `schema_pass_rate`（必須 ≥0.98）
- 再比 `eval_score_avg`
- 若分數接近（<0.5 差距），優先選 grounding_avg 較高者

### Round 2（細調：coverage vs grounding）

根據 Round 1 的觀察做「小步」調整（最多 2 組）：

- 若 `detailed_description_f1` 偏低 → **加長訓練或稍提高 lr**（步數 +100 或 lr +0.5e-4）
- 若 `grounding_avg` 偏低 → **調整 prompt**（anchor 要求更嚴格）通常比調參有效
- 若 `original_evidence` 偏低 → **提高 max_length 到 1536**（但注意上下文與生成時間）

### Round 3（幾乎沒進步時停止）

停止條件（任一成立即可）：
- 最近兩輪 `eval_score_avg` 提升 < 0.3
- 或 `schema_pass_rate` 已達標但 `eval_score_avg` 在 94.5~95 之間來回震盪

此時應優先做：
- 改 prompt（更明確的 JSON-only、限制條數、強制 anchors 必須出自 context）
- 或改善檢索（top_k / chunk size / 更精準的 source_path）

## 3) 訓練參數的優先序（最小改動）

1. `max_steps`（先調，最直觀）
2. `lr`（小幅調整）
3. `max_length`（影響 evidence/條列完整度，但也影響速度與 OOM 風險）
4. LoRA（`r/alpha/dropout`）：除非確定欠擬合/過擬合，否則不要頻繁動

## 4) 每輪必記錄（寫入 registry / 報告）

每次 train+eval 都要有：
- `model_id`、`adapter_id`（或 registry 的 UUID）
- `prompt_version`、`prompt_sha256`
- `index_version`
- `eval_set_version`
- `eval_score_avg`、`schema_pass_rate`、`grounding_avg`
- `train_params`（lr/steps/max_length/lora_*）

對應實作位置：
- Eval prompt：`docs/prompts/eval_json_v2.md`
- Gate set：`docs/eval_sets/question_v2_dev100.json`
- 打分：`docs/plans/2026-02-04-eval-json-v2-scoring.md`

