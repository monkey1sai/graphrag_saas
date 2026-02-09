# Eval JSON v2 — 打分規格（ExpectedV2 / answer_example）

本文件定義 `docs/question.schema.json`（ExpectedV2）在本專案中的「可重現」評估打分方式，配合固定評估 prompt：`docs/prompts/eval_json_v2.md`。

## 1. 評估輸入

- 問題集：`docs/question_v1.json` 或 `docs/question_v2.json`（JSON array，1000 題）。
- 每題包含：
  - `question`（使用者提問）
  - `expected.answer_example`（目標欄位）
  - `expected.source_map`（來源標註；file/page/anchors）

## 2. 評估輸出（模型必須回 JSON）

模型需輸出單一 JSON 物件，欄位固定：

- `target_audience`（string）
- `main_topic`（string）
- `sub_topic`（string）
- `detailed_description`（string[]；最多 8 條）
- `original_evidence`（string）
- `source_map`（array；refs.file 必須來自 RAG context 的 source_path；anchors 必須可在 context 中找到）
- `predicted_questions`（string[]；最多 5 條）

> 若模型輸出不是合法 JSON 或缺欄位，該題 `schema_ok=false` 且 `score=0`。

## 3. 子分數計算（0~1）

### 3.1 三個分類欄位（硬匹配 + 模糊匹配）

- `target_audience` / `main_topic` / `sub_topic`：
  - 若 `expected` 與 `got` 互為包含（substring），或字元 bigram Jaccard ≥ 0.72 → 1，否則 0。

### 3.2 detailed_description（覆蓋率 + 防灌水）

對 `expected.detailed_description` 與 `got.detailed_description` 做「去空白正規化」後的模糊比對：

- `recall = matched_expected / len(expected)`
- `precision = matched_got / len(got)`
- `F1 = 2PR/(P+R)`

防灌水策略：
- 只取 `got.detailed_description[:12]` 參與比對（超過視為無助於分數）。

### 3.3 predicted_questions（同 detailed_description）

同上使用 F1，比對 `expected.predicted_questions` 與 `got.predicted_questions`：

防灌水策略：
- 只取 `got.predicted_questions[:10]` 參與比對。

### 3.4 original_evidence（關鍵詞命中 + 最小長度）

`expected.original_evidence` 抽取關鍵詞（英文 token + 中文 bi-gram），再計算 `got.original_evidence` 的關鍵詞命中：

- 取 `expected` 前 30 個關鍵詞
- 命中數 `hit` 越多越高：`ev_kw_score = min(1, hit/8)`（約 8 個命中視為滿分）
- 長度不足懲罰：`ev_len_pen = 1 if len(got) >= 40 else len(got)/40`
- `original_evidence_score = ev_kw_score * ev_len_pen`

### 3.5 grounding（來源可追溯性）

利用「RAG 檢索到的 context chunks」做 proxy grounding：

對 `got.source_map[*].refs[*]`：
- `file` 必須是本題 context chunks 的 `source_path` 之一（嚴格）
- `anchors` 至少有 1 個字串能在 concatenated context text 找到（substring）

`grounding = ok_refs / total_refs`（只評估前 12 個 source_map entry、每 entry 前 6 個 refs、每 ref 前 6 個 anchors）

> 若 `source_map` 缺失或空，grounding=0。

## 4. 總分（0~100）

加權平均（總和=1）：

- `target_audience`：0.10
- `main_topic`：0.10
- `sub_topic`：0.10
- `detailed_description_f1`：0.30
- `original_evidence`：0.20
- `predicted_questions_f1`：0.10
- `grounding`：0.10

`total = clamp(sum(w_i * s_i), 0..1)`，`score = 100 * total`。

## 5. Gate 規則（dev100）

在 `docs/eval_sets/question_v2_dev100.json` 上：

- `eval_score_avg >= 95`
- `schema_pass_rate >= 0.98`

通過 gate 才進 full（1000 題）。

## 6. 可重現性要求（每次 eval 必須記錄）

每次評估報告必須包含：

- `prompt_sha256`、`prompt_version`
- `index_version`
- `model_id`（base 模型名）
- `adapter_id`（或 registry model_id）
- `eval_set_version`（例如 `question_v2@<sha1>`）

對應工具：
- 評估 prompt：`docs/prompts/eval_json_v2.md`
- Dev subset 產生：`graphrag_saas/backend/scripts/make_eval_subset.py`
- Eval runner：`graphrag_saas/backend/scripts/eval_json_v2.py`

