# Eval Breakdown Report

- run_dir: `D:\智能客服助手\graphrag_saas\graphrag_saas\backend\reports\evals\dev100_run4_p8_repair_short_tok512_raw_full`
- metrics: `D:\智能客服助手\graphrag_saas\graphrag_saas\backend\reports\evals\dev100_run4_p8_repair_short_tok512_raw_full\metrics.rescored.json`
- per_item: `D:\智能客服助手\graphrag_saas\graphrag_saas\backend\reports\evals\dev100_run4_p8_repair_short_tok512_raw_full\per_item.rescored_flat.json`

## Overall score
- mean=64.476 p50=65.833 p90=68.333 p95=81.143 min=48.667 max=81.143
- repair_used: 50/100 (50.0%)
- primary_fail_reason: `{"parse_failed": 50}`

## Subs stats (mean / p50 / p95)
- detailed_description_f1: mean=0.4069 p50=0.5000 p95=0.7500
- detailed_description_precision: mean=0.3453 p50=0.3333 p95=0.7500
- detailed_description_recall: mean=0.5625 p50=0.7500 p95=1.0000
- grounding: mean=1.0000 p50=1.0000 p95=1.0000
- grounding_ok_refs: mean=1.0800 p50=1.0000 p95=2.0000
- grounding_total_refs: mean=1.0800 p50=1.0000 p95=2.0000
- main_topic: mean=0.7500 p50=1.0000 p95=1.0000
- original_evidence: mean=0.6400 p50=0.6667 p95=1.0000
- predicted_questions_f1: mean=0.3670 p50=0.5000 p95=1.0000
- predicted_questions_precision: mean=0.3617 p50=0.5000 p95=1.0000
- predicted_questions_recall: mean=0.3750 p50=0.5000 p95=1.0000
- sub_topic: mean=1.0000 p50=1.0000 p95=1.0000
- target_audience: mean=0.8300 p50=1.0000 p95=1.0000

## Label hit-rate
- target_audience: 83/100 (83.0%)
- main_topic: 75/100 (75.0%)
- sub_topic: 100/100 (100.0%)

## Confusion Top-15: target_audience (pred -> expected)
- 9× `業主/管理顧問` -> `人力資源部/管委會`
- 8× `業主/管理顧問` -> `餐飲服務部`

## Confusion Top-15: main_topic (pred -> expected)
- 17× `服務方案特色` -> `人事管理`
- 8× `環境維護標準` -> `人事管理`

## Confusion Top-15: sub_topic (pred -> expected)

## Grounding=0 reasons (diagnosis)
- (no grounding=0 items or insufficient data)

