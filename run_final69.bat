@echo off
echo ================================================
echo  Full 1000 Eval - Final 65 questions (Q0936-Q1000)
echo  With 3-minute progress monitoring
echo ================================================
echo.

set OUT_DIR=full1000_run1_repair_p10_tok512_final69
set QUESTIONS=/app/eval_docs/eval_sets/question_v2_remaining_Q932_1000.json
set PROMPT=/app/eval_docs/prompts/eval_json_v2.md
set MODEL=Qwen/Qwen3-4B-Instruct-2507
set ADAPTER=/app/reports/train_20260205T015838743819Z_sft_adapter
set ADAPTER_ID=train_20260205T015838743819Z_sft_adapter

echo Starting eval at %date% %time%...
docker compose -f graphrag_saas/docker-compose.yml -f graphrag_saas/docker-compose.gpu.yml run --rm backend python scripts/eval_json_v2.py --questions %QUESTIONS% --out-dir /app/reports/evals/%OUT_DIR% --prompt-asset %PROMPT% --llm-model %MODEL% --adapter-path %ADAPTER% --adapter-id %ADAPTER_ID% --repair-json 1 --max-context-chars 9000 --top-k 5 --provider local --max-new-tokens 512 --index-dir /app/data/index --flush-every 1

echo.
echo Eval finished at %date% %time% with error code %ERRORLEVEL%
pause
