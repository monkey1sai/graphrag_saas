"""IRS (Iterative Rejection Sampling) 模組

8GB VRAM 環境下的 DW-GRPO 替代方案：
- Generate: Unsloth 4-bit 推理，每題產 N 個候選
- Score: CPU 上跑 DW-GRPO 動態加權評分
- SFT: 用金牌答案微調 LoRA adapter
"""

from __future__ import annotations

import os

# 環境模式檢查
BACKEND_MODE = os.getenv("BACKEND_MODE", "")
IS_IRS_UNSLOTH = BACKEND_MODE == "IRS_UNSLOTH"

# 條件式導入 Unsloth
UNSLOTH_AVAILABLE = False
if IS_IRS_UNSLOTH:
    try:
        from unsloth import FastLanguageModel  # noqa: F401

        UNSLOTH_AVAILABLE = True
    except ImportError:
        pass

__all__ = [
    "BACKEND_MODE",
    "IS_IRS_UNSLOTH",
    "UNSLOTH_AVAILABLE",
]
