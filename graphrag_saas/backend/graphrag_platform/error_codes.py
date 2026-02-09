from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

ErrorStage = Literal["ingest", "query", "benchmark", "train", "models", "unknown"]


@dataclass(frozen=True)
class CodedError:
    code: str
    stage: ErrorStage
    message: str
    detail: dict[str, Any] | None = None


def _name(e: BaseException) -> str:
    return type(e).__name__


def classify_error(*, e: BaseException, stage: ErrorStage) -> CodedError:
    msg = str(e) if str(e) else _name(e)

    # OCR / Tesseract issues
    if "Tesseract OCR is not installed" in msg or "TesseractNotFoundError" in msg:
        return CodedError(code="OCR_TESSERACT_NOT_FOUND", stage=stage, message=msg)
    if "OCR failed for image:" in msg:
        return CodedError(code="OCR_FAILED", stage=stage, message=msg)

    # Dataset / file issues
    if isinstance(e, FileNotFoundError) or "Dataset path does not exist" in msg:
        return CodedError(code="DATASET_NOT_FOUND", stage=stage, message=msg)

    # KB / index issues
    if "Knowledge base not initialised" in msg:
        return CodedError(code="KB_NOT_READY", stage=stage, message=msg)

    # Training dependency issues
    if "Missing PEFT dependency" in msg:
        return CodedError(code="TRAIN_DEP_PEFT_MISSING", stage=stage, message=msg)
    if "requires TRL" in msg or "Missing TRL" in msg:
        return CodedError(code="TRAIN_DEP_TRL_MISSING", stage=stage, message=msg)
    if "bitsandbytes" in msg and "unavailable" in msg:
        return CodedError(code="TRAIN_DEP_BITSANDBYTES_MISSING", stage=stage, message=msg)

    # IRS/Unsloth mode issues
    if "IRS requires BACKEND_MODE=IRS_UNSLOTH" in msg:
        return CodedError(code="IRS_MODE_REQUIRED", stage=stage, message=msg)
    if "IRS module not available" in msg:
        return CodedError(code="IRS_NOT_AVAILABLE", stage=stage, message=msg)
    if "Unsloth:" in msg:
        return CodedError(code="IRS_UNSLOTH_ERROR", stage=stage, message=msg)

    # HTTP client issues (Ollama)
    if "OllamaTeacher failed" in msg:
        return CodedError(code="OLLAMA_TEACHER_FAILED", stage=stage, message=msg)

    return CodedError(code="UNCLASSIFIED", stage=stage, message=msg, detail={"type": _name(e)})

