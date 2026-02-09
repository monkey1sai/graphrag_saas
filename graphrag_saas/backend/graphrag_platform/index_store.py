from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IndexMeta:
    dataset_path: str
    version: str
    created_at: str
    file_count: int
    chunk_count: int


class IndexStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.root / "index_meta.json"
        self.chunks_path = self.root / "chunks.jsonl"
        self.entity_names_path = self.root / "entity_names.json"
        self.files_path = self.root / "files.json"

    def has_index(self) -> bool:
        return self.meta_path.exists() and self.chunks_path.exists()

    def read_meta(self) -> IndexMeta | None:
        if not self.meta_path.exists():
            return None
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return IndexMeta(**data)

    def write_index(
        self,
        *,
        dataset_path: str,
        version: str,
        files: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        entity_names: dict[int, str],
    ) -> IndexMeta:
        created_at = datetime.now(timezone.utc).isoformat()
        meta = IndexMeta(
            dataset_path=dataset_path,
            version=version,
            created_at=created_at,
            file_count=len(files),
            chunk_count=len(chunks),
        )
        self.files_path.write_text(json.dumps(files, ensure_ascii=False, indent=2), encoding="utf-8")
        self.entity_names_path.write_text(
            json.dumps({str(k): v for k, v in entity_names.items()}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.chunks_path.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        self.meta_path.write_text(json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    def load_chunks(self) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        if not self.chunks_path.exists():
            return chunks
        with self.chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))
        return chunks

    def load_entity_names(self) -> dict[int, str]:
        if not self.entity_names_path.exists():
            return {}
        data = json.loads(self.entity_names_path.read_text(encoding="utf-8"))
        return {int(k): str(v) for k, v in data.items()}

    def load_files(self) -> list[dict[str, Any]]:
        if not self.files_path.exists():
            return []
        return json.loads(self.files_path.read_text(encoding="utf-8"))


def compute_dataset_version(files: list[dict[str, Any]]) -> str:
    """Fast fingerprint based on relative path + size + mtime.

    Avoids hashing full file contents (expensive for large images).
    """

    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.get("rel_path", "")):
        rel_path = str(f.get("rel_path", ""))
        size = str(f.get("size", 0))
        mtime = str(f.get("mtime", 0))
        h.update(rel_path.encode("utf-8", errors="ignore"))
        h.update(b"\0")
        h.update(size.encode("utf-8"))
        h.update(b"\0")
        h.update(mtime.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]
