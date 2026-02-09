from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .chunker import TextChunker
from .extractor import EntityExtractor
from .index_store import IndexMeta, IndexStore, compute_dataset_version
from .loaders import Loader, default_loaders, iter_supported_files


@dataclass(frozen=True)
class IngestResult:
    meta: IndexMeta
    warnings: list[str]


def ingest_dataset(
    dataset_path: str,
    *,
    index_dir: str,
    loaders: list[Loader] | None = None,
    chunker: TextChunker | None = None,
) -> IngestResult:
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    loaders = loaders or default_loaders()
    chunker = chunker or TextChunker(chunk_size=900, overlap=120)

    store = IndexStore(Path(index_dir))

    warnings: list[str] = []
    files: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    extractor = EntityExtractor()

    def pick_loader(p: Path) -> Loader | None:
        for l in loaders:
            if l.can_load(p):
                return l
        return None

    for p in iter_supported_files(root):
        loader = pick_loader(p)
        if not loader:
            continue

        try:
            stat = p.stat()
        except OSError:
            continue

        rel_path = os.path.relpath(p, root)
        files.append({"rel_path": rel_path, "size": stat.st_size, "mtime": int(stat.st_mtime)})

    version = compute_dataset_version(files)

    for f in files:
        rel_path = f["rel_path"]
        p = root / rel_path
        loader = pick_loader(p)
        if not loader:
            continue

        try:
            text = (loader.load_text(p) or "").strip()
        except Exception as e:
            warnings.append(f"Failed to load {rel_path}: {e}")
            continue

        if not text:
            warnings.append(f"Empty extracted text: {rel_path}")
            continue

        for chunk in chunker.chunk(text):
            chunk_id = len(chunks)
            chunks.append({"id": chunk_id, "source_path": rel_path, "text": chunk})
            try:
                extractor.extract(chunk)
            except Exception:
                # Best-effort entity extraction
                pass

    entity_names = {idx: name for name, idx in extractor.entities.items()}
    meta = store.write_index(
        dataset_path=str(root),
        version=version,
        files=files,
        chunks=chunks,
        entity_names=entity_names,
    )

    return IngestResult(meta=meta, warnings=warnings)
