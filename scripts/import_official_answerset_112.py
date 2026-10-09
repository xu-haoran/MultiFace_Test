"""Audit and import the 112 ``official_answerset`` problems into Chroma.

The command is dry-run by default.  It scans existing documents by normalized
problem text as well as by the stable ``official_answerset::<source_index>``
identifier, so reruns do not create duplicate records under another id.

Usage::

    python scripts/import_official_answerset_112.py SOURCE.md
    python scripts/import_official_answerset_112.py SOURCE.md --apply
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from pathlib import Path


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.database_client import DatabaseClient  # noqa: E402


COLLECTION = "olympiad_problems"
MAX_TEXT_CHARS = 5000
_SECTION_RE = re.compile(
    r"^##\s+第\s*(\d+)\s*题（source_index=(\d+)）\s*$"
    r"\s*^###\s+题面\s*$\s*(.*?)\s*"
    r"^###\s+数据库解答\s*$\s*(.*?)"
    r"(?=^##\s+第\s*\d+\s*题（source_index=\d+）\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BOILERPLATE_RE = re.compile(
    r"Remember\s+to\s+put\s+your\s+final\s+answer\s+within\s+"
    r"\\boxed\s*\{\s*\}\s*\.?",
    re.IGNORECASE,
)


def parse_source(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    rows = []
    for match in _SECTION_RE.finditer(text):
        question_number = int(match.group(1))
        source_index = int(match.group(2))
        problem = match.group(3).strip()
        solution = match.group(4).strip()
        if question_number != source_index + 1:
            raise ValueError(
                f"question/index mismatch: question={question_number}, "
                f"source_index={source_index}"
            )
        if not problem or not solution:
            raise ValueError(f"empty problem or solution at source_index={source_index}")
        rows.append(
            {
                "question_number": question_number,
                "source_index": source_index,
                "problem": problem,
                "solution": solution,
            }
        )
    indices = [row["source_index"] for row in rows]
    if indices != list(range(112)):
        raise ValueError(
            f"expected consecutive source_index 0..111, got {len(rows)} rows: "
            f"{indices[:5]}...{indices[-5:] if indices else []}"
        )
    return rows


def parse_problem(document: str) -> str:
    match = re.search(
        r"^[ \t]*##[ \t]+Problem\b[ \t]*(.*?)"
        r"(?=^[ \t]*##[ \t]+Solution\b|\Z)",
        str(document or ""),
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def problem_key(problem: str) -> str:
    text = unicodedata.normalize("NFKC", str(problem or ""))
    text = _BOILERPLATE_RE.sub("", text)
    replacements = {
        r"\leqslant": "<=", r"\leq": "<=", "≤": "<=",
        r"\geqslant": ">=", r"\geq": ">=", "≥": ">=",
        r"\neq": "!=", "≠": "!=",
        r"\cdot": "*", r"\times": "*", "×": "*",
        r"\ldots": "...", r"\dots": "...", "…": "...",
        "−": "-", "–": "-", "—": "-",
        r"\(" : "", r"\)": "", r"\[": "", r"\]": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    # PDF text extraction commonly flattens x_i to xi.  Braces and underscores
    # are TeX layout here, not mathematical content.
    text = text.replace("_", "").replace("{", "").replace("}", "")
    text = re.sub(r"[^0-9a-zA-Z一-鿿+\-*/=<>|]+", "", text)
    return text.casefold()


def existing_problem_map(collection, wanted_keys: set[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    offset = 0
    while True:
        got = collection.get(
            limit=1000, offset=offset, include=["documents"]
        )
        ids = got.get("ids") or []
        documents = got.get("documents") or []
        for record_id, document in zip(ids, documents):
            key = problem_key(parse_problem(document))
            if key in wanted_keys:
                found.setdefault(key, []).append(str(record_id))
        if len(ids) < 1000:
            break
        offset += len(ids)
    return found


def stable_id_map(collection, rows: list[dict]) -> dict[str, str]:
    ids = [f"official_answerset::{row['source_index']}" for row in rows]
    got = collection.get(ids=ids, include=["documents"])
    return {
        str(record_id): parse_problem(document)
        for record_id, document in zip(got.get("ids") or [], got.get("documents") or [])
    }


def encode_batches(model, documents: list[str], batch_size: int = 8):
    chunks = []
    for start in range(0, len(documents), batch_size):
        chunks.append(
            model.encode(
                documents[start : start + batch_size],
                batch_size=min(batch_size, len(documents) - start),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )
    if not chunks:
        return []
    import numpy as np

    return np.vstack(chunks).tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    rows = parse_source(args.source.resolve())
    device = args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    db = DatabaseClient(device=device)
    db._load_collection()
    collection = db._collection
    before = collection.count()

    wanted = {problem_key(row["problem"]): row for row in rows}
    if len(wanted) != len(rows):
        raise ValueError("source contains duplicate normalized problem texts")
    by_text = existing_problem_map(collection, set(wanted))
    by_id = stable_id_map(collection, rows)

    conflicts = []
    missing = []
    present_by_id = 0
    present_by_other_id = 0
    for row in rows:
        record_id = f"official_answerset::{row['source_index']}"
        key = problem_key(row["problem"])
        if record_id in by_id:
            if problem_key(by_id[record_id]) != key:
                conflicts.append(record_id)
            else:
                present_by_id += 1
            continue
        if key in by_text:
            present_by_other_id += 1
            continue
        missing.append(row)

    print(f"source_rows={len(rows)}")
    print(f"db_before={before}")
    print(f"present_by_stable_id={present_by_id}")
    print(f"present_by_other_id={present_by_other_id}")
    print(f"missing={len(missing)}")
    print(f"stable_id_conflicts={len(conflicts)}")
    print("missing_source_indices=" + ",".join(
        str(row["source_index"]) for row in missing
    ))
    if conflicts:
        print("conflict_ids=" + ",".join(conflicts))
        return 2
    if not args.apply or not missing:
        print("mode=" + ("apply_nothing_to_do" if args.apply else "dry_run"))
        return 0

    db._load_model()
    documents = []
    ids = []
    metadatas = []
    for row in missing:
        idx = row["source_index"]
        problem = row["problem"][:MAX_TEXT_CHARS]
        solution = row["solution"][:MAX_TEXT_CHARS]
        documents.append(f"## Problem\n{problem}\n\n## Solution\n{solution}".strip())
        ids.append(f"official_answerset::{idx}")
        metadatas.append(
            {
                "source": args.source.name,
                "source_dataset": "official_answerset",
                "dataset": "official_answerset",
                "contest": "official_answerset",
                "year": "",
                "idx": idx,
                "record_idx": idx,
                "solution_type": "worked_solution",
                "trust_level": "source_solution",
            }
        )

    started = time.time()
    embeddings = encode_batches(db._model, documents)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    after = collection.count()
    print(f"inserted={len(ids)}")
    print(f"db_after={after}")
    print(f"elapsed_seconds={time.time() - started:.1f}")
    if after != before + len(ids):
        raise RuntimeError(
            f"count mismatch: expected {before + len(ids)}, got {after}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
