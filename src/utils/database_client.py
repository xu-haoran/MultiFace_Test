"""ChromaDB client for unified mathematics-problem retrieval.

Loads Qwen3-Embedding-0.6B from ModelScope cache (CPU) and queries the
pre-built ChromaDB index. Returns top-k similar problems+solutions.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DatabaseClient:
    """Client for querying the olympiad + undergraduate ChromaDB collection."""

    def __init__(
        self,
        db_dir: str | Path | None = None,
        collection_name: str = "olympiad_problems",
        model_id: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cpu",
    ):
        """Initialize the database client.

        Args:
            db_dir: path to ChromaDB directory; relative paths are resolved
                against the submission root rather than the process cwd
            collection_name: ChromaDB collection name
            model_id: embedding model ID (ModelScope)
            device: "cpu" or "cuda"
        """
        raw_db_dir = Path(db_dir) if db_dir is not None else Path("data") / "database"
        self.db_dir = (
            raw_db_dir.resolve()
            if raw_db_dir.is_absolute()
            else (PROJECT_ROOT / raw_db_dir).resolve()
        )
        self.collection_name = collection_name
        self.model_id = model_id
        self.device = device
        self._model = None
        self._client = None
        self._collection = None

    def _load_model(self):
        """Lazy-load embedding model from ModelScope cache."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        model_dir = self._resolve_model_dir()
        self._model = SentenceTransformer(
            str(model_dir), device=self.device, trust_remote_code=True
        )
        self._model.max_seq_length = 512

    def _resolve_model_dir(self) -> Path:
        """Find model in project directory or ModelScope cache."""
        project_model = (PROJECT_ROOT / "resources" / "models" /
                         "Qwen3-Embedding-0.6B" / "snapshots" / "master")
        weights = project_model / "model.safetensors"
        if ((project_model / "config.json").exists()
                and weights.exists() and weights.stat().st_size > 1_000_000):
            return project_model

        try:
            from modelscope import snapshot_download
            return Path(snapshot_download(self.model_id))
        except Exception:
            pass

        for cand in (
            Path.home() / ".cache/modelscope/models/Qwen--Qwen3-Embedding-0.6B/snapshots/master",
            Path.home() / ".cache/modelscope/hub/Qwen/Qwen3-Embedding-0.6B",
        ):
            if (cand / "config.json").exists():
                return cand

        raise FileNotFoundError(
            f"Model {self.model_id} not found. Expected locations:\n"
            f"  1. Project-local: {project_model}\n"
            f"  2. ModelScope cache: ~/.cache/modelscope/\n"
            "To download: pip install modelscope && python -c "
            "\"from modelscope import snapshot_download; "
            "snapshot_download('Qwen/Qwen3-Embedding-0.6B')\""
        )

    def _load_collection(self):
        """Lazy-load ChromaDB collection."""
        if self._collection is not None:
            return

        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        import chromadb

        if not (self.db_dir / "chroma.sqlite3").exists():
            raise FileNotFoundError(
                f"ChromaDB not found at {self.db_dir}. "
                "Ensure the repository contains data/database/chroma.sqlite3."
            )

        # Keep an absolute path throughout the client lifetime.  The legacy
        # persistent backend resolves its vector-segment directory lazily;
        # temporarily changing cwd would make that directory relative to the
        # caller's cwd after initialization, leaving the index unloaded on
        # Windows paths containing non-ASCII characters.
        chroma_path = str(self.db_dir)
        try:
            # Use the Python persistent backend first.  The submission
            # bundles a portable hnswlib compatibility layer that stores
            # vectors.npy; the Rust backend can open the SQLite catalog
            # but cannot read that local index format after a restart.
            from chromadb.config import Settings
            settings = Settings(
                is_persistent=True,
                persist_directory=chroma_path,
                chroma_api_impl="chromadb.api.segment.SegmentAPI",
                allow_reset=False,
            )
            self._client = chromadb.Client(settings)
            self._collection = self._client.get_collection(self.collection_name)
        except Exception as persistent_error:
            self._client = chromadb.PersistentClient(path=chroma_path)
            self._collection = self._client.get_collection(self.collection_name)

    def query(self, problem: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Query database for similar problems+solutions.

        Args:
            problem: problem text (query)
            top_k: number of results to return

        Returns:
            List of dicts with keys:
                - problem: problem text
                - solution: solution text
                - similarity: cosine similarity (0-1)
                - source: source file path
                - contest: contest name
                - year: year (if available)
        """
        self._load_model()
        self._load_collection()

        vec = self._model.encode(
            [problem], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()

        res = self._collection.query(
            query_embeddings=[vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        for record_id, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            problem_text, solution_text = self._parse_document(doc)
            results.append({
                "record_id": str(record_id),
                "problem": problem_text,
                "solution": solution_text,
                "similarity": 1 - dist,  # cosine distance -> similarity
                "source": meta.get("source", ""),
                "source_dataset": meta.get("source_dataset") or meta.get("dataset", "olympiad"),
                "solution_type": meta.get("solution_type", "worked_solution"),
                "trust_level": meta.get("trust_level", "source_solution"),
                "subject": meta.get("subject", ""),
                "reference_answer": meta.get("reference_answer", ""),
                "record_idx": meta.get("record_idx", ""),
                "contest": meta.get("contest", ""),
                "year": meta.get("year", ""),
            })

        return results

    def query_many(self, problems: List[str], top_k: int = 3) -> List[List[Dict[str, Any]]]:
        """Batch dense retrieval for long-problem views.

        Hybrid retrieval may create up to four compact views of one prompt. A
        single embedding pass avoids repeatedly invoking the local encoder while
        preserving the same Chroma ranking and metadata contract as ``query``.
        """
        values = [str(problem or "") for problem in problems]
        if not values:
            return []
        self._load_model()
        self._load_collection()
        vectors = self._model.encode(
            values, normalize_embeddings=True, convert_to_numpy=True,
            batch_size=min(8, len(values)),
        ).tolist()
        res = self._collection.query(
            query_embeddings=vectors,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        output: List[List[Dict[str, Any]]] = []
        for row_index in range(len(values)):
            row: List[Dict[str, Any]] = []
            ids = (res.get("ids") or [[]])[row_index]
            docs = (res.get("documents") or [[]])[row_index]
            metas = (res.get("metadatas") or [[]])[row_index]
            distances = (res.get("distances") or [[]])[row_index]
            for record_id, doc, meta, dist in zip(ids, docs, metas, distances):
                meta = meta or {}
                problem_text, solution_text = self._parse_document(doc)
                row.append({
                    "record_id": str(record_id),
                    "problem": problem_text,
                    "solution": solution_text,
                    "similarity": 1 - dist,
                    "source": meta.get("source", ""),
                    "source_dataset": meta.get("source_dataset") or meta.get("dataset", "olympiad"),
                    "solution_type": meta.get("solution_type", "worked_solution"),
                    "trust_level": meta.get("trust_level", "source_solution"),
                    "subject": meta.get("subject", ""),
                    "reference_answer": meta.get("reference_answer", ""),
                    "record_idx": meta.get("record_idx", ""),
                    "contest": meta.get("contest", ""),
                    "year": meta.get("year", ""),
                })
            output.append(row)
        return output

    def fetch_by_ids(
        self,
        record_ids: List[str],
        problem: str,
        lexical_scores: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        """Load sparse-only candidates using the legacy-safe Chroma API.

        The bundled HNSW format supports ``collection.query`` but some Chroma
        versions cannot read stored embeddings back through ``get``. Sparse
        candidates retain the lexical score supplied by the caller.
        No network or answer extraction is involved.
        """
        if not record_ids:
            return []
        self._load_collection()
        requested_ids = [str(value) for value in record_ids]
        res = self._collection.get(
            ids=requested_ids,
            include=["documents", "metadatas"],
        )
        results: List[Dict[str, Any]] = []
        lexical_scores = lexical_scores or {}
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metadatas = res.get("metadatas", [])
        for record_id, doc, meta in zip(ids, docs, metadatas):
            meta = meta or {}
            problem_text, solution_text = self._parse_document(doc)
            record_key = str(record_id)
            similarity = float(lexical_scores.get(record_key, 0.0))
            results.append({
                "record_id": str(record_id),
                "problem": problem_text,
                "solution": solution_text,
                "similarity": similarity,
                "source": meta.get("source", ""),
                "source_dataset": meta.get("source_dataset") or meta.get("dataset", "olympiad"),
                "solution_type": meta.get("solution_type", "worked_solution"),
                "trust_level": meta.get("trust_level", "source_solution"),
                "subject": meta.get("subject", ""),
                "reference_answer": meta.get("reference_answer", ""),
                "record_idx": meta.get("record_idx", ""),
                "contest": meta.get("contest", ""),
                "year": meta.get("year", ""),
            })
        return results

    @staticmethod
    def _parse_document(doc: str) -> tuple[str, str]:
        """Parse a document into (problem, solution) pair.

        Documents are stored as:
            <problem text>

            <solution text>
        """
        prob_match = re.search(
            r'^[ \t]*##[ \t]+Problem\b[ \t]*(.*?)(?=^[ \t]*##[ \t]+Solution\b|\Z)',
            doc, re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        sol_match = re.search(
            r'^[ \t]*##[ \t]+Solution\b[ \t]*(.*?)(?=^[ \t]*##[ \t]+Problem\b|\Z)',
            doc, re.DOTALL | re.IGNORECASE | re.MULTILINE
        )

        problem = prob_match.group(1).strip() if prob_match else ""
        solution = sol_match.group(1).strip() if sol_match else ""

        return problem, solution

