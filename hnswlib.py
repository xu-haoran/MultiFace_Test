"""Small NumPy-backed compatibility implementation for Chroma's local API.

The competition bundle contains legacy hnswlib files, while the Windows build
of Chroma 1.0.20 may not ship the native hnswlib wheel.  This module implements
the subset of the hnswlib API used by Chroma's ``SegmentAPI``.  It reads the
bundled hnswlib ``data_level0.bin`` format and persists new/updated vectors in
``vectors.npy``.  Search is exact cosine search; the corpus is small enough for
the evaluator's per-problem workload and this keeps the submission portable.
"""

from __future__ import annotations

import pickle
import struct
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


class Index:
    file_handle_count = 0

    def __init__(self, space: str = "l2", dim: int = 0):
        self.space = str(space)
        self.dim = int(dim)
        self._vectors: dict[int, np.ndarray] = {}
        self._max_elements = 0
        self._ef = 10
        self._folder: Optional[Path] = None

    def init_index(self, max_elements: int, ef_construction: int = 100,
                   M: int = 16, allow_replace_deleted: bool = False,
                   is_persistent_index: bool = False,
                   persistence_location: Optional[str] = None, **_: object) -> None:
        self._max_elements = int(max_elements)
        if persistence_location:
            self._folder = Path(persistence_location)
            self._folder.mkdir(parents=True, exist_ok=True)

    def load_index(self, path: str, max_elements: Optional[int] = None,
                   is_persistent_index: bool = False, **_: object) -> None:
        self._folder = Path(path)
        custom = self._folder / "vectors.npy"
        if custom.exists():
            array = np.load(custom, allow_pickle=False)
            if array.ndim == 2:
                self.dim = int(array.shape[1])
                self._vectors = {int(row[0]): np.asarray(row[1:], dtype=np.float32)
                                 for row in []}
                # vectors.npy is stored as a dense matrix and labels are kept
                # in a sidecar file to avoid object arrays.
                labels_path = self._folder / "vectors.labels.npy"
                labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
                self._vectors = {
                    int(label): np.asarray(vector, dtype=np.float32)
                    for label, vector in zip(labels.tolist(), array)
                }
                self._max_elements = int(max_elements or len(self._vectors))
                return

        header_path = self._folder / "header.bin"
        data_path = self._folder / "data_level0.bin"
        meta_path = self._folder / "index_metadata.pickle"
        if not (header_path.exists() and data_path.exists() and meta_path.exists()):
            self._max_elements = int(max_elements or 0)
            return
        header = header_path.read_bytes()
        current_count = struct.unpack_from("<I", header, 20)[0]
        size_per_element = struct.unpack_from("<I", header, 28)[0]
        label_offset = struct.unpack_from("<I", header, 36)[0]
        vector_offset = struct.unpack_from("<I", header, 44)[0]
        dimensionality = (label_offset - vector_offset) // 4
        persist = pickle.loads(meta_path.read_bytes())
        mapping = (persist.get("label_to_id", {}) if isinstance(persist, dict)
                   else getattr(persist, "label_to_id", {}))
        valid_labels = {int(k) for k in mapping.keys()}
        mm = np.memmap(data_path, dtype=np.uint8, mode="r")
        labels = np.ndarray((current_count,), dtype=np.uint64, buffer=mm,
                            offset=label_offset, strides=(size_per_element,))
        vectors = np.ndarray((current_count, dimensionality), dtype=np.float32,
                             buffer=mm, offset=vector_offset,
                             strides=(size_per_element, 4))
        self.dim = int(dimensionality)
        self._vectors = {
            int(label): np.asarray(vectors[i], dtype=np.float32).copy()
            for i, label in enumerate(labels.tolist()) if int(label) in valid_labels
        }
        self._max_elements = int(max_elements or struct.unpack_from("<I", header, 12)[0])

    def set_ef(self, ef: int) -> None:
        self._ef = int(ef)

    def set_num_threads(self, num_threads: int) -> None:
        return None

    def resize_index(self, new_size: int) -> None:
        self._max_elements = max(self._max_elements, int(new_size))

    def get_max_elements(self) -> int:
        return int(self._max_elements)

    def get_current_count(self) -> int:
        return len(self._vectors)

    def add_items(self, data: Iterable[Iterable[float]], ids: Optional[Iterable[int]] = None,
                  num_threads: int = -1, replace_deleted: bool = False) -> None:
        array = np.asarray(list(data), dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if self.dim == 0:
            self.dim = int(array.shape[1])
        labels = list(ids) if ids is not None else list(range(len(self._vectors),
                                                               len(self._vectors) + len(array)))
        for label, vector in zip(labels, array):
            self._vectors[int(label)] = np.asarray(vector, dtype=np.float32).copy()

    def mark_deleted(self, label: int) -> None:
        self._vectors.pop(int(label), None)

    def unmark_deleted(self, label: int) -> None:
        return None

    def get_items(self, labels: Iterable[int], return_type: str = "numpy") -> np.ndarray:
        return np.asarray([self._vectors[int(label)] for label in labels], dtype=np.float32)

    def knn_query(self, data: Iterable[Iterable[float]], k: int = 1,
                  filter: Optional[object] = None, num_threads: int = -1):
        queries = np.asarray(list(data), dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        labels = [label for label in self._vectors
                  if filter is None or bool(filter(label))]
        if not labels or k <= 0:
            return np.empty((len(queries), 0), dtype=np.int64), np.empty((len(queries), 0), dtype=np.float32)
        matrix = np.asarray([self._vectors[label] for label in labels], dtype=np.float32)
        if self.space in {"cosine", "ip"}:
            matrix_norm = np.linalg.norm(matrix, axis=1)
            query_norm = np.linalg.norm(queries, axis=1)
            scores = (queries @ matrix.T) / np.maximum(query_norm[:, None] * matrix_norm[None, :], 1e-12)
            distances = 1.0 - scores if self.space == "cosine" else -scores
        else:
            distances = ((queries[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
        take = min(int(k), len(labels))
        order = np.argpartition(distances, take - 1, axis=1)[:, :take]
        row_order = np.take_along_axis(distances, order, axis=1).argsort(axis=1)
        order = np.take_along_axis(order, row_order, axis=1)
        return (np.asarray([[labels[i] for i in row] for row in order], dtype=np.int64),
                np.take_along_axis(distances, order, axis=1).astype(np.float32))

    def persist_dirty(self) -> None:
        if self._folder is None:
            return
        self._folder.mkdir(parents=True, exist_ok=True)
        labels = np.asarray(sorted(self._vectors), dtype=np.int64)
        vectors = np.asarray([self._vectors[int(label)] for label in labels], dtype=np.float32)
        np.save(self._folder / "vectors.labels.npy", labels, allow_pickle=False)
        np.save(self._folder / "vectors.npy", vectors, allow_pickle=False)

    def save_index(self, path: str) -> None:
        self._folder = Path(path)
        self.persist_dirty()

    def open_file_handles(self) -> None:
        return None

    def close_file_handles(self) -> None:
        return None
