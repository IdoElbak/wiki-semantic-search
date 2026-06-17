"""Embedding utilities (sentence-transformers/all-MiniLM-L6-v2 only)."""
from __future__ import annotations

import time
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("  Loading MiniLM model...", end=" ", flush=True)
        t0 = time.time()
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        _model.max_seq_length = 256
        print(f"✓ ({time.time()-t0:.1f}s)")
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 128) -> np.ndarray:
    """Return L2-normalized embeddings, shape (n, dim)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)

    model = get_model()
    cleaned = [str(t).strip() for t in texts]
    total = len(cleaned)
    n_batches = (total + batch_size - 1) // batch_size

    all_vectors = []
    t0 = time.time()

    with tqdm(
        total=total,
        desc="  Embedding",
        unit="chunk",
        bar_format="  [{bar:40}] {percentage:5.1f}%  {n_fmt}/{total_fmt} chunks  "
                   "{rate_fmt}  ETA {remaining}",
    ) as pbar:
        for i in range(n_batches):
            batch = cleaned[i * batch_size : (i + 1) * batch_size]
            vecs = model.encode(
                batch,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            all_vectors.append(np.asarray(vecs, dtype=np.float32))
            pbar.update(len(batch))

    elapsed = time.time() - t0
    vectors = np.vstack(all_vectors)
    print(f"  ✓ Embedded {total:,} chunks in {elapsed:.1f}s "
          f"({total/elapsed:.0f} chunks/sec)")
    return vectors


def embed_queries(queries: List[str], *, batch_size: int = 128) -> np.ndarray:
    return embed_texts(queries, batch_size=batch_size)