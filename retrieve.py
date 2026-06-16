<<<<<<< HEAD
"""Query-time retrieval: FAISS + lexical scoring with RRF fusion + PRF."""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
=======
"""Query-time retrieval (timed portion includes query embedding)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
>>>>>>> f187b524f361147680c03b3492c4d8df957fad66

import numpy as np

from embed import embed_queries
from index import load_index
<<<<<<< HEAD
from utils import ARTIFACTS_DIR, K_EVAL

KIND_BONUS: Dict[str, float] = {
    "title": 0.030,
    "lead":  0.020,
    "chunk": 0.0,
}

# Top-K chunks to average per page (topk_mean aggregation)
TOPK_MEAN = 3

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when",
    "where", "which", "who", "how", "did", "was", "were", "are", "about",
    "also", "into", "after", "before", "during", "over", "under", "they",
    "their", "can", "link", "links", "learned", "together", "served",
}

PRF_ALPHA = 0.85
PRF_TOP_N = 5

_CACHE: dict = {}
_VECTORS_CACHE: dict = {}


def _load(artifacts_dir: Optional[Path]):
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_index(artifacts_dir)
    return _CACHE[key]


def _load_vectors(artifacts_dir: Optional[Path], index) -> np.ndarray:
    """Reconstruct corpus vectors from FAISS index for PRF."""
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _VECTORS_CACHE:
        n, dim = index.ntotal, index.d
        vectors = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            index.reconstruct(i, vectors[i])
        _VECTORS_CACHE[key] = vectors
    return _VECTORS_CACHE[key]


def _tokenize(query: str) -> List[str]:
    toks = [
        t for t in TOKEN_RE.findall(query.lower())
        if len(t) >= 3 and t not in STOPWORDS
    ]
    result = list(toks)
    result += [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
    result += [f"{toks[i]}_{toks[i+1]}_{toks[i+2]}" for i in range(len(toks) - 2)]
    return result


def _aggregate_dense(
    faiss_indices: np.ndarray,
    faiss_scores: np.ndarray,
    page_ids: List[int],
    kinds: List[str],
) -> Dict[int, float]:
    """
    Aggregate chunk scores to page level using top-K mean.
    Takes the mean of the best TOPK_MEAN chunks per page.
    """
    page_hits: Dict[int, List[float]] = defaultdict(list)

    for idx, score in zip(faiss_indices.tolist(), faiss_scores.tolist()):
        if idx < 0:
            continue
        pid  = int(page_ids[idx])
        kind = kinds[idx] if idx < len(kinds) else "chunk"
        page_hits[pid].append(float(score) + KIND_BONUS.get(kind, 0.0))

    page_scores: Dict[int, float] = {}
    for pid, hits in page_hits.items():
        hits.sort(reverse=True)
        top_hits = hits[:TOPK_MEAN]
        page_scores[pid] = float(np.mean(top_hits))

    return page_scores


def _lexical_scores(query: str, lexical: Dict) -> Dict[int, float]:
    docs  = lexical["docs"]
    terms = lexical["terms"]
    scores: Dict[int, float] = {}
    LENGTH_NORM  = 0.0018
    PHRASE_BOOST = 3.4

    for tok in _tokenize(query):
        item = terms.get(tok)
        if item is None:
            continue
        idf, postings = item
        if float(idf) < 1.4 or len(postings) > 4000:
            continue
        parts = tok.count("_") + 1
        boost = 6.0 if tok.isdigit() else (1.0 + PHRASE_BOOST * (parts - 1))
        for doc_idx, count in postings:
            doc    = docs[int(doc_idx)]
            pid    = int(doc["page_id"])
            length = float(doc["length"])
            tf     = 1.0 + math.log1p(float(count))
            scores[pid] = scores.get(pid, 0.0) + boost * float(idf) * tf / (1.0 + LENGTH_NORM * length)

    return scores


def _prf_expand(
    query_vec: np.ndarray,
    faiss_indices: np.ndarray,
    page_ids: List[int],
    corpus_vectors: np.ndarray,
    alpha: float = PRF_ALPHA,
    top_n: int = PRF_TOP_N,
) -> np.ndarray:
    """Rocchio PRF: expand query with centroid of top-N page vectors."""
    seen_pages: Dict[int, int] = {}
    for idx in faiss_indices:
        if idx < 0:
            continue
        pid = int(page_ids[idx])
        if pid not in seen_pages:
            seen_pages[pid] = int(idx)
        if len(seen_pages) >= top_n:
            break

    if not seen_pages:
        return query_vec

    feedback_vecs = np.stack([corpus_vectors[i] for i in seen_pages.values()])
    centroid = feedback_vecs.mean(axis=0)
    expanded = alpha * query_vec + (1.0 - alpha) * centroid
    norm = np.linalg.norm(expanded)
    return (expanded / norm).astype(np.float32) if norm > 0 else query_vec


def _rrf_fusion(
    dense: Dict[int, float],
    lexical: Dict[int, float],
    top_k: int,
    rrf_k: int,
    wf: float,
    wb: float,
) -> List[int]:
    rrf: Dict[int, float] = defaultdict(float)

    for rank, (pid, _) in enumerate(
        sorted(dense.items(), key=lambda x: x[1], reverse=True)
    ):
        rrf[pid] += wf / (rrf_k + rank)

    max_lex = max(lexical.values()) if lexical else 1.0
    for rank, (pid, score) in enumerate(
        sorted(lexical.items(), key=lambda x: x[1], reverse=True)
    ):
        rrf[pid] += wb * (score / max_lex) / (rrf_k + rank)

    return [
        pid for pid, _ in
        sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ]
=======
from utils import K_EVAL
>>>>>>> f187b524f361147680c03b3492c4d8df957fad66


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
<<<<<<< HEAD

    index, meta = _load(artifacts_dir)
    page_ids = meta["page_ids"]
    kinds    = meta.get("kinds", ["chunk"] * len(page_ids))
    lexical  = meta["lexical"]

=======
    """
    Return ranked page_id lists (best first) for each query.

    Default: brute-force dot product on L2-normalized vectors.
    Replace with FAISS / reranking as needed.
    """
    corpus_vectors, page_ids = load_index(artifacts_dir)
>>>>>>> f187b524f361147680c03b3492c4d8df957fad66
    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

<<<<<<< HEAD
    wf    = float(os.environ.get("TUNE_WF",  "1.0"))
    wb    = float(os.environ.get("TUNE_WB",  "1.5"))
    rrf_k = int(os.environ.get("TUNE_RRF",  "60"))

    n_pages    = len(lexical["docs"])
    avg_chunks = max(1.0, len(page_ids) / max(1, n_pages))
    faiss_k    = min(index.ntotal, int(top_k * avg_chunks * 15))

    # First FAISS pass
    faiss_scores_all, faiss_indices_all = index.search(
        np.ascontiguousarray(query_vectors.astype(np.float32)), faiss_k
    )

    # Load corpus vectors for PRF
    corpus_vectors = _load_vectors(artifacts_dir, index)

    # PRF: expand queries
    expanded_vectors = np.stack([
        _prf_expand(query_vectors[i], faiss_indices_all[i], page_ids, corpus_vectors)
        for i in range(len(queries))
    ]).astype(np.float32)

    # Second FAISS pass with expanded queries
    faiss_scores_prf, faiss_indices_prf = index.search(
        np.ascontiguousarray(expanded_vectors), faiss_k
    )

    results = []
    for i, query in enumerate(queries):
        # Merge both passes — take best score per page
        dense1 = _aggregate_dense(faiss_indices_all[i], faiss_scores_all[i], page_ids, kinds)
        dense2 = _aggregate_dense(faiss_indices_prf[i], faiss_scores_prf[i], page_ids, kinds)
        dense_combined = {**dense1}
        for pid, score in dense2.items():
            if pid not in dense_combined or score > dense_combined[pid]:
                dense_combined[pid] = score

        lex = _lexical_scores(query, lexical)
        results.append(_rrf_fusion(dense_combined, lex, top_k, rrf_k, wf, wb))

    return results
=======
    scores = query_vectors @ corpus_vectors.T
    ranked: List[List[int]] = []
    for row in scores:
        order = np.argsort(-row)
        seen: set[int] = set()
        ids: List[int] = []
        for idx in order:
            pid = page_ids[int(idx)]
            if pid in seen:
                continue
            seen.add(pid)
            ids.append(pid)
            if len(ids) >= top_k:
                break
        ranked.append(ids)
    return ranked
>>>>>>> f187b524f361147680c03b3492c4d8df957fad66
