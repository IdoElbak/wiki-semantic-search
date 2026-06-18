from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

KIND_BONUS: Dict[str, float] = {
    "title": 0.060,
    "lead":  0.035,
    "chunk": 0.0,
}

TOPK_MEAN         = 3
DENSE_CANDIDATES  = 2000
RERANK_CANDIDATES = 350

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when",
    "where", "which", "who", "how", "did", "was", "were", "are", "about",
    "also", "into", "after", "before", "during", "over", "under", "they",
    "their", "can", "link", "links", "learned", "together", "served",
}

_CACHE: dict = {}
_VECTORS_CACHE: dict = {}
_PAGE_TO_CHUNKS_CACHE: dict = {}


def _load(artifacts_dir):
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_index(artifacts_dir)
    return _CACHE[key]


def _load_vectors(artifacts_dir, index):
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _VECTORS_CACHE:
        n, dim = index.ntotal, index.d
        vectors = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            index.reconstruct(i, vectors[i])
        _VECTORS_CACHE[key] = vectors
    return _VECTORS_CACHE[key]


def _build_page_to_chunks(page_ids: List[int]) -> Dict[int, np.ndarray]:
    key = id(page_ids)
    if key not in _PAGE_TO_CHUNKS_CACHE:
        tmp: Dict[int, List[int]] = defaultdict(list)
        for ci, pid in enumerate(page_ids):
            tmp[int(pid)].append(ci)
        _PAGE_TO_CHUNKS_CACHE[key] = {
            pid: np.asarray(idxs, dtype=np.int64)
            for pid, idxs in tmp.items()
        }
    return _PAGE_TO_CHUNKS_CACHE[key]


def _topk_mean(scores: np.ndarray, pool_k: int) -> float:
    if pool_k > 0 and scores.shape[0] > pool_k:
        scores = np.partition(scores, -pool_k)[-pool_k:]
    return float(scores.mean())


def _tokenize(query: str) -> List[str]:
    query = re.sub(
        r'\b(\d{3}0)s\b',
        lambda m: " ".join(str(int(m.group(1)) + i) for i in range(10)),
        query
    )
    toks = [
        t for t in TOKEN_RE.findall(query.lower())
        if len(t) >= 3 and t not in STOPWORDS
    ]
    result = list(toks)
    result += [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
    result += [f"{toks[i]}_{toks[i+1]}_{toks[i+2]}" for i in range(len(toks) - 2)]
    return result


def _compute_specificity(query: str, lexical: Dict) -> float:
    terms = lexical["terms"]
    raw_toks = [
        t for t in TOKEN_RE.findall(query.lower())
        if len(t) >= 3 and t not in STOPWORDS
    ]
    if not raw_toks:
        return 0.5
    idf_scores = [float(terms[t][0]) for t in raw_toks if t in terms]
    idf_signal = float(np.clip((np.mean(idf_scores) - 1.0) / 6.0, 0.0, 1.0)) if idf_scores else 0.0
    found = sum(1 for t in raw_toks if t in terms)
    coverage_signal = found / len(raw_toks)
    length_signal = float(np.clip((len(raw_toks) - 3) / 5.0, 0.0, 1.0))
    numeric_count = sum(1 for t in raw_toks if re.match(r'^\d+$', t))
    numeric_signal = float(np.clip(numeric_count / max(1, len(raw_toks)), 0.0, 1.0))
    return float(np.clip(
        0.40 * idf_signal + 0.30 * coverage_signal +
        0.20 * length_signal + 0.10 * numeric_signal,
        0.0, 1.0
    ))


def _dynamic_weights(query, lexical, base_wf, base_wb) -> Tuple[float, float]:
    spec = _compute_specificity(query, lexical)
    wf = base_wf * (1.5 - 0.9 * spec)
    wb = base_wb * (0.5 + 1.2 * spec)
    return wf, wb


def _stage1_candidates(
    faiss_indices, faiss_scores, page_ids, kinds, top_n
) -> List[int]:
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
        page_scores[pid] = float(np.mean(hits[:TOPK_MEAN]))
    return sorted(page_scores, key=page_scores.get, reverse=True)[:top_n]


def _stage2_rescore(
    query_vec, candidates, page_to_chunks, corpus_vectors, pool_k=TOPK_MEAN
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for pid in candidates:
        if pid not in page_to_chunks:
            continue
        chunk_idxs = page_to_chunks[pid]
        chunk_sims = corpus_vectors[chunk_idxs] @ query_vec
        scores[pid] = _topk_mean(chunk_sims, pool_k)
    return scores


def _lexical_scores(query, lexical) -> Dict[int, float]:
    docs  = lexical["docs"]
    terms = lexical["terms"]
    scores: Dict[int, float] = {}
    LENGTH_NORM  = 0.0018
    PHRASE_BOOST = 2.2
    for tok in _tokenize(query):
        item = terms.get(tok)
        if item is None:
            continue
        idf, postings = item
        if float(idf) < 1.2 or len(postings) > 6000:
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


def _rrf_fusion(dense, lexical, top_k, rrf_k, wf, wb) -> List[int]:
    rrf: Dict[int, float] = defaultdict(float)
    for rank, (pid, _) in enumerate(sorted(dense.items(), key=lambda x: x[1], reverse=True)):
        rrf[pid] += wf / (rrf_k + rank)
    max_lex = max(lexical.values()) if lexical else 1.0
    for rank, (pid, score) in enumerate(sorted(lexical.items(), key=lambda x: x[1], reverse=True)):
        rrf[pid] += wb * (score / max_lex) / (rrf_k + rank)
    return [pid for pid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def search_batch(queries, *, top_k=K_EVAL, artifacts_dir=None):
    index, meta = _load(artifacts_dir)
    page_ids = meta["page_ids"]
    kinds    = meta.get("kinds", ["chunk"] * len(page_ids))
    lexical  = meta["lexical"]

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    base_wf = float(os.environ.get("TUNE_WF",  "1.2"))
    base_wb = float(os.environ.get("TUNE_WB",  "1.0"))
    rrf_k   = int(os.environ.get("TUNE_RRF",   "120"))

    faiss_k = min(index.ntotal, max(DENSE_CANDIDATES, int(top_k * 60)))
    faiss_scores_all, faiss_indices_all = index.search(
        np.ascontiguousarray(query_vectors.astype(np.float32)), faiss_k
    )

    corpus_vectors = _load_vectors(artifacts_dir, index)
    page_to_chunks = _build_page_to_chunks(page_ids)

    results = []
    for i, query in enumerate(queries):
        candidates = _stage1_candidates(
            faiss_indices_all[i], faiss_scores_all[i],
            page_ids, kinds, RERANK_CANDIDATES
        )
        dense = _stage2_rescore(
            query_vectors[i], candidates, page_to_chunks, corpus_vectors
        )
        lex = _lexical_scores(query, lexical)
        wf, wb = _dynamic_weights(query, lexical, base_wf, base_wb)
        results.append(_rrf_fusion(dense, lex, top_k, rrf_k, wf, wb))

    return results