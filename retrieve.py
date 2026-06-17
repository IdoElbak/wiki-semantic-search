"""Retrieve C: No PRF + 2000 candidates + decade expansion."""
from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import ARTIFACTS_DIR, K_EVAL

KIND_BONUS: Dict[str, float] = {
    "title": 0.030,
    "lead":  0.020,
    "chunk": 0.0,
}

TOPK_MEAN = 5
DENSE_CANDIDATES = 2000

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when",
    "where", "which", "who", "how", "did", "was", "were", "are", "about",
    "also", "into", "after", "before", "during", "over", "under", "they",
    "their", "can", "link", "links", "learned", "together", "served",
}

_CACHE: dict = {}


def _load(artifacts_dir):
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_index(artifacts_dir)
    return _CACHE[key]


def _tokenize(query: str) -> List[str]:
    """Tokenize with bigrams, trigrams, and decade expansion."""
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


def _aggregate_dense(faiss_indices, faiss_scores, page_ids, kinds):
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
    return page_scores


def _lexical_scores(query, lexical):
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


def _rrf_fusion(dense, lexical, top_k, rrf_k, wf, wb):
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

    wf    = float(os.environ.get("TUNE_WF",  "1.0"))
    wb    = float(os.environ.get("TUNE_WB",  "0.8"))
    rrf_k = int(os.environ.get("TUNE_RRF",  "90"))

    faiss_k = min(index.ntotal, max(DENSE_CANDIDATES, int(top_k * 60)))

    faiss_scores_all, faiss_indices_all = index.search(
        np.ascontiguousarray(query_vectors.astype(np.float32)), faiss_k
    )

    results = []
    for i, query in enumerate(queries):
        dense = _aggregate_dense(faiss_indices_all[i], faiss_scores_all[i], page_ids, kinds)
        lex   = _lexical_scores(query, lexical)
        results.append(_rrf_fusion(dense, lex, top_k, rrf_k, wf, wb))
    return results
