"""Query-time retrieval pipeline: dense + lexical + fusion."""
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

# Score bonus added to a chunk's similarity before page aggregation,
# based on the chunk's kind. Title and lead chunks are rewarded because
# they tend to be more representative of the page topic.
KIND_BONUS: Dict[str, float] = {
    "title": 0.060,
    "lead":  0.035,
    "chunk": 0.0,
}

TOPK_MEAN         = 3     # Number of top chunk scores averaged per page.
DENSE_CANDIDATES  = 2000  # FAISS chunks retrieved per query in Stage 1.
RERANK_CANDIDATES = 350   # Pages kept for Stage 2 dense rescoring.

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when",
    "where", "which", "who", "how", "did", "was", "were", "are", "about",
    "also", "into", "after", "before", "during", "over", "under", "they",
    "their", "can", "link", "links", "learned", "together", "served",
}

# Module-level caches so artifacts are loaded only once per process.
_CACHE: dict = {}
_VECTORS_CACHE: dict = {}
_PAGE_TO_CHUNKS_CACHE: dict = {}


def _load(artifacts_dir):
    """Load and cache the FAISS index and metadata from disk.

    Args:
        artifacts_dir: Path to the artifacts directory, or None for the default.

    Returns:
        Tuple of (faiss.Index, metadata dict) as returned by load_index().
    """
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _CACHE:
        _CACHE[key] = load_index(artifacts_dir)
    return _CACHE[key]


def _load_vectors(artifacts_dir, index):
    """Reconstruct and cache all corpus vectors from the FAISS index.

    Materializes every vector stored in the index into a NumPy array so
    that Stage 2 rescoring can compute dot products without repeated FAISS
    calls.

    Args:
        artifacts_dir: Path to the artifacts directory, or None for the default.
        index:         The loaded FAISS index.

    Returns:
        Float32 array of shape (n_vectors, dim).
    """
    key = str((artifacts_dir or ARTIFACTS_DIR).resolve())
    if key not in _VECTORS_CACHE:
        n, dim = index.ntotal, index.d
        vectors = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            index.reconstruct(i, vectors[i])
        _VECTORS_CACHE[key] = vectors
    return _VECTORS_CACHE[key]


def _build_page_to_chunks(page_ids: List[int]) -> Dict[int, np.ndarray]:
    """Build and cache a mapping from page_id to its FAISS vector indices.

    Args:
        page_ids: Ordered list of page_id values, one per FAISS vector.

    Returns:
        Dict mapping each page_id to a sorted int64 array of vector indices.
    """
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
    """Return the mean of the top-pool_k values in scores.

    If scores has fewer than pool_k elements, all elements are averaged.

    Args:
        scores: 1-D array of similarity scores.
        pool_k: Number of top scores to average.

    Returns:
        Scalar mean of the selected scores.
    """
    if pool_k > 0 and scores.shape[0] > pool_k:
        scores = np.partition(scores, -pool_k)[-pool_k:]
    return float(scores.mean())


def _tokenize(query: str) -> List[str]:
    """Tokenize a query into unigrams, bigrams, and trigrams.

    Decade expressions such as "1990s" are first expanded to the individual
    years 1990–1999 so that numeric queries match lexical postings correctly.
    Tokens shorter than 3 characters or in STOPWORDS are discarded.

    Args:
        query: Raw query string.

    Returns:
        Flat list of unigram, bigram ("a_b"), and trigram ("a_b_c") features.
    """
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
    """Estimate how lexically specific a query is, as a score in [0, 1].

    Combines four signals:
        - IDF signal:      normalized mean IDF of query terms found in the index.
        - Coverage signal: fraction of query tokens present in the lexical index.
        - Length signal:   normalized query length (longer → more specific).
        - Numeric signal:  fraction of query tokens that are pure digits.

    Weights: 0.40 · IDF + 0.30 · coverage + 0.20 · length + 0.10 · numeric.

    Args:
        query:   Raw query string.
        lexical: Lexical index dict (as stored in metadata["lexical"]).

    Returns:
        Specificity score in [0, 1]. Returns 0.5 for empty queries.
    """
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
    """Compute query-adaptive dense and lexical fusion weights.

    Higher specificity shifts weight toward the lexical signal; lower
    specificity shifts weight toward the dense (semantic) signal.

    Args:
        query:   Raw query string.
        lexical: Lexical index dict.
        base_wf: Base dense weight (default 1.2, overridable via TUNE_WF).
        base_wb: Base lexical weight (default 1.0, overridable via TUNE_WB).

    Returns:
        Tuple (wf, wb) — the dense and lexical weights for this query.
    """
    spec = _compute_specificity(query, lexical)
    wf = base_wf * (1.4 - 0.8 * spec)
    wb = base_wb * (0.4 + 1.4 * spec)
    return wf, wb


def _stage1_candidates(
    faiss_indices, faiss_scores, page_ids, kinds, top_n
) -> List[int]:
    """Select the top-N candidate pages from raw FAISS chunk results.

    Each chunk score is adjusted by its KIND_BONUS, then pages are ranked
    by the mean of their top-TOPK_MEAN adjusted scores.

    Args:
        faiss_indices: 1-D array of FAISS vector indices for one query.
        faiss_scores:  Corresponding similarity scores.
        page_ids:      List mapping vector index → page_id.
        kinds:         List mapping vector index → chunk kind string.
        top_n:         Number of candidate pages to return.

    Returns:
        List of up to top_n page_ids, sorted by descending Stage 1 score.
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
        page_scores[pid] = float(np.mean(hits[:TOPK_MEAN]))
    return sorted(page_scores, key=page_scores.get, reverse=True)[:top_n]


def _stage2_rescore(
    query_vec, candidates, page_to_chunks, corpus_vectors, pool_k=TOPK_MEAN
) -> Dict[int, float]:
    """Rescore candidate pages using all of their stored chunk vectors.

    Unlike Stage 1 (which uses only chunks returned by FAISS), Stage 2
    computes dot products with every chunk belonging to each candidate page,
    then takes the mean of the top-pool_k similarities.

    Args:
        query_vec:      1-D float32 query embedding (L2-normalized).
        candidates:     Ordered list of candidate page_ids from Stage 1.
        page_to_chunks: Mapping from page_id to its FAISS vector indices.
        corpus_vectors: Full corpus embedding matrix, shape (n, dim).
        pool_k:         Number of top chunk scores to average per page.

    Returns:
        Dict mapping page_id → Stage 2 dense score.
    """
    scores: Dict[int, float] = {}
    for pid in candidates:
        if pid not in page_to_chunks:
            continue
        chunk_idxs = page_to_chunks[pid]
        chunk_sims = corpus_vectors[chunk_idxs] @ query_vec
        scores[pid] = _topk_mean(chunk_sims, pool_k)
    return scores


def _lexical_scores(query, lexical) -> Dict[int, float]:
    """Compute BM25-inspired lexical scores for all matching pages.

    Scoring details:
        - Pure digit tokens receive a boost of 6.0 (numeric facts).
        - Bigrams and trigrams receive a PHRASE_BOOST per extra term.
        - Terms with IDF < 1.2 or document frequency > 6,000 are skipped.
        - Document length normalization uses a small additive factor.

    Args:
        query:   Raw query string.
        lexical: Lexical index dict with "docs" and "terms".

    Returns:
        Dict mapping page_id → lexical score (only pages with ≥1 match).
    """
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
    """Combine dense and lexical rankings via weighted reciprocal-rank fusion.

    The dense contribution uses rank only. The lexical contribution uses
    both rank and normalized lexical score, so pages with strong lexical
    evidence are rewarded beyond their rank position.

    Final score for page p:
        wf / (rrf_k + rank_dense(p))
        + wb * (lex_score(p) / max_lex) / (rrf_k + rank_lexical(p))

    Args:
        dense:   Dict of page_id → Stage 2 dense score.
        lexical: Dict of page_id → lexical score.
        top_k:   Number of page IDs to return.
        rrf_k:   Smoothing constant (default 120).
        wf:      Weight for the dense ranking contribution.
        wb:      Weight for the lexical ranking contribution.

    Returns:
        List of top_k page_ids sorted by descending fusion score.
    """
    rrf: Dict[int, float] = defaultdict(float)
    for rank, (pid, _) in enumerate(sorted(dense.items(), key=lambda x: x[1], reverse=True)):
        rrf[pid] += wf / (rrf_k + rank)
    max_lex = max(lexical.values()) if lexical else 1.0
    for rank, (pid, score) in enumerate(sorted(lexical.items(), key=lambda x: x[1], reverse=True)):
        rrf[pid] += wb * (score / max_lex) / (rrf_k + rank)
    return [pid for pid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]]


def search_batch(queries, *, top_k=K_EVAL, artifacts_dir=None):
    """Run the full retrieval pipeline for a batch of queries.

    Steps per query:
        1. Embed all queries with MiniLM.
        2. Run batched FAISS search (DENSE_CANDIDATES chunks per query).
        3. Stage 1: aggregate chunk hits into page candidates.
        4. Stage 2: rescore candidates using all their stored chunks.
        5. Compute lexical scores for the query.
        6. Compute query-adaptive fusion weights.
        7. Fuse dense and lexical rankings and return top_k page IDs.

    Artifacts, corpus vectors, and the page-to-chunk mapping are cached
    in module-level dicts and reused across calls within the same process.

    Fusion hyperparameters can be overridden via environment variables:
        TUNE_WF  — base dense weight   (default 1.2)
        TUNE_WB  — base lexical weight (default 1.0)
        TUNE_RRF — RRF smoothing k     (default 120)

    Args:
        queries:       List of raw query strings.
        top_k:         Number of page IDs to return per query (default K_EVAL=10).
        artifacts_dir: Override for the artifacts directory path.

    Returns:
        List of lists: one ranked list of page_id per query.
    """
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