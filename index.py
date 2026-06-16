<<<<<<< HEAD
"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import gzip
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_NAME      = "corpus.index"
INDEX_META_NAME = "index_meta.json"
BM25_INDEX_NAME = "bm25_index.json.gz"

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "what", "when",
    "where", "which", "who", "how", "did", "was", "were", "are", "about",
    "also", "into", "after", "before", "during", "over", "under", "they",
    "their", "can", "link", "links", "learned", "together", "served",
}


def _progress(label: str, current: int, total: int, start_time: float, width: int = 40) -> None:
    pct     = current / max(1, total)
    filled  = int(width * pct)
    bar     = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - start_time
    eta     = (elapsed / pct - elapsed) if pct > 0 else 0
    print(f"\r  [{bar}] {pct*100:5.1f}%  {current}/{total}  "
          f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s  — {label}",
          end="", flush=True)


def _tokens(text: str) -> List[str]:
    return [
        t for t in TOKEN_RE.findall(text.lower())
        if len(t) >= 3 and t not in STOPWORDS
    ]


def _features(text: str) -> List[str]:
    toks  = _tokens(text)
    feats = list(toks)
    feats += [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks) - 1)]
    feats += [f"{toks[i]}_{toks[i+1]}_{toks[i+2]}" for i in range(len(toks) - 2)]
    return feats


def _build_bm25(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Page-level BM25 with bigrams/trigrams. Title doubled for entity boost."""
    docs: List[Dict[str, Any]] = []
    postings: Dict[str, Dict[int, int]] = {}

    total = len(records)
    t0 = time.time()

    for i, record in enumerate(records):
        if i % 500 == 0 or i == total - 1:
            _progress("building BM25", i + 1, total, t0)

        page_id = int(record["page_id"])
        title   = str(record.get("title", ""))
        content = " ".join(str(record.get("content", "")).split()[:1200])

        # Title doubled — gives extra weight to entity name matching
        counts: Dict[str, int] = {}
        for feat in _features(title + " " + title + " " + content):
            counts[feat] = counts.get(feat, 0) + 1

        docs.append({"page_id": page_id, "length": max(1, sum(counts.values()))})
        for feat, count in counts.items():
            postings.setdefault(feat, {})[i] = count

    print()
    n_docs = len(docs)
    terms: Dict[str, Any] = {}
    for feat, posting in postings.items():
        df = len(posting)
        if (df <= 1 and not feat.isdigit()) or df > 3000:
            continue
        idf = math.log((n_docs + 1) / (df + 0.5))
        terms[feat] = [round(idf, 6), list(posting.items())]

    print(f"  ✓ BM25 done — {len(terms):,} terms ({time.time()-t0:.1f}s)")
    return {"docs": docs, "terms": terms}


def _build_faiss(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _save_artifacts(
    out_dir: Path,
    index: faiss.Index,
    chunks: List[Chunk],
    bm25: Dict[str, Any],
) -> None:
    faiss.write_index(index, str(out_dir / INDEX_NAME))
    meta: Dict[str, Any] = {
        "page_ids":    [c.page_id  for c in chunks],
        "chunk_ids":   [c.chunk_id for c in chunks],
        "kinds":       [getattr(c, "kind", "chunk") for c in chunks],
        "model":       "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(chunks),
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with gzip.open(out_dir / BM25_INDEX_NAME, "wt", encoding="utf-8") as f:
        json.dump(bm25, f, separators=(",", ":"))


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[int]]:

    total_start = time.time()
    out_dir = artifacts_dir or ensure_artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("━" * 60)
    print("📂  [1/4] Loading Wikipedia entries...")
    records = list(iter_entries(entries_dir))
    print(f"    ✓ {len(records):,} pages loaded ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("━" * 60)
    print("✂️   [2/4] Chunking corpus...")
    chunks = chunk_corpus(records)
    texts  = [c.text for c in chunks]
    print(f"    ✓ {len(chunks):,} chunks from {len(records):,} pages "
          f"({len(chunks)/max(1,len(records)):.1f} avg chunks/page, {time.time()-t0:.1f}s)")

    t0 = time.time()
    print("━" * 60)
    print(f"🔢  [3/4] Embedding {len(chunks):,} chunks with MiniLM...")
    vectors = embed_texts(texts)
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    print(f"    ✓ Vectors shape: {vectors.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("━" * 60)
    print("🗂️   [4/4] Building FAISS + BM25 indexes...")
    index = _build_faiss(vectors)
    bm25  = _build_bm25(records)
    print(f"    ✓ FAISS: {index.ntotal:,} vectors ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("━" * 60)
    print("💾  Saving artifacts...")
    _save_artifacts(out_dir, index, chunks, bm25)
    print(f"    ✓ Saved to: {out_dir}  ({time.time()-t0:.1f}s)")

    total = time.time() - total_start
    print("━" * 60)
    print(f"✅  Done — total time: {total:.1f}s ({total/60:.1f} min)")
    print("━" * 60)

    return index, [c.page_id for c in chunks]


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, Dict[str, Any]]:
    root  = artifacts_dir or ARTIFACTS_DIR
    index = faiss.read_index(str(root / INDEX_NAME))
    meta  = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    meta["page_ids"]  = [int(x) for x in meta["page_ids"]]
    meta["chunk_ids"] = [int(x) for x in meta["chunk_ids"]]
    with gzip.open(root / BM25_INDEX_NAME, "rt", encoding="utf-8") as f:
        meta["lexical"] = json.load(f)
    return index, meta
=======
"""Offline index compilation and storage using FAISS"""
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import faiss

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_NAME = "corpus.index"
INDEX_META_NAME = "index_meta.json"

def _compile_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Initialize an optimized flat inner-product vector search space"""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def _serialize_artifacts(out_dir: Path, index: faiss.Index, chunks: List[Chunk]) -> None:
    """Serialize the compiled index and contextual metadata to disk"""
    faiss.write_index(index, str(out_dir / INDEX_NAME))
    
    meta = {
        "page_ids": [c.page_id for c in chunks],
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(chunks),
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[int]]:
    """
    Main offline compilation pipeline
    
    Ingests raw JSON, splits text via sliding windows, embeds, 
    and saves optimized FAISS structures to disk
    """
    print("Loading Wikipedia entries...")
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    
    print("Executing chunking process...")
    chunks = chunk_corpus(records)
    texts = [c.text for c in chunks]
    
    print(f"Embedding {len(texts)} segments...")
    vectors = embed_texts(texts)
    
    print("Constructing localized FAISS network...")
    index = _compile_faiss_index(vectors)
    
    print("Writing structural artifacts to Disk...")
    _serialize_artifacts(out_dir, index, chunks)
    
    page_ids = [c.page_id for c in chunks]
    print(f"Pipeline successfully finalized. Target directory: {out_dir}")
    return index, page_ids

def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[faiss.Index, List[int]]:
    """Reconstruct an active FAISS framework from local storage files"""
    root = artifacts_dir or ARTIFACTS_DIR
    index = faiss.read_index(str(root / INDEX_NAME))
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return index, page_ids
>>>>>>> f187b524f361147680c03b3492c4d8df957fad66
