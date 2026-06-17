# Section B — Wikipedia Semantic Search

**Course:** Data Analysis and Visualization 

**Team:** Ronit Ness · Ido Elbak

**Repository:** [https://github.com/IdoElbak/wiki-semantic-search](https://github.com/IdoElbak/wiki-semantic-search)

**Video Presentation:** _[link to be added]_

---

## Overview

An end-to-end retrieval pipeline over a corpus of Wikipedia-style articles. Given a natural-language query, the system returns a ranked list of the 10 most relevant `page_id` values, scored by **mean NDCG@10**.

The pipeline is built entirely on allowed dependencies: `sentence-transformers/all-MiniLM-L6-v2`, `faiss-cpu`, and `numpy`. No external ranking models are used.

---

## Quick Start

### 1. Prerequisites

```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):
```
numpy
sentence-transformers
faiss-cpu
```

### 2. Build the Index (offline, run once)

> The corpus must be in `data/Wikipedia Entries/` (one JSON file per page).
> The index is already prebuilt in `artifacts/` — you do **not** need to run this at evaluation time.

```bash
python scripts/build_index.py
```

Expected output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📂  [1/4] Loading Wikipedia entries...
    ✓ 27,xxx pages loaded
✂️   [2/4] Chunking corpus...
    ✓ ~120,000 chunks (avg 4–5 per page)
🔢  [3/4] Embedding chunks with MiniLM...
    [████████████████████] 100%
🗂️   [4/4] Building FAISS + BM25 indexes...
✅  Done — total ~10 min on GPU
```

### 3. Evaluate on Public Queries

```bash
python scripts/eval_public.py
```

Prints: `mean_ndcg@10=0.XXXX` over the 29 public queries.

---

## Artifacts

All files are committed under `artifacts/`.

| File | Format | Description |
|------|--------|-------------|
| `artifacts/corpus.index` | FAISS binary | IndexFlatIP over all chunk embeddings (dim=384) |
| `artifacts/index_meta.json` | JSON | `page_ids`, `chunk_ids`, `kinds` (title/lead/chunk) per vector |
| `artifacts/index_vectors.npy` | NumPy binary | Raw chunk embedding matrix (shape: n_chunks × 384) |
| `artifacts/bm25_index.json.gz` | gzip JSON | Custom BM25 inverted index with bigrams & trigrams |

`run()` calls `load_index()` which reads all four files from `artifacts/` at query time.

---

## Pipeline

### `chunk.py` — Section-Aware Chunking

Each Wikipedia page is split into multiple typed chunks:

1. **Title chunk** (`kind="title"`) — the page title alone; catches entity-name queries.
2. **Lead chunk** (`kind="lead"`) — `"{title}. {first paragraph}"`. The lead paragraph is the most fact-dense part of any Wikipedia article.
3. **Body chunks** (`kind="chunk"`) — section-aware if the article has ≥2 detected headers, otherwise a sliding window over sentences (window=7, stride=4). Every body chunk is prefixed with `"{title} - {section}."` so the entity name travels with each passage.

Average ~4–6 chunks per page, capped at 8.

### `embed.py` — Embedding

- Model: `sentence-transformers/all-MiniLM-L6-v2` (dim=384)
- `max_seq_length = 256` to prevent silent truncation
- `batch_size = 128`, L2-normalized output
- GPU-accelerated when available (Tesla M60 on Technion VM: ~10 min to embed the full corpus)

### `index.py` — Offline Index Build

**FAISS:** `IndexFlatIP` (exact inner product, equivalent to cosine on normalized vectors).

**BM25:** A custom implementation (no external BM25 library) with:
- Page-level documents (all chunks aggregated per page)
- Title text doubled — gives extra weight to entity name matches
- Features: unigrams + bigrams + trigrams (e.g. `"los_angeles"`, `"los_angeles_lakers"`)
- Low-frequency terms (`df ≤ 1`) and very common terms (`df > 3000`) pruned at build time
- Compressed with gzip (`bm25_index.json.gz`) for efficient storage

### `retrieve.py` — Two-Stage Retrieval with RRF Fusion

At query time, `run(queries)` in `main.py` delegates to `search_batch()` in `retrieve.py`, which runs the following steps:

1. **Load** the prebuilt index from `artifacts/` (cached after first call)
2. **Embed** all queries in one batch with MiniLM
3. **FAISS search** — `faiss_k = max(2000, top_k × 60)` candidates per query

**Stage 1 — Candidate aggregation:**
- Chunk scores aggregated to page level using **top-K mean pooling** (`TOPK_MEAN=3`): mean of the 3 best-scoring chunks per page
- `KIND_BONUS` applied: +0.060 for title chunks, +0.035 for lead chunks

**Stage 2 — Dense re-score:**
- Top 350 candidates re-scored using exact dot products against the full corpus vector matrix
- Eliminates any approximation error for the shortlist

**Lexical scoring (BM25-style):**
- Custom TF-IDF with IDF threshold ≥ 1.2, phrase boost × 2.2 for bigrams/trigrams
- Numbers get a dedicated boost (× 6.0) — important for population, date, and score queries
- **Decade expansion**: `"1820s"` → tokens `1820 1821 … 1829` for temporal queries
- **Year signal**: IDF-weighted exact year matches, added as a bonus score
- **Number signal**: overlap between numeric tokens in the query and the page

**Fusion:**
- Reciprocal Rank Fusion (RRF) combining dense + lexical + year + number signals
- Dynamic per-query weighting: a specificity score (derived from token IDF and coverage) adjusts `wf`/`wb` at query time
- Returns top-10 `page_id` per query; all query-time computation runs on GPU (≪ 60 s for 50 queries)
- Best hyperparameters found by grid search: `WF=1.2, WB=1.0, RRF_k=120`

---

## Empirical Results

All scores are mean NDCG@10 on the 29 public queries.

| Version | `chunk.py` | `index.py` | `retrieve.py` | NDCG@10 |
|---------|-----------|-----------|--------------|---------|
| **Baseline** | 1 chunk/page — title + first 200 words only | Basic FAISS flat + single-document BM25 | Simple RRF; no chunk aggregation | 0.320 |
| **Chunking overhaul** | Section-aware: title chunk + lead chunk + body chunks (sliding window over sentences, window=7, stride=4); title prefix on every chunk; ~4–6 chunks/page | Rebuilt with multi-chunk pages; BM25 now indexes full page vocabulary | Max-pooling to collapse multiple chunks per page to a single page score | 0.408 |
| **Hyperparameter tuning** | — | — | Grid-searched WF, WB, RRF_k over 80 combinations; best: WF=1.0, WB=0.8, RRF_k=90 | 0.421 |
| **Candidate pool + temporal expansion** | — | Title text doubled in BM25 for entity-name boost; bigrams/trigrams added | FAISS candidates raised to 2000; decade tokens expanded ("1820s" → 1820–1829); PRF removed (hurt performance) | 0.439 |
| **Top-5 mean pooling** | — | — | Chunk aggregation changed from top-3 to top-5 mean per page | 0.440 |
| **Two-stage rescore + signals** | — | — | Exact dot-product re-score of top-350 candidates; year-match and number-overlap bonus signals; dynamic per-query RRF weighting | **0.448** |

**Key findings from ablation:**
- PRF (Pseudo-Relevance Feedback) slightly hurt performance — removed in final version
- `RRF_k = 120` outperforms smaller values across all variants
- Chunking quality (section-aware structure, title prefix on every chunk) was the single highest-impact change — moving from 1 chunk/page to typed multi-chunk pages accounted for most of the gain from 0.32 to 0.44+

---

## Project Structure

```
student/
├── main.py            # Entry point: run(queries)
├── chunk.py           # Section-aware chunking
├── embed.py           # MiniLM embedding utilities
├── index.py           # Offline FAISS + BM25 index builder & loader
├── retrieve.py        # Query-time retrieval and fusion
├── utils.py           # Shared paths and helpers
├── eval.py            # NDCG@10 utilities (read-only)
├── requirements.txt
├── README.md
├── scripts/
│   ├── build_index.py      # Offline build script (read-only)
│   └── eval_public.py      # Self-evaluation on public queries (read-only)
├── data/
│   ├── public_queries.json
│   └── Wikipedia Entries/  # One JSON per page
└── artifacts/
    ├── corpus.index         # FAISS index
    ├── index_meta.json      # Chunk metadata (page_ids, kinds)
    ├── index_vectors.npy    # Raw chunk embedding matrix
    └── bm25_index.json.gz   # Compressed BM25 index
```
