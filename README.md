# Section B — Hybrid Retrieval Pipeline

This repository contains our solution for **Section B of Project A**.

The system receives a batch of natural-language queries and returns, for each query, a ranked list of Wikipedia page IDs. The final ranking combines semantic retrieval with exact lexical matching.

Only the first 10 page IDs are evaluated using **mean NDCG@10**.

> **Presentation video:** https://youtu.be/091c2a2tqJY

---

## Repository structure

```text
student/
├── main.py
├── chunk.py
├── embed.py
├── index.py
├── retrieve.py
├── utils.py
├── eval.py
├── requirements.txt
├── README.md
├── artifacts/
│   ├── corpus.index
│   ├── index_meta.json
│   └── bm25_index.json.gz
├── data/
│   ├── public_queries.json
│   └── Wikipedia Entries/
└── scripts/
    ├── build_index.py
    └── eval_public.py
```

The supplied evaluation files were not modified:

- `eval.py`
- `scripts/build_index.py`
- `scripts/eval_public.py`

---

## Setup

Run from the `student/` directory:

```bash
pip install -r requirements.txt
```

The corpus must be located at:

```text
data/Wikipedia Entries/
```

Each corpus file contains one page:

```json
{
  "page_id": 25051,
  "title": "...",
  "content": "..."
}
```

---

## Submitted artifacts

A fresh clone of the repository already contains all files required by `run()`.

| Artifact | Format | Contents |
|---|---|---|
| `artifacts/corpus.index` | FAISS binary index | Normalized MiniLM vectors for all chunks |
| `artifacts/index_meta.json` | JSON | The `page_id`, `chunk_id`, and chunk type associated with every vector |
| `artifacts/bm25_index.json.gz` | Gzipped JSON | Page-level lexical documents, term statistics, and inverted postings |

The pretrained MiniLM model is loaded through `sentence-transformers` and is not included in the repository.

Large artifact files may be stored with Git LFS.

---

## Build the index offline

The index is built locally and is not rebuilt during grading.

```bash
python scripts/build_index.py
```

The build process:

1. loads the Wikipedia entries;
2. creates the page chunks;
3. embeds all chunks with MiniLM;
4. builds the FAISS and lexical indexes;
5. saves the required files under `artifacts/`.

After rebuilding, the new artifact files must be committed to the repository.

---

## Run the public evaluation

```bash
python scripts/eval_public.py
```

This command verifies that the submitted artifacts can be loaded directly, without rebuilding the index.

The autograder calls:

```python
from main import run

results = run(queries)
```

`run()` returns one ranked list of integer page IDs for every query:

```python
list[list[int]]
```

---

## Retrieval pipeline

### 1. Chunking

Each page is represented by at most six chunks.

#### Title chunk

The page title is stored as a separate chunk with:

```text
kind = "title"
```

This helps queries that directly mention an entity or page name.

#### Lead chunk

The title and the first paragraph are combined into one chunk:

```text
title + lead paragraph
```

#### Body chunks

The remaining content is handled in one of two ways:

- **Pages with detected section headers:** paragraphs are grouped by section. Long sections may be split into two parts.
- **Pages without detected section headers:** the text is divided into overlapping windows of 5 sentences with a stride of 3, up to 4 body windows.

This keeps the number of vectors per page bounded while preserving local context.

---

### 2. Embeddings

Corpus chunks and queries are embedded with the required model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Configuration:

- vector dimension: `384`
- maximum sequence length: `256`
- batch size: `128`
- L2-normalized output vectors

Because the vectors are normalized, inner product is equivalent to cosine similarity.

---

### 3. Dense FAISS retrieval

All chunk vectors are stored in:

```python
faiss.IndexFlatIP
```

At query time, FAISS retrieves up to **2,000 chunk matches** for each query.

The matches are grouped by `page_id`. Before page aggregation, small fixed bonuses are added according to chunk type:

| Chunk type | Bonus |
|---|---:|
| Title | `0.060` |
| Lead | `0.035` |
| Regular chunk | `0.000` |

For each page, the system calculates the mean of its best **up to three** adjusted chunk scores. The best **350 pages** are retained as candidates.

These values are manually selected hyperparameters; they are not learned by the embedding model.

---

### 4. Dense page reranking

For each of the 350 retained pages, the system compares the query with all stored chunks belonging to that page.

The dense page score is the mean of the page's best up to three chunk similarities:

```text
dense_page_score = mean(top 3 query-to-chunk similarities)
```

The first dense stage is used for candidate selection. This second stage produces the dense page ranking used in the final fusion.

---

### 5. Lexical retrieval

A separate page-level inverted index is built from:

- unigrams
- adjacent bigrams
- adjacent trigrams

During indexing:

- text is converted to lowercase;
- short terms and selected stop words are removed;
- the title is included twice to strengthen direct title matches;
- at most the first 1,200 words of the page content are indexed.

The lexical score uses term rarity, term frequency, phrase boosts, numeric boosts, and document-length normalization.

For example, a query containing `1990s` is expanded to the years `1990` through `1999`.

The artifact is named `bm25_index.json.gz`, but the retrieval code uses a custom **BM25-inspired lexical score**, not a standard BM25 library implementation.

---

### 6. Query-dependent weights

The balance between dense and lexical retrieval changes according to the query.

The code calculates a specificity score from four signals:

```text
specificity =
    0.40 * IDF signal
  + 0.30 * lexical coverage
  + 0.20 * query-length signal
  + 0.10 * numeric signal
```

Here, **lexical coverage** is the fraction of filtered query terms that appear in the lexical index.

The final weights are:

```text
dense_weight   = 1.2 * (1.4 - 0.8 * specificity)
lexical_weight = 1.0 * (0.4 + 1.4 * specificity)
```

Therefore:

- broader queries receive more dense semantic weight;
- queries with rare terms, phrases, or numbers receive more lexical weight.

The coefficients are fixed hyperparameters selected during development.

---

### 7. Final fusion

The dense and lexical page rankings are combined with a weighted reciprocal-rank calculation.

For each page:

```text
final_score =
    dense_weight / (120 + dense_rank)
  + lexical_weight * normalized_lexical_score
    / (120 + lexical_rank)
```

Ranks are zero-based in the implementation. The lexical score is normalized by the largest lexical score for the same query.

Pages are sorted by `final_score`, and the top 10 unique page IDs are returned.

---

## Main hyperparameters

| Parameter | Value |
|---|---:|
| Maximum chunks per page | `6` |
| Sentence window size | `5` |
| Sentence stride | `3` |
| Maximum MiniLM sequence length | `256` |
| FAISS chunk matches | `2000` |
| Pages retained for reranking | `350` |
| Page aggregation | Mean of top `3` |
| Title bonus | `0.060` |
| Lead bonus | `0.035` |
| Base dense weight | `1.2` |
| Base lexical weight | `1.0` |
| Fusion constant | `120` |
| Evaluation cutoff | `10` |



## Query-time behavior

Corpus preprocessing and corpus embedding are completed during the offline build.

During grading, `run()` only:

1. loads the submitted artifacts;
2. embeds the query batch;
3. performs batched FAISS search;
4. selects and reranks page candidates;
5. calculates lexical scores;
6. fuses the rankings;
7. returns the top page IDs.

The loaded artifacts, reconstructed corpus vectors, and page-to-chunk mapping are cached in memory for reuse within the same process.
