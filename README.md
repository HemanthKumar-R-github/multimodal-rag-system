# DocIntel — Multimodal Document Intelligence Platform (Flask)

You can upload PDFs, scanned pages, images, tables and charts, then ask questions about them. Answers come back grounded in the documents with numbered **source citations**. The retrieval and answer quality can be measured with a built-in **evaluation harness**.

```
             ┌──────────── Ingestion ────────────┐
 PDF/Image → │ PyMuPDF text blocks               │
             │ pdfplumber / PyMuPDF tables → MD  │──► chunks (text | table | image | chart | ocr)
             │ embedded images + vector charts   │        │            + page, bbox, image
             │   → VLM description / OCR         │        │
             │ scanned pages → VLM / Tesseract   │        ▼
             └───────────────────────────────────┘   SQLite store
                                                        │
                         ┌──────────────┬───────────────┘
                         ▼              ▼
                    BM25 (sparse)   FAISS IndexFlatIP (dense embeddings)
                         └──── RRF fusion (k=60) ────┘
                                     ▼
                     Reranker (cross-encoder | LLM | lexical)
                                     ▼
                 LLM answer constrained to numbered sources [n]
                   (extractive fallback when no LLM configured)
                                     ▼
          Evaluation: Hit/Recall/Precision@k, MRR, nDCG, latency,
          F1, ROUGE-L, faithfulness, citation precision/coverage, LLM judge
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # lightweight core (CPU, no torch)
# optional, best quality: local bi-encoder + cross-encoder reranker
pip install -r requirements-ml.txt
# optional OCR for scans: apt install tesseract-ocr  |  brew install tesseract  |  dnf install tesseract

cp .env.example .env                       # add LLM_API_KEY to enable LLM answers + VLM
python scripts/make_sample_docs.py --ingest   # demo corpus: text, table, chart, scanned page
python app.py                              # http://localhost:5000
```

Run the tests with `pytest`. To evaluate from the command line, run `python scripts/run_eval.py data/eval/sample_eval.json`.

## Components and fallbacks

Every stage has a production backend and a fallback that needs no extra dependencies, so the app works offline from the start.

| Stage | Preferred | Fallback |
|---|---|---|
| Text, layout | PyMuPDF blocks, de-hyphenation, paragraph-aware chunks with overlap | — |
| Tables | pdfplumber → Markdown (header repeated when split) | PyMuPDF `find_tables` |
| Figures/charts | VLM (GPT-4o-mini etc.) → kind, description, extracted data table | caption context + Tesseract OCR |
| Scanned pages | VLM transcription to markdown | Tesseract OCR |
| Dense embeddings | sentence-transformers `all-MiniLM-L6-v2` or API (`EMBEDDING_BACKEND=openai`) | hashed char/word n-gram vectors |
| Vector index | FAISS `IndexIDMap2(IndexFlatIP)` persisted to disk | NumPy |
| Sparse | BM25 (Lucene IDF), inverted index | — |
| Fusion | Reciprocal Rank Fusion | — |
| Reranker | cross-encoder `ms-marco-MiniLM-L-6-v2`, or `RERANKER_BACKEND=llm` | lexical-feature reranker |
| Generation | Any OpenAI-compatible LLM (Vercel AI Gateway, OpenAI, Ollama, vLLM) | extractive, cited sentences/table rows |

If you change the embedding model, the FAISS index is rebuilt automatically on startup.

## Evaluation

A dataset is a JSON file in `data/eval/`:

```json
{"name": "My QA", "items": [
  {"question": "What was Q3 operating margin?", "answer": "21%",
   "relevant": [{"doc": "report.pdf", "page": 2}, {"contains": "Q3"}]}
]}
```

- **Retrieval** (per config: `bm25`, `dense`, `hybrid`, `hybrid+rerank`): Hit@k, Recall@k (over relevance targets), Precision@k, MRR, nDCG@k, latency.
- **Answers**: token F1 and ROUGE-L against the reference, plus:
  - lexical faithfulness: share of answer sentences supported by the retrieved context
  - citation coverage: share of sentences that cite a source
  - citation precision: share of cited sources that are relevant
  - answer relevance
  - abstention rate
  - optional LLM-as-judge faithfulness and correctness
- **Synthetic sets**: the Evaluate page can generate QA pairs from indexed chunks, using the LLM when one is configured and a heuristic otherwise.

Runs are stored in SQLite and can be browsed in the UI.

## REST API

| Method | Path | Body / notes |
|---|---|---|
| GET | `/api/status` | backends, counts |
| GET/POST | `/api/documents` | multipart `files` (upload; ingested in background) |
| GET | `/api/documents/<id>/chunks` | parsed chunks |
| DELETE | `/api/documents/<id>` | removes from SQLite, FAISS, BM25 |
| POST | `/api/query` | `{question, mode: hybrid|bm25|dense, rerank, top_k, use_llm}` |
| GET | `/api/eval/datasets` · `/api/eval/runs[/<id>]` | |
| POST | `/api/eval/run` | `{file, configs, k, generate, use_llm, llm_judge}` |
| POST | `/api/eval/synthesize` | `{n}` |

```bash
curl -s localhost:5000/api/query -H 'Content-Type: application/json' \
  -d '{"question":"What capex did the board approve for 2026?"}' | jq .answer
```

## Project layout

```
app.py                     Flask app factory + security headers
config.py                  env-driven settings
docintel/
  ingestion/               parsers (PDF, image, text), chunker, OCR, VLM
  retrieval/               embeddings, BM25, FAISS, RRF hybrid, rerankers
  generation.py            cited answer generation (+ extractive fallback)
  evaluation/              metrics + evaluator + synthetic QA
  pipeline.py              orchestration, background ingestion
  store.py                 SQLite persistence
  web.py                   pages + JSON API
templates/, static/        UI (Ask · Documents · Evaluate)
scripts/                   sample corpus generator, CLI evaluation
data/eval/                 evaluation datasets
tests/                     unit tests
```

## Production notes

- Indexes are kept in process memory, so run **one worker process** with several threads, e.g. `gunicorn -w 1 --threads 8 app:app`. For multiple workers, move the vectors to a shared store such as pgvector, Qdrant or OpenSearch.
- For large corpora, use `faiss.IndexHNSWFlat` or IVF-PQ instead of the flat index.
- Set `SECRET_KEY`, put the app behind HTTPS, and add authentication before you expose uploads publicly.
