# NCERT GraphRAG

A **graph-based Retrieval-Augmented Generation** system over NCERT textbooks (grades 6–12).

Standard RAG retrieves text chunks by vector similarity and often misses anything the
question doesn't lexically resemble. This project adds a **knowledge-graph layer** so it
can reason across related concepts and answer both pinpoint and thematic questions —
always grounded in the source PDFs, with citations.

```
┌──────────────┐   embed query    ┌─────────┐   seed entities   ┌────────┐
│  Your query  │ ───────────────► │ Qdrant  │ ────────────────► │ Neo4j  │
└──────────────┘   (bge-m3)       │ vectors │                   │ graph  │
                                  └─────────┘                   └────┬───┘
                                       ▲  retrieve & rank chunks      │ expand
                                       │                              ▼ neighborhood
                                  ┌────┴─────┐   grounded answer  ┌────────────┐
                                  │   LLM    │ ◄───────────────── │  context   │
                                  │ (Gemini) │   (streamed)       │  builder   │
                                  └──────────┘                    └────────────┘
```

---

## Why GraphRAG?

| | Standard RAG | This project (GraphRAG) |
|---|---|---|
| Retrieval | top-k chunks by vector similarity | entities → **graph neighborhood** → chunks |
| Multi-hop reasoning | ✗ | ✓ (traverses relationships) |
| Thematic questions | weak | **Global search** over concept communities |
| Grounding | chunk text | chunk text **+ knowledge graph** |

Two query modes:

- **Local search** — for specific concepts (*"What is Newton's second law?"*). Embeds the
  query, finds seed entities, expands their neighborhood in the graph, then retrieves and
  ranks the linked source chunks.
- **Global search** — for broad/thematic questions (*"What themes does class-10 science
  cover?"*). Retrieves LLM-generated summaries of concept **communities**.

---

## Architecture

The system splits into an **offline ingestion** stage (heavy, GPU) and **online serving**
(lightweight, local).

### Offline ingestion — `run_pipeline.py`
1. **Parse** NCERT PDFs (PyMuPDF) → text + chapter detection.
2. **Chunk** hierarchically — L1 section / L2 paragraph / L3 atomic — with semantic
   boundaries (`src/ingestion/chunker.py`).
3. **Embed** each chunk with **BAAI/bge-m3** (dense + sparse hybrid vectors).
4. **Extract** entities & relationships per chunk with an LLM (**Qwen2.5-7B**), with robust
   JSON parsing that salvages truncated output (`src/ingestion/llm_output.py`).
5. **Build** a deduplicated knowledge graph (`src/ingestion/graph_builder.py`).
6. **Detect communities** with the **Leiden** algorithm and **summarize** them.

The pipeline is **resumable and idempotent** — each stage skips work whose output already
exists, and extraction checkpoints to a JSONL it can resume from.

### Online serving — `cli.py`
- `src/retrieval/local_search.py` / `global_search.py` — the two query modes.
- `src/storage/` — async Neo4j (graph) and Qdrant (hybrid vector) clients.
- `src/llm/client.py` — any OpenAI-compatible LLM (Gemini, Ollama, Groq, OpenAI).
- A `rich` terminal UI that **streams** answers token-by-token with source citations.

---

## Tech stack

**Python** · **Neo4j** (knowledge graph) · **Qdrant** (hybrid dense+sparse vectors) ·
**bge-m3** (embeddings) · **Qwen2.5-7B** (offline extraction) · **Gemini / Ollama**
(answering) · **PyMuPDF** · **igraph + Leiden** (communities) · `async` I/O throughout ·
`rich` (CLI). No LangChain — every stage is built from primitives.

---

## Quick start

### Prerequisites
- Python 3.10+
- Docker (for Neo4j + Qdrant)
- An OpenAI-compatible LLM: a free [Gemini key](https://aistudio.google.com/apikey), or
  local [Ollama](https://ollama.com).

### 1. Install & configure
```bash
pip install -r requirements.txt
cp .env.example .env          # then edit .env with your LLM provider + key
```

### 2. Start the databases
```bash
make up                        # docker compose up -d  (Neo4j :7687, Qdrant :6333)
```

### 3. Get the data into the stores
Ingestion runs on a GPU machine and produces artifacts; you then import them locally.

```bash
# (a) on a GPU machine — download PDFs and run the offline pipeline
python download_ncert.py
python run_pipeline.py --batch 32 --max-new-tokens 768   # -> ./artifacts/

# (b) locally — drop the artifacts in data/artifacts/ and import
python -m src.ingestion.import_artifacts
```

### 4. Ask questions
```bash
python cli.py
```
```
❯ explain osmosis
❯ local: what is Newton's second law?
❯ global: what topics does class 9 science cover?
```

---

## Project structure

```
.
├── cli.py                       # streaming rich terminal Q&A
├── run_pipeline.py              # offline ingestion orchestrator (resumable)
├── download_ncert.py            # fetch NCERT PDFs from ncert.nic.in
├── docker-compose.yml           # Neo4j + Qdrant
├── src/
│   ├── config.py                # pydantic-settings, reads .env
│   ├── models/                  # Pydantic schemas (chunks, entities, graph)
│   ├── ingestion/
│   │   ├── pdf_parser.py         # PDF text + chapter detection
│   │   ├── chunker.py            # hierarchical semantic chunking (L1/L2/L3)
│   │   ├── embedder.py           # bge-m3 dense + sparse embeddings
│   │   ├── extractor.py          # LLM entity/relationship extraction (batched)
│   │   ├── llm_output.py         # shared LLM-output parsing (salvages truncated JSON)
│   │   ├── graph_builder.py      # entity dedup + relationship build
│   │   ├── community_detector.py # Leiden community detection
│   │   ├── community_summarizer.py
│   │   └── import_artifacts.py   # load artifacts → Neo4j + Qdrant
│   ├── retrieval/
│   │   ├── local_search.py       # entity → graph → chunks
│   │   ├── global_search.py      # community-summary search
│   │   └── context_builder.py    # dedup, order, token-budget assembly
│   ├── storage/
│   │   ├── neo4j_client.py        # async graph queries + batched upserts
│   │   └── qdrant_client.py       # async hybrid vector search
│   └── llm/
│       ├── client.py              # OpenAI-compatible async client (stream + complete)
│       └── prompts.py
└── tests/                        # pytest
```

---

## Notable engineering details

- **No N+1 queries** — a single Cypher query expands all seed entities' neighborhoods in
  one round-trip (`get_neighbors_batch`), returning chunk ids inline.
- **GPU-batched extraction** — the HuggingFace pipeline is configured for true batched
  generation (left-padding + batch size), turning ~0.05 → ~7 chunks/sec on an A100.
- **Truncation-safe parsing** — `parse_json_block` salvages complete entities/relationships
  from JSON the model cut off at the token limit.
- **Provider-agnostic LLM** — the OpenAI SDK points at any compatible endpoint via env vars,
  so you can run fully local (Ollama) or on a hosted API (Gemini) with no code changes.
- **Idempotent, resumable pipeline** — safe to re-run; stages skip completed work.

---

## License

[MIT](LICENSE)
