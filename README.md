# NCERT GraphRAG

A **graph-based Retrieval-Augmented Generation** system over NCERT textbooks (grades 6–12).

Standard RAG retrieves text chunks by vector similarity and often misses anything the
question doesn't lexically resemble. This project adds a **knowledge-graph layer**: it
finds the entities a question is about, walks their relationships in the graph to pull in
related concepts, and only then retrieves the source passages — so answers are grounded in
the textbooks, with citations, and benefit from multi-hop context.

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

| | Standard RAG | This project |
|---|---|---|
| Retrieval | top-k chunks by vector similarity | entities → **graph neighborhood** → chunks |
| Multi-hop reasoning | ✗ | ✓ (traverses relationships) |
| Grounding | chunk text | chunk text **+ knowledge graph** |
| Citations | sometimes | every answer cites source PDF + chapter |

**Query flow:** embed the query → find seed entities in Qdrant → expand their neighborhood
in Neo4j → gather and rank the linked source chunks → assemble a token-budgeted context →
generate a cited answer, streamed to the terminal.

---

## Architecture

The system splits into a heavy **offline ingestion** stage (GPU) and a lightweight
**online serving** stage (local).

### Offline ingestion — `run_pipeline.py`
1. **Parse** NCERT PDFs (PyMuPDF) → text + chapter detection.
2. **Chunk** hierarchically — L1 section / L2 paragraph / L3 atomic — with semantic
   boundaries (`src/ingestion/chunker.py`).
3. **Embed** each chunk with **BAAI/bge-m3** (dense + sparse hybrid vectors).
4. **Extract** entities & relationships per chunk with an LLM (**Qwen2.5-7B**), with
   JSON parsing that salvages output truncated at the token limit
   (`src/ingestion/llm_output.py`).
5. **Build** a deduplicated knowledge graph (`src/ingestion/graph_builder.py`).

The pipeline is **resumable and idempotent** — each stage skips work whose output already
exists, and extraction checkpoints to a JSONL it resumes from.

### Online serving — `cli.py`
- `src/retrieval/local_search.py` — entity → graph → chunks retrieval.
- `src/storage/` — async Neo4j (graph) and Qdrant (hybrid vector) clients.
- `src/llm/client.py` — any OpenAI-compatible LLM (Gemini, Ollama, Groq, OpenAI).
- A `rich` terminal UI that **streams** answers token-by-token with source citations.

---

## Ingestion on Lightning AI

Entity extraction over the full corpus (~6,500 chunks with a 7B model) needs a GPU, so
ingestion runs on a cloud GPU ([Lightning AI](https://lightning.ai), an A100 Studio) and
produces a folder of **artifacts** that you download and import locally. This two-stage
design keeps the serving side light — no GPU needed to answer questions.

```
┌─ GPU Studio (Lightning AI) ─┐         ┌─ Your machine ──────────────┐
│  download_ncert.py          │         │  docker compose up -d       │
│  run_pipeline.py            │  copy   │  import_artifacts.py        │
│   └─ artifacts/             │ ──────► │   └─ Neo4j + Qdrant         │
│       chunks.parquet        │ folder  │  cli.py  (ask questions)    │
│       entities.json         │         │                             │
│       relationships.json    │         │                             │
└─────────────────────────────┘         └─────────────────────────────┘
```

```bash
# on the GPU machine
pip install -r requirements-lightning.txt
python download_ncert.py
python run_pipeline.py --batch 32 --max-new-tokens 768     # -> ./artifacts/

# locally: drop artifacts/ into data/artifacts/, then
python -m src.ingestion.import_artifacts
```

A few engineering problems this stage solves:

- **Extraction throughput.** A naive HuggingFace pipeline runs prompts one at a time
  (~0.05 chunk/s — over a day for the corpus). Configuring true batched generation
  (left-padding + a real batch size in `src/ingestion/extractor.py`) takes it to
  ~7 chunk/s on an A100.
- **Truncated JSON.** When a chunk is information-dense the model can hit the token limit
  mid-JSON. `parse_json_block` salvages the complete entities/relationships instead of
  dropping the whole chunk.
- **Resumability.** Extraction appends to a checkpoint JSONL; a disconnect or GPU-time
  limit never loses completed work — re-running continues where it stopped.
- **Dependency pinning.** `transformers` is pinned (FlagEmbedding needs a symbol newer
  releases removed), so the GPU image builds reproducibly.

---

## Tech stack

**Python** · **Neo4j** (knowledge graph) · **Qdrant** (hybrid dense+sparse vectors) ·
**bge-m3** (embeddings) · **Qwen2.5-7B** (offline extraction) · **Gemini / Ollama**
(answering) · **PyMuPDF** · `async` I/O throughout · `rich` (CLI). No LangChain — every
stage is built from primitives.

---

## Quick start

### Prerequisites
- Python 3.10+
- Docker (for Neo4j + Qdrant)
- An OpenAI-compatible LLM: a free [Gemini key](https://aistudio.google.com/apikey), or
  local [Ollama](https://ollama.com).

```bash
# 1. install & configure
pip install -r requirements.txt
cp .env.example .env          # edit .env with your LLM provider + key

# 2. start the databases
make up                        # Neo4j :7687, Qdrant :6333

# 3. import data (see "Ingestion on Lightning AI" to generate the artifacts)
python -m src.ingestion.import_artifacts

# 4. ask questions
python cli.py
```
```
❯ explain osmosis
❯ derive sin²θ + cos²θ = 1
❯ what is Newton's second law?
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
│   ├── models/                  # Pydantic schemas (chunks, entities, relationships)
│   ├── ingestion/
│   │   ├── pdf_parser.py         # PDF text + chapter detection
│   │   ├── chunker.py            # hierarchical semantic chunking (L1/L2/L3)
│   │   ├── embedder.py           # bge-m3 dense + sparse embeddings
│   │   ├── extractor.py          # LLM entity/relationship extraction (GPU-batched)
│   │   ├── llm_output.py         # LLM-output parsing (salvages truncated JSON)
│   │   ├── graph_builder.py      # entity dedup + relationship build
│   │   └── import_artifacts.py   # load artifacts → Neo4j + Qdrant
│   ├── retrieval/
│   │   ├── local_search.py       # entity → graph → chunks
│   │   └── context_builder.py    # dedup, order, token-budget assembly
│   ├── storage/
│   │   ├── neo4j_client.py        # async graph queries + batched upserts
│   │   └── qdrant_client.py       # async hybrid vector search
│   └── llm/
│       ├── client.py              # OpenAI-compatible async client (stream + complete)
│       └── prompts.py
└── requirements*.txt
```

---

## Notable engineering details

- **No N+1 queries** — a single Cypher query expands all seed entities' neighborhoods in
  one round-trip (`get_neighbors_batch`), returning chunk ids inline so retrieval needs no
  per-entity follow-up.
- **GPU-batched extraction** — true batched generation (left-padding + batch size) turns
  ~0.05 → ~7 chunks/sec on an A100.
- **Truncation-safe parsing** — `parse_json_block` recovers complete records from JSON the
  model cut off at the token limit.
- **Provider-agnostic LLM** — the OpenAI SDK points at any compatible endpoint via env
  vars, so you can run fully local (Ollama) or hosted (Gemini) with no code changes.
- **Idempotent, resumable pipeline** — safe to re-run; stages skip completed work.

---

## Evaluation

Retrieval and generation are evaluated separately (a RAG can fail at either), with an
**ablation** that isolates what the knowledge-graph layer adds over plain vector search.
Run it with `python eval.py` (retrieval metrics, free) or `python eval.py --judge`
(adds LLM-graded faithfulness/relevance/correctness).

- **Retrieval** — Recall@k and MRR against a hand-written gold set, matching on chunk
  content (`eval/gold_set.json`, 50 questions across all 8 subjects).
- **Ablation** — the same metrics **with vs. without** the Neo4j graph-expansion step.

**Concept-lookup questions (n=50):** dense retrieval alone is already near-perfect, so the
graph adds little headroom — but after fixing retrieval to *union* graph-neighborhood
chunks (rather than filter by them) it edges ahead instead of trailing:

| Metric | Vector only | GraphRAG |
|---|---:|---:|
| Recall@1 | 92.0% | **94.0%** |
| MRR | 0.960 | **0.963** |

**Hard multi-hop questions (n=8)** — where the linked concepts sit in *different chapters
and aren't lexically similar to each other* (`eval/gold_multihop_hard.json`) — is where the
graph earns its keep:

| Metric | Vector only | GraphRAG | Δ |
|---|---:|---:|---:|
| Hop coverage | 93.8% | **100%** | +6.2 pp |
| Full coverage | 87.5% | **100%** | **+12.5 pp** |

For example, on *"trace the energy from sunlight in photosynthesis to a muscle
contraction,"* plain vector search retrieves the `photosynthesis` and `muscle` passages
(both named in the query) but misses the intermediate `glucose` and `respiration` hops —
they aren't lexically similar to the query. Graph expansion follows the relationship edges
to those connected passages and achieves full coverage. This is the canonical GraphRAG
win: **retrieving a relevant passage the query isn't lexically similar to, because a graph
edge bridges to it.** The benefit appears only on questions that genuinely require bridging
distant concepts — single-concept lookups are already saturated for vector search.

---

## License

[MIT](LICENSE)
