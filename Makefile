.PHONY: up down ingest import cli test lint format

# Start Neo4j + Qdrant locally.
up:
	docker compose up -d

down:
	docker compose down

# Offline ingestion (run on a GPU machine). Produces artifacts in ./artifacts/.
# e.g. make ingest ARGS="--batch 32 --max-new-tokens 768"
ingest:
	python run_pipeline.py $(ARGS)

# Load downloaded artifacts (data/artifacts/) into local Neo4j + Qdrant.
import:
	python -m src.ingestion.import_artifacts

# Interactive terminal Q&A.
cli:
	python cli.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
