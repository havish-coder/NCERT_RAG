"""
GraphBuilder
============
Takes raw entity/relationship dicts from the extractor and:
1. Normalises entity names (lowercase, strip) → dedup key
2. Merges duplicate entities (same name across different chunks/grades)
3. Outputs clean entities.json + relationships.json for local import

Kept simple: exact-match dedup on normalised name.
NCERT deliberately repeats concepts across grades at increasing depth
(e.g., "photosynthesis" appears in grades 6, 7, 11) — merging them
gives entities richer descriptions and broader grade coverage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4


def build_graph(
    extraction_results: list[dict],  # from extractor.extract_batch()
    chunk_metadata: dict[str, dict], # chunk_id -> {subject, grade, chapter}
    output_dir: Path,
) -> tuple[list[dict], list[dict]]:
    """
    Process all extraction results into deduplicated entity and relationship lists.
    Saves entities.json and relationships.json to output_dir.
    Returns (entities, relationships).
    """
    entity_map: dict[str, dict] = {}   # normalised_name -> entity dict
    relationships: list[dict] = []

    for result in extraction_results:
        chunk_id = result["chunk_id"]
        meta = chunk_metadata.get(chunk_id, {})
        subject = meta.get("subject", "unknown")
        grade = meta.get("grade", 0)

        # ── Entities ──────────────────────────────────────────────────────────
        local_id_map: dict[str, str] = {}  # raw name -> canonical entity_id

        for raw in result.get("entities", []):
            name = raw.get("name", "").strip()
            if not name:
                continue

            norm = _normalise(name)
            if norm in entity_map:
                e = entity_map[norm]
                if chunk_id not in e["chunk_ids"]:
                    e["chunk_ids"].append(chunk_id)
                if subject not in e["subjects"]:
                    e["subjects"].append(subject)
                if grade and grade not in e["grades"]:
                    e["grades"].append(grade)
                # Extend description with new context if different
                if raw.get("description") and raw["description"] not in e["description"]:
                    e["description"] = e["description"] + " | " + raw["description"]
            else:
                entity_map[norm] = {
                    "entity_id": str(uuid4()),
                    "name": name,
                    "entity_type": raw.get("entity_type", "CONCEPT"),
                    "description": raw.get("description", ""),
                    "subjects": [subject] if subject else [],
                    "grades": [grade] if grade else [],
                    "chunk_ids": [chunk_id],
                }
            local_id_map[name.lower()] = entity_map[norm]["entity_id"]

        # ── Relationships ──────────────────────────────────────────────────────
        for raw in result.get("relationships", []):
            src_name = raw.get("source", "").lower()
            tgt_name = raw.get("target", "").lower()
            src_id = local_id_map.get(src_name)
            tgt_id = local_id_map.get(tgt_name)
            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            relationships.append({
                "rel_id": str(uuid4()),
                "source_entity_id": src_id,
                "target_entity_id": tgt_id,
                "relation_type": raw.get("relation_type", "RELATES_TO"),
                "description": raw.get("description", ""),
                "weight": 1.0,
                "chunk_ids": [chunk_id],
            })

    entities = list(entity_map.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "entities.json").write_text(json.dumps(entities, indent=2))
    (output_dir / "relationships.json").write_text(json.dumps(relationships, indent=2))

    return entities, relationships


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", name.lower().strip())
