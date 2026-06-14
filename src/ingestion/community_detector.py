"""
CommunityDetector
=================
Builds an in-memory igraph from entities + relationships,
runs the Leiden algorithm at two resolutions to get a 2-level hierarchy:
  Level 0 (fine)   — res=1.5: small, cohesive topic clusters
  Level 1 (coarse) — res=0.5: broader theme groups

Outputs community_memberships.json with each entity's community assignment.

Runs locally on CPU after artifacts are downloaded from Kaggle.
No GPU needed — igraph + leidenalg handle graphs with 10K–100K nodes easily.

Why Leiden over Louvain?
Leiden guarantees internally connected communities. Louvain can produce
disconnected subsets that look like communities but aren't — this makes LLM
summaries incoherent. See: Traag et al. (2019).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import igraph as ig
import leidenalg


def detect_communities(
    entities_path: Path,
    relationships_path: Path,
    output_path: Path,
    resolutions: tuple[float, float] = (1.5, 0.5),
) -> dict[str, dict]:
    """
    Loads entity/relationship JSON, builds igraph, runs Leiden at two resolutions.
    Returns and saves community_memberships.json:
    {
      entity_id: {
        "community_0": "c-<uuid-prefix>",  # fine
        "community_1": "c-<uuid-prefix>"   # coarse
      }
    }
    """
    entities = json.loads(entities_path.read_text())
    rels = json.loads(relationships_path.read_text())

    if not entities:
        return {}

    # Build igraph
    id_to_idx = {e["entity_id"]: i for i, e in enumerate(entities)}
    graph = ig.Graph(n=len(entities), directed=False)
    graph.vs["entity_id"] = [e["entity_id"] for e in entities]

    edges, weights = [], []
    for r in rels:
        src = id_to_idx.get(r["source_entity_id"])
        tgt = id_to_idx.get(r["target_entity_id"])
        if src is not None and tgt is not None and src != tgt:
            edges.append((src, tgt))
            weights.append(float(r.get("weight", 1.0)))

    graph.add_edges(edges)
    graph.es["weight"] = weights

    # Leiden at fine then coarse resolution
    fine_partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolutions[0],
        n_iterations=10,
        seed=42,
    )
    coarse_partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolutions[1],
        n_iterations=10,
        seed=42,
    )

    # Map fine → coarse by majority: whichever coarse community contains the most
    # members of a fine community is its parent
    fine_to_coarse = _map_fine_to_coarse(fine_partition.membership, coarse_partition.membership)

    # Build community IDs
    fine_ids = {c: f"c0-{c}" for c in set(fine_partition.membership)}
    coarse_ids = {c: f"c1-{c}" for c in set(coarse_partition.membership)}

    memberships: dict[str, dict] = {}
    for i, entity in enumerate(entities):
        fc = fine_partition.membership[i]
        cc = coarse_partition.membership[i]
        memberships[entity["entity_id"]] = {
            "community_0": fine_ids[fc],
            "community_1": coarse_ids[cc],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(memberships, indent=2))

    # Also save community member lists (needed by summarizer)
    community_members: dict[str, list[str]] = {}
    for eid, comms in memberships.items():
        for level_key, comm_id in comms.items():
            community_members.setdefault(comm_id, []).append(eid)

    members_path = output_path.parent / "community_members.json"
    members_path.write_text(json.dumps(community_members, indent=2))

    return memberships


def _map_fine_to_coarse(fine: list[int], coarse: list[int]) -> dict[int, int]:
    fine_to_coarse: dict[int, Counter] = {}
    for f, c in zip(fine, coarse):
        fine_to_coarse.setdefault(f, Counter())[c] += 1
    return {f: counter.most_common(1)[0][0] for f, counter in fine_to_coarse.items()}
