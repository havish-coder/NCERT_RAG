"""
cli.py — terminal Q&A for the NCERT GraphRAG.

Run (after artifacts are imported into Neo4j + Qdrant and the LLM is configured):
    python cli.py

Type a question and press Enter. The mode auto-routes:
  - broad/thematic questions   -> global search (community summaries)
  - specific concept questions -> local search (entity graph + chunks)
Force a mode by prefixing:  "local: ..."  or  "global: ..."
Type 'quit' to exit.
"""
from __future__ import annotations

import asyncio
import logging
import sys

# Force UTF-8 so the box-drawing / icon glyphs never hit a legacy-console
# (cp1252) UnicodeEncodeError on Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except Exception:
    pass

# Quiet the chatty library loggers so they don't print over the chat UI
# (Neo4j deprecation/notification warnings, httpx request lines, etc.).
for _noisy in ("neo4j", "neo4j.notifications", "httpx", "httpcore"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

from rich.box import ROUNDED
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from src.config import settings
from src.ingestion.embedder import EmbeddingEngine
from src.llm.client import LLMClient
from src.retrieval.global_search import GlobalSearch
from src.retrieval.local_search import LocalSearch
from src.storage.neo4j_client import Neo4jClient
from src.storage.qdrant_client import QdrantClientWrapper

THEME = Theme({
    "accent": "#d98a5c",        # warm terracotta — the highlight colour
    "accent2": "#5c9ad9",       # cool blue for the global mode
    "muted": "grey50",
    "ok": "#7cae7a",
    "err": "#d96c6c",
})
console = Console(theme=THEME)

GLOBAL_HINTS = ("compare", "overview", "major themes", "all topics",
                "across grades", "curriculum", "what subjects")

# Mode → (display label, colour, icon)
MODES = {
    "local":  ("LOCAL",  "accent",  "◆"),
    "global": ("GLOBAL", "accent2", "◇"),
}

LOGO = r"""[accent] _   _  ___ ___ ___ _____    ___              _    ___  ___   ___ [/]
[accent]| \ | |/ __| __| _ \_   _|  / __|_ _ __ _ _ __| |_ | _ \/ _ \ / __|[/]
[accent]|  \| | (__| _||   / | |   | (_ | '_/ _` | '_ \ ' \|   / (_) | (_ |[/]
[accent]|_|\_|\___|___|_|_\ |_|    \___|_| \__,_| .__/_||_|_|_\\___/ \___|[/]
[muted]                                        |_|  graph-rag over NCERT[/]"""


def route(question: str) -> tuple[str, str]:
    """Return (mode, cleaned_question) from an optional prefix or keywords."""
    q = question.strip()
    if q.lower().startswith("global:"):
        return "global", q[7:].strip()
    if q.lower().startswith("local:"):
        return "local", q[6:].strip()
    mode = "global" if any(h in q.lower() for h in GLOBAL_HINTS) else "local"
    return mode, q


def banner() -> Panel:
    hint = Text.from_markup(
        "[muted]Ask anything from NCERT grades 6–12. "
        "Prefix [/][accent]local:[/][muted] / [/][accent2]global:[/][muted] to force a mode · "
        "[/][muted]type[/] quit [muted]to exit[/]"
    )
    return Panel(
        Group(Text.from_markup(LOGO), Text(""), hint),
        box=ROUNDED, border_style="accent", padding=(1, 3),
    )


def answer_panel(mode: str, body, footer: str | None = None) -> Panel:
    label, colour, icon = MODES[mode]
    return Panel(
        body,
        title=f"[{colour}]{icon} {label}[/]",
        title_align="left",
        subtitle=footer,
        subtitle_align="right",
        box=ROUNDED, border_style=colour, padding=(1, 2),
    )


def sources_table(sources: list[dict]) -> Table:
    t = Table(box=ROUNDED, border_style="muted", header_style="muted",
              title="[muted]sources[/]", title_justify="left", expand=False)
    t.add_column("PDF", style="accent", no_wrap=True)
    t.add_column("Ch", justify="right", style="muted")
    t.add_column("Page", justify="right", style="muted")
    t.add_column("Subject")
    t.add_column("Gr", justify="right")
    for s in sources[:5]:
        t.add_row(str(s["source_pdf"]), str(s["chapter"]), str(s["page_start"]),
                  str(s["subject"]), str(s["grade"]))
    return t


async def ask(question: str, local: LocalSearch, glob: GlobalSearch, llm: LLMClient) -> None:
    mode, q = route(question)
    label, colour, icon = MODES[mode]

    # 1. Retrieval (everything except the LLM call) under a spinner
    with console.status(f"[{colour}]searching the {label.lower()} graph…", spinner="dots"):
        engine = glob if mode == "global" else local
        prepared = await engine.prepare(q)

    # 2. Empty / short-circuit result → just print it
    if "prompt" not in prepared:
        console.print(answer_panel(mode, Markdown(prepared["answer"])))
        return

    # 3. Stream the answer token-by-token into a live-updating panel
    system, user = prepared["prompt"]
    acc = ""
    console.print()
    with Live(answer_panel(mode, Text("▍", style=colour)), console=console,
              refresh_per_second=16, vertical_overflow="visible") as live:
        async for delta in llm.stream_complete(system, user):
            acc += delta
            live.update(answer_panel(mode, Markdown(acc + " ▍")))
        live.update(answer_panel(mode, Markdown(acc or "_(no answer)_"),
                                 footer=f"{settings.online_model}"))

    # 4. Sources (local only)
    if prepared.get("sources"):
        console.print(sources_table(prepared["sources"]))


async def main() -> None:
    neo4j, qdrant = Neo4jClient(), QdrantClientWrapper()
    embedder, llm = EmbeddingEngine(), LLMClient()

    console.print(banner())
    try:
        with console.status("[accent]connecting to Neo4j + Qdrant and loading bge-m3…", spinner="dots"):
            await neo4j.connect()
            await qdrant.connect()
            embedder.load()
    except Exception as exc:
        console.print(f"[err]✗ startup failed:[/] {exc}")
        console.print("[muted]Is docker compose up? (Neo4j :7687, Qdrant :6333)[/]")
        return

    local = LocalSearch(neo4j, qdrant, embedder, llm)
    glob = GlobalSearch(qdrant, embedder, llm)
    console.print("[ok]●[/] [muted]ready[/]\n")

    try:
        while True:
            question = await asyncio.to_thread(console.input, "[accent]❯[/] ")
            if not question.strip():
                continue
            if question.strip().lower() in {"quit", "exit"}:
                break
            try:
                await ask(question, local, glob, llm)
            except Exception as exc:  # one bad query shouldn't kill the session
                console.print(f"[err]✗ search failed:[/] {exc}")
    finally:
        await neo4j.close()
        await qdrant.close()
        console.print("\n[muted]bye[/]")


if __name__ == "__main__":
    asyncio.run(main())
