"""
cli.py — terminal Q&A for the NCERT GraphRAG.

Run (after artifacts are imported into Neo4j + Qdrant and the LLM is configured):
    python cli.py

Type a question and press Enter — it runs graph-based local search
(seed entities -> graph neighborhood -> source chunks -> grounded answer).
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
from src.retrieval.local_search import LocalSearch
from src.storage.neo4j_client import Neo4jClient
from src.storage.qdrant_client import QdrantClientWrapper

THEME = Theme({
    "accent": "#d98a5c",        # warm terracotta — the highlight colour
    "muted": "grey50",
    "ok": "#7cae7a",
    "err": "#d96c6c",
})
console = Console(theme=THEME)

_LOGO_RAW = r"""
 _   _  ___ ___ ___ _____    ___              _    ___  ___   ___
| \ | |/ __| __| _ \_   _|  / __|_ _ __ _ _ __| |_ | _ \/ _ \ / __|
|  \| | (__| _||   / | |   | (_ | '_/ _` | '_ \ ' \|   / (_) | (_ |
|_|\_|\___|___|_|_\ |_|    \___|_| \__,_| .__/_||_|_|_\\___/ \___|
"""
LOGO_LINES = _LOGO_RAW.strip("\n").split("\n")
_LOGO_W = max(len(line) for line in LOGO_LINES)

# Warm gradient stops (terracotta → gold → terracotta) painted across the logo.
_STOPS = [(0xD9, 0x8A, 0x5C), (0xF2, 0xD0, 0x90), (0xD9, 0x8A, 0x5C)]

_HINT = Text.from_markup(
    "[muted]Ask anything from NCERT grades 6–12 · "
    "grounded in the textbooks with citations · "
    "type[/] quit [muted]to exit[/]"
)


def _grad(t: float) -> tuple[int, int, int]:
    t = min(max(t, 0.0), 1.0)
    seg = t * (len(_STOPS) - 1)
    i = min(int(seg), len(_STOPS) - 2)
    f = seg - i
    a, b = _STOPS[i], _STOPS[i + 1]
    return tuple(round(a[j] + (b[j] - a[j]) * f) for j in range(3))


def render_logo(sweep: float | None) -> Text:
    """Logo as a horizontal gradient; `sweep` adds a moving white glow column."""
    t = Text()
    for li, line in enumerate(LOGO_LINES):
        if li:
            t.append("\n")
        for x, ch in enumerate(line):
            if ch == " ":
                t.append(" ")
                continue
            rgb = _grad(x / (_LOGO_W - 1))
            if sweep is not None and abs(x - sweep) < 7:
                glow = (1 - abs(x - sweep) / 7) ** 2
                rgb = tuple(round(rgb[j] + (255 - rgb[j]) * 0.85 * glow) for j in range(3))
            t.append(ch, style="#%02x%02x%02x" % rgb)
    return t


def banner_panel(logo: Text) -> Panel:
    return Panel(Group(logo, Text(""), _HINT),
                 box=ROUNDED, border_style="accent", padding=(1, 3))


def answer_panel(body, footer: str | None = None) -> Panel:
    return Panel(
        body,
        title="[accent]◆ ANSWER[/]",
        title_align="left",
        subtitle=footer,
        subtitle_align="right",
        box=ROUNDED, border_style="accent", padding=(1, 2),
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


async def ask(question: str, local: LocalSearch, llm: LLMClient) -> None:
    # 1. Retrieval (everything except the LLM call) under a spinner
    with console.status("[accent]searching the knowledge graph…", spinner="dots"):
        prepared = await local.prepare(question)

    # 2. Empty / short-circuit result → just print it
    if "prompt" not in prepared:
        console.print(answer_panel(Markdown(prepared["answer"])))
        return

    # 3. Stream the answer token-by-token into a live-updating panel
    system, user = prepared["prompt"]
    acc = ""
    console.print()
    with Live(answer_panel(Text("▍", style="accent")), console=console,
              refresh_per_second=16, vertical_overflow="visible") as live:
        async for delta in llm.stream_complete(system, user):
            acc += delta
            live.update(answer_panel(Markdown(acc + " ▍")))
        live.update(answer_panel(Markdown(acc or "_(no answer)_"),
                                 footer=f"{settings.online_model}"))

    # 4. Sources
    if prepared.get("sources"):
        console.print(sources_table(prepared["sources"]))


async def main() -> None:
    neo4j, qdrant = Neo4jClient(), QdrantClientWrapper()
    embedder, llm = EmbeddingEngine(), LLMClient()

    # Shimmer the logo while the backend connects + bge-m3 loads, then settle.
    async def _connect() -> None:
        await neo4j.connect()
        await qdrant.connect()
        await asyncio.to_thread(embedder.load)   # blocking model load off the event loop

    loader = asyncio.create_task(_connect())
    frame = 0
    with Live(banner_panel(render_logo(0)), console=console, refresh_per_second=30) as live:
        while not loader.done():
            live.update(banner_panel(render_logo((frame % (_LOGO_W + 16)) - 8)))
            await asyncio.sleep(0.03)
            frame += 1
        live.update(banner_panel(render_logo(None)))  # final static gradient
    try:
        await loader
    except Exception as exc:
        console.print(f"[err]✗ startup failed:[/] {exc}")
        console.print("[muted]Is docker compose up? (Neo4j :7687, Qdrant :6333)[/]")
        return

    local = LocalSearch(neo4j, qdrant, embedder, llm)
    console.print("[ok]●[/] [muted]ready[/]\n")

    try:
        while True:
            question = await asyncio.to_thread(console.input, "[accent]❯[/] ")
            if not question.strip():
                continue
            if question.strip().lower() in {"quit", "exit"}:
                break
            try:
                await ask(question, local, llm)
            except Exception as exc:  # one bad query shouldn't kill the session
                console.print(f"[err]✗ search failed:[/] {exc}")
    finally:
        await neo4j.close()
        await qdrant.close()
        console.print("\n[muted]bye[/]")


if __name__ == "__main__":
    asyncio.run(main())
