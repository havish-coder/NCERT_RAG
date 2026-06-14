from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
import structlog

logger = structlog.get_logger(__name__)

_CHAPTER_PATTERNS = [
    re.compile(r"^chapter\s+(\d+)\s*[:\-–]?\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(\d+)\.\s+([A-Z][A-Z\s,\-]+)$", re.MULTILINE),
]
_FOOTER_PATTERNS = [
    re.compile(r"^\d+\s*$", re.MULTILINE),
    re.compile(r"ncert|www\.ncert\.nic\.in", re.IGNORECASE),
    re.compile(r"^[A-Z\s]+\|\s*\d+$", re.MULTILINE),
]


class PDFParser:
    """Extracts page-level text blocks from NCERT PDFs with chapter metadata."""

    def __init__(self, subject: str, grade: int) -> None:
        self.subject = subject
        self.grade = grade

    def parse(self, pdf_path: Path) -> list[dict]:
        """
        Returns a list of page dicts:
        {page_num, text, chapter, chapter_title, section, source_pdf}
        """
        doc = fitz.open(str(pdf_path))
        pages: list[dict] = []
        current_chapter = 1
        current_chapter_title = "Introduction"
        current_section: str | None = None

        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            text = self._clean_text(raw_text)
            if not text.strip():
                continue

            new_ch, new_title = self._detect_chapter_boundary(text)
            if new_ch is not None:
                current_chapter = new_ch
                current_chapter_title = new_title
                current_section = None

            pages.append({
                "page_num": page_num,
                "text": text,
                "chapter": current_chapter,
                "chapter_title": current_chapter_title,
                "section": current_section,
                "source_pdf": pdf_path.name,
            })

        doc.close()
        logger.info(
            "pdf_parsed",
            path=pdf_path.name,
            subject=self.subject,
            grade=self.grade,
            pages=len(pages),
        )
        return pages

    def _detect_chapter_boundary(self, text: str) -> tuple[int | None, str]:
        for pattern in _CHAPTER_PATTERNS:
            m = pattern.search(text[:500])
            if m:
                try:
                    ch_num = int(m.group(1))
                    ch_title = m.group(2).strip().title()
                    return ch_num, ch_title
                except (IndexError, ValueError):
                    pass
        return None, ""

    def _clean_text(self, raw: str) -> str:
        lines = raw.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if any(p.search(stripped) for p in _FOOTER_PATTERNS):
                continue
            cleaned.append(stripped)
        # Preserve line breaks: chapter/section detectors use ^ (MULTILINE) to
        # anchor headings to the start of a line. Joining with spaces would
        # collapse every page into one line and break heading detection.
        return "\n".join(cleaned)
