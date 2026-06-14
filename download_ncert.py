"""
NCERT PDF Downloader
====================
Downloads all NCERT textbooks (grades 6-12) from ncert.nic.in
Run: python download_ncert.py

PDFs are saved to data/raw/<subject>/grade_<N>/<book_code>_ch<N>.pdf
"""

import time
from pathlib import Path

import requests

# ── NCERT book codes → (subject, grade, num_chapters) ─────────────────────────
# URL pattern: https://ncert.nic.in/textbook/pdf/{code}{chapter:02d}.pdf
# Codes: first letter = grade (f=6,g=7,h=8,i=9,j=10,k=11,l=12)
#        remaining = subject abbreviation

BOOKS = [
    # Grade 6
    ("fesc1",  "science",        6,  16),
    ("femh1",  "mathematics",    6,  14),
    ("fehs1",  "history",        6,  12),
    ("feeg1",  "geography",      6,   8),
    ("fess1",  "polity",         6,   9),

    # Grade 7
    ("gesc1",  "science",        7,  18),
    ("gemh1",  "mathematics",    7,  15),
    ("gehs1",  "history",        7,  10),
    ("geeg1",  "geography",      7,   9),
    ("gess1",  "polity",         7,   9),

    # Grade 8
    ("hesc1",  "science",        8,  18),
    ("hemh1",  "mathematics",    8,  16),
    ("hehs1",  "history",        8,  12),
    ("heeg1",  "geography",      8,   6),
    ("hess1",  "polity",         8,  10),

    # Grade 9
    ("iesc1",  "science",        9,  15),
    ("iemh1",  "mathematics",    9,   8),
    ("iemh2",  "mathematics",    9,   7),
    ("iehs1",  "history",        9,   5),
    ("ieeg1",  "geography",      9,   6),
    ("iess1",  "polity",         9,   6),
    ("ieec1",  "economics",      9,   5),

    # Grade 10
    ("jesc1",  "science",       10,  16),
    ("jemh1",  "mathematics",   10,   8),
    ("jemh2",  "mathematics",   10,   7),
    ("jehs1",  "history",       10,   5),
    ("jeeg1",  "geography",     10,   7),
    ("jess1",  "polity",        10,   8),
    ("jeec1",  "economics",     10,   5),

    # Grade 11
    ("keph1",  "physics",       11,   8),
    ("keph2",  "physics",       11,   7),
    ("kech1",  "chemistry",     11,   7),
    ("kech2",  "chemistry",     11,   7),
    ("kebo1",  "biology",       11,  22),
    ("kemh1",  "mathematics",   11,   9),
    ("kemh2",  "mathematics",   11,   8),
    ("kehs1",  "history",       11,  11),
    ("keeg1",  "geography",     11,  12),
    ("keeg2",  "geography",     11,   8),
    ("keps1",  "polity",        11,  10),
    ("keps2",  "polity",        11,   9),
    ("keec1",  "economics",     11,   6),
    ("keec2",  "economics",     11,   6),

    # Grade 12
    ("leph1",  "physics",       12,   8),
    ("leph2",  "physics",       12,   6),
    ("lech1",  "chemistry",     12,   8),
    ("lech2",  "chemistry",     12,   8),
    ("lebo1",  "biology",       12,  16),
    ("lemh1",  "mathematics",   12,   7),
    ("lemh2",  "mathematics",   12,   6),
    ("lehs1",  "history",       12,  15),
    ("leeg1",  "geography",     12,  12),
    ("leeg2",  "geography",     12,   6),
    ("leps1",  "polity",        12,   9),
    ("leps2",  "polity",        12,   9),
    ("leec1",  "economics",     12,   6),
    ("leec2",  "economics",     12,   6),
]

BASE_URL = "https://ncert.nic.in/textbook/pdf/{code}{ch:02d}.pdf"
OUT_ROOT = Path("data/raw")
DELAY    = 1.0   # seconds between requests — be polite to the server


def is_pdf(content: bytes) -> bool:
    return content[:4] == b"%PDF"


def download_all() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (educational use)"

    total_ok = total_skip = total_fail = 0

    for code, subject, grade, max_chapters in BOOKS:
        dest_dir = OUT_ROOT / subject / f"grade_{grade}"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for ch in range(1, max_chapters + 1):
            url  = BASE_URL.format(code=code, ch=ch)
            dest = dest_dir / f"{code}_ch{ch:02d}.pdf"

            if dest.exists():
                print(f"  skip  {dest.name} (already downloaded)")
                total_skip += 1
                continue

            try:
                resp = session.get(url, timeout=30)
                if resp.status_code == 200 and is_pdf(resp.content):
                    dest.write_bytes(resp.content)
                    size_kb = len(resp.content) // 1024
                    print(f"  ok    {dest.name}  ({size_kb} KB)")
                    total_ok += 1
                else:
                    print(f"  miss  {url}  [{resp.status_code}]")
                    total_fail += 1
            except Exception as exc:
                print(f"  err   {url}  {exc}")
                total_fail += 1

            time.sleep(DELAY)

    print(f"\nDone — downloaded: {total_ok}  skipped: {total_skip}  failed: {total_fail}")
    print(f"PDFs saved to: {OUT_ROOT.resolve()}")


if __name__ == "__main__":
    download_all()
