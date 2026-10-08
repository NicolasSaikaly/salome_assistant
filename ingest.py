"""Step 2a - Turn the downloaded SALOME HTML docs into text chunks.

Reads every HTML page under data/raw/, keeps only the main content,
splits it by section (h1/h2/h3), then cuts long sections into chunks
of at most MAX_CHARS characters. Code examples are kept intact
as ```python blocks so the assistant can quote them.

Output: data/chunks.jsonl (one JSON object per chunk).
"""

import json
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup

# Where the HTML docs are. Two options:
#  - default: the pages downloaded with wget into data/raw/ (path = URL)
#  - a local SALOME install:  DOCS_DIR=~/salome/.../doc/salome/gui/SMESH python ingest.py
#    (links in answers still point to the online docs, via BASE_URL)
DOCS_DIR = os.environ.get("DOCS_DIR")
RAW_DIR = Path(os.path.expanduser(DOCS_DIR)) if DOCS_DIR else Path("data/raw")
BASE_URL = os.environ.get("BASE_URL", "https://docs.salome-platform.org/latest/gui/SMESH/")
OUT_FILE = Path("data/chunks.jsonl")
MAX_CHARS = 1500      # max size of a chunk
MIN_CHARS = 80        # drop near-empty chunks
HEADING_MARK = "§§H§§"   # temporary marker for headings
CODE_MARK = "§§C§§"      # temporary marker for code examples
# Elements that start a new line (everything else, like links, stays inline)
BLOCK_TAGS = ["p", "li", "dt", "dd", "tr", "table", "ul", "ol", "dl",
              "div", "section", "blockquote", "br", "figure", "caption"]

# Sphinx pages that are not real content
SKIP_NAMES = {"genindex.html", "search.html", "py-modindex.html"}
SKIP_DIRS = {"_sources", "_static", "_images", "_modules"}


def page_url(path: Path) -> str:
    """Online URL of a local page.

    wget copy:      data/raw/docs.salome-platform.org/a/b.html -> https://docs.salome-platform.org/a/b.html
    SALOME install: <DOCS_DIR>/b.html                         -> BASE_URL + b.html
    """
    rel = path.relative_to(RAW_DIR).as_posix()
    return BASE_URL + rel if DOCS_DIR else "https://" + rel


def extract_sections(html: str):
    """Return (page_title, [(section_title, anchor, text), ...])."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else ""
    title = title.split("—")[0].strip()  # "Page — SALOME docs" -> "Page"

    main = (soup.select_one('[role="main"]')
            or soup.select_one("div.body")
            or soup.body)
    if main is None:
        return title, []

    # Remove noise: scripts, nav, the "¶" permalinks
    for tag in main.select("script, style, nav, a.headerlink"):
        tag.decompose()

    # 1. Set code examples aside (their spaces and line breaks matter)
    codes = []
    for pre in main.find_all("pre"):
        codes.append(pre.get_text().rstrip())
        pre.replace_with(f"{CODE_MARK}{len(codes) - 1}{CODE_MARK}")

    # 2. In normal text, line breaks in the HTML source mean nothing:
    #    "how a\n<a>Wire Discretization</a>\nshould" -> "how a Wire Discretization should"
    for s in list(main.find_all(string=True)):
        s.replace_with(re.sub(r"\s+", " ", s))

    # 3. Real paragraph breaks only around block elements
    for tag in main.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")

    # 4. Put a marker before each heading so we can split on it later
    for h in main.find_all(["h1", "h2", "h3"]):
        parent = h.find_parent(["section", "div"])
        anchor = h.get("id") or (parent.get("id") if parent else "") or ""
        h.replace_with(f"\n{HEADING_MARK}{anchor}|{h.get_text(' ', strip=True)}\n")

    text = main.get_text()
    # 5. Put the code examples back as ```python blocks
    text = re.sub(f"{CODE_MARK}(\\d+){CODE_MARK}",
                  lambda m: f"\n```python\n{codes[int(m.group(1))]}\n```\n", text)
    sections = []
    for part in text.split(HEADING_MARK):
        first_line, _, body = part.partition("\n")
        if "|" in first_line:
            anchor, sec_title = first_line.split("|", 1)
        else:  # text before the first heading
            anchor, sec_title, body = "", title, part
        body = clean(body)
        if body:
            sections.append((sec_title.strip(), anchor.strip(), body))
    return title, sections


def clean(text: str) -> str:
    """Collapse blank lines and stray spaces, without touching code blocks."""
    out, in_code = [], False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line.strip())
        elif in_code:
            out.append(line)
        else:
            line = re.sub(r"\s+", " ", line).strip()
            if line or (out and out[-1] != ""):
                out.append(line)
    return "\n".join(out).strip()


def split_long(text: str):
    """Cut a long section into chunks <= MAX_CHARS, on paragraph boundaries.

    A code block is never cut in the middle (it may exceed MAX_CHARS alone).
    """
    blocks, current, in_code = [], [], False
    for line in text.split("\n"):
        current.append(line)
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and line == "":
            blocks.append("\n".join(current).strip())
            current = []
    if current:
        blocks.append("\n".join(current).strip())

    chunks, buf = [], ""
    for block in filter(None, blocks):
        if buf and len(buf) + len(block) > MAX_CHARS:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n\n{block}" if buf else block
    if buf:
        chunks.append(buf)
    return chunks


def main():
    pages = [p for p in RAW_DIR.rglob("*.html")
             if p.name not in SKIP_NAMES and not SKIP_DIRS & set(p.parts)]
    if not pages:
        raise SystemExit(f"No HTML page found in {RAW_DIR} - check DOCS_DIR")
    n_chunks = 0
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for path in sorted(pages):
            title, sections = extract_sections(path.read_text(encoding="utf-8", errors="ignore"))
            url = page_url(path)
            for sec_title, anchor, body in sections:
                for i, chunk in enumerate(split_long(body)):
                    if len(chunk) < MIN_CHARS:
                        continue
                    record = {
                        "id": f"{path.relative_to(RAW_DIR).as_posix()}#{anchor}#{i}",
                        "title": title,
                        "section": sec_title,
                        "url": f"{url}#{anchor}" if anchor else url,
                        # The header gives the model context about where the text comes from
                        "text": f"{title} > {sec_title}\n\n{chunk}",
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_chunks += 1
    print(f"{len(pages)} pages -> {n_chunks} chunks written to {OUT_FILE}")


if __name__ == "__main__":
    main()
