"""
tools_docs.py - let the agent answer from YOUR documents.

Drop PDFs, .txt and .md files into a folder called  docs/  next to this file
(annual reports, course notes, your resume, cheat sheets...). The agent finds
the most relevant passages and answers from them, quoting file and page.

Needs: pip install pypdf

How it works (no paid embeddings needed): each document is cut into small
chunks and ranked against your question with BM25, the keyword-scoring
formula classic search engines use. It matches WORDS, not meaning, so
questions that reuse the document's own terms work best.
"""
import math
import re
from collections import Counter
from pathlib import Path

from langchain_core.tools import tool

DOCS_DIR = Path(__file__).parent / "docs"
CHUNK_CHARS = 900
OVERLAP_CHARS = 150
SUPPORTED = {".pdf", ".txt", ".md"}

_cache = {"signature": None, "chunks": [], "tokens": [], "doc_freq": Counter(), "avg_len": 0.0}


def _tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def _split(text: str) -> list:
    """Cut text into overlapping chunks, preferring paragraph boundaries."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            if cut == -1:
                cut = text.rfind(". ", start + CHUNK_CHARS // 2, end)
            end = cut + 1 if cut != -1 else end
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


def _files() -> list:
    DOCS_DIR.mkdir(exist_ok=True)
    return [p for p in sorted(DOCS_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED]


def _pages(path: Path):
    """Yield (page_number, text) for one file."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield number, page.extract_text() or ""
    else:
        yield 1, path.read_text(encoding="utf-8", errors="replace")


def _build_index():
    """(Re)build the search index, but only when files were added or changed."""
    files = _files()
    signature = tuple((str(p), p.stat().st_mtime_ns) for p in files)
    if signature == _cache["signature"]:
        return
    chunks = []
    for path in files:
        try:
            for page_no, text in _pages(path):
                for piece in _split(text):
                    chunks.append({"source": path.name, "page": page_no, "text": piece})
        except Exception as e:  # one bad file should not break the rest
            chunks.append({"source": path.name, "page": 0, "text": f"[could not read file: {e}]"})
    tokens = [_tokenize(c["text"]) for c in chunks]
    doc_freq = Counter()
    for toks in tokens:
        doc_freq.update(set(toks))
    _cache.update(
        signature=signature,
        chunks=chunks,
        tokens=tokens,
        doc_freq=doc_freq,
        avg_len=(sum(len(t) for t in tokens) / len(tokens)) if tokens else 0.0,
    )


def _bm25(query_tokens: list, doc_tokens: list, k1: float = 1.5, b: float = 0.75) -> float:
    n_docs = len(_cache["chunks"])
    counts = Counter(doc_tokens)
    score = 0.0
    for term in set(query_tokens):
        if term not in counts:
            continue
        df = _cache["doc_freq"][term]
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = counts[term]
        norm = 1 - b + b * len(doc_tokens) / (_cache["avg_len"] or 1)
        score += idf * tf * (k1 + 1) / (tf + k1 * norm)
    return score


@tool
def search_documents(query: str) -> str:
    """Search the user's OWN documents (PDFs, notes, resume, reports in the docs
    folder) and return the most relevant passages with file name and page.
    Use this FIRST when the question mentions 'my notes', 'my resume', 'the
    report', 'the PDF', or a document the user may have added. Use keywords
    from the likely wording of the document."""
    _build_index()
    if not _cache["chunks"]:
        return "The docs folder is empty. Ask the user to add PDF, .txt or .md files to docs/."
    q = _tokenize(query)
    ranked = sorted(
        ((_bm25(q, toks), i) for i, toks in enumerate(_cache["tokens"])), reverse=True
    )
    top = [(s, i) for s, i in ranked[:4] if s > 0]
    if not top:
        return "No matching passages found in the user's documents. Try different keywords."
    return "\n\n".join(
        f"[{_cache['chunks'][i]['source']}, p.{_cache['chunks'][i]['page']}]\n{_cache['chunks'][i]['text']}"
        for _, i in top
    )


@tool
def list_documents() -> str:
    """List the documents available in the user's docs folder, with how many
    searchable passages each has. Use when the user asks what files you can see."""
    _build_index()
    if not _cache["chunks"]:
        return "The docs folder is empty."
    counts = Counter(c["source"] for c in _cache["chunks"])
    lines = [f"- {name}: {n} passages" for name, n in sorted(counts.items())]
    return "\n".join(lines) + (
        "\n(A file with very few passages may be a scanned PDF with no readable text.)"
    )


docs_tools = [search_documents, list_documents]