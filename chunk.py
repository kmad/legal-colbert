"""
Hierarchical chunker for legal documents (contracts, credit agreements, etc.).

Two strategies:
  1. Structural (regex) — for well-formatted PDFs with Article/Section headers
  2. Semantic (embedding similarity) — for messy text-only data

Auto-detects which strategy to use based on structural marker density.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Chunk:
    text: str
    section_path: list[str]
    chunk_type: str  # "definition", "clause", "preamble", "exhibit", "toc"
    page: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def header(self) -> str:
        return " > ".join(self.section_path) if self.section_path else ""

    @property
    def full_text(self) -> str:
        """Chunk text with section path prepended for retrieval context."""
        if self.header:
            return f"[{self.header}]\n{self.text}"
        return self.text


# ---------------------------------------------------------------------------
# Regex patterns for legal document structure
# ---------------------------------------------------------------------------

# Article headers: "ARTICLE 1", "ARTICLE IV", etc.
RE_ARTICLE = re.compile(
    r"^(ARTICLE\s+(?:\d+|[IVXLC]+))\s*\n\s*(.+?)$",
    re.MULTILINE,
)

# Section headers: "Section 1.01.", "Section 12.03." (not TOC lines with dotted leaders)
RE_SECTION = re.compile(
    r"^(Section\s+\d+\.\d+)\.\s{1,4}(\S.+?)(?:\.\s|\s{2,})",
    re.MULTILINE,
)

# Subsection markers: "(a)", "(b)", "(i)", "(ii)", "(1)", "(2)"
RE_SUBSECTION = re.compile(
    r"^\s*(\([a-z]\)|\([ivxlc]+\)|\(\d+\))\s+",
    re.MULTILINE,
)

# Definition entries: "Term": or "Term" means (handles smart quotes)
RE_DEFINITION = re.compile(
    r'^["\u201c]([^"\u201d]+)["\u201d](?::|\s+means?\b)',
    re.MULTILINE,
)

# Page number / footer noise
RE_PAGE_NOISE = re.compile(
    r"^\d+\s*$|^\(NY\)\s+\d+/.*\.doc\s*$",
    re.MULTILINE,
)

# Table of contents lines (dotted leaders)
RE_TOC_LINE = re.compile(r"\.{5,}")


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict[int, int]]:
    """Extract text from PDF, returning full text and char-offset-to-page mapping."""
    import fitz

    doc = fitz.open(pdf_path)
    full_text = ""
    offset_to_page = {}
    for i, page in enumerate(doc):
        start = len(full_text)
        full_text += page.get_text()
        offset_to_page[start] = i + 1
    return full_text, offset_to_page


def get_page_for_offset(offset: int, offset_to_page: dict[int, int]) -> int:
    """Find page number for a character offset."""
    page = 1
    for start_offset, page_num in sorted(offset_to_page.items()):
        if start_offset <= offset:
            page = page_num
        else:
            break
    return page


def clean_text(text: str) -> str:
    """Remove page numbers, footers, and excessive whitespace."""
    text = RE_PAGE_NOISE.sub("", text)
    # Collapse runs of 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_toc(text: str) -> tuple[str, str]:
    """Separate preamble/TOC from the body of a legal document.

    Looks for common body-start markers like 'agree as follows' or the first
    ARTICLE heading that is NOT followed by dotted TOC leaders.
    """
    # Try "agree as follows" marker
    body_markers = [
        r"agree\s+as\s+follows",
        r"WITNESSETH",
        r"NOW,?\s+THEREFORE",
    ]
    for pattern in body_markers:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            # Find the next ARTICLE after this marker
            article_after = RE_ARTICLE.search(text[m.end():])
            if article_after:
                body_start = m.end() + article_after.start()
                return text[:body_start].strip(), text[body_start:].strip()
            # No article found, return from marker
            return text[:m.end()].strip(), text[m.end():].strip()

    # Fallback: find first ARTICLE that is NOT in a TOC (no dotted leaders nearby)
    for m in RE_ARTICLE.finditer(text):
        # Check if this article line is followed by TOC-style content
        next_200 = text[m.end():m.end() + 200]
        if not RE_TOC_LINE.search(next_200):
            return text[:m.start()].strip(), text[m.start():].strip()

    return "", text


def is_toc_section(text: str) -> bool:
    """Check if text is a table of contents section."""
    return bool(RE_TOC_LINE.search(text))


def split_into_articles(text: str) -> list[tuple[str, str, int]]:
    """Split document into (article_header, article_body, offset) tuples."""
    matches = list(RE_ARTICLE.finditer(text))
    if not matches:
        return [("", text, 0)]

    # Everything before first article is preamble
    articles = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        articles.append(("PREAMBLE", preamble, 0))

    for i, match in enumerate(matches):
        article_id = match.group(1).strip()
        article_title = match.group(2).strip()
        header = f"{article_id}: {article_title}"
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        articles.append((header, body, start))

    return articles


def split_into_sections(article_body: str) -> list[tuple[str, str, int]]:
    """Split an article body into (section_header, section_body, offset) tuples."""
    matches = list(RE_SECTION.finditer(article_body))
    if not matches:
        return [("", article_body, 0)]

    sections = []
    # Text before first section (usually the article header line)
    pre = article_body[: matches[0].start()].strip()
    if pre and not is_toc_section(pre):
        sections.append(("", pre, 0))

    for i, match in enumerate(matches):
        sec_id = match.group(1).strip()
        sec_title = match.group(2).strip()
        header = f"{sec_id}: {sec_title}"
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(article_body)
        body = article_body[start:end].strip()
        sections.append((header, body, start))

    return sections


def split_subsections(text: str) -> list[str]:
    """Split section text at subsection markers like (a), (b), keeping markers attached."""
    parts = RE_SUBSECTION.split(text)
    if len(parts) <= 1:
        return [text]

    # parts alternates: [pre_text, marker1, body1, marker2, body2, ...]
    result = []
    if parts[0].strip():
        result.append(parts[0].strip())

    for i in range(1, len(parts), 2):
        marker = parts[i]
        body = parts[i + 1] if i + 1 < len(parts) else ""
        result.append(f"{marker} {body}".strip())

    return result


def detect_definitions_section(text: str) -> bool:
    """Heuristic: section is a definitions block if it has 5+ definition entries."""
    return len(RE_DEFINITION.findall(text)) >= 5


def split_definitions(text: str) -> list[tuple[str, str]]:
    """Split a definitions section into individual (term, definition_text) pairs."""
    matches = list(RE_DEFINITION.finditer(text))
    if not matches:
        return [("", text)]

    defs = []
    for i, match in enumerate(matches):
        term = match.group(1)
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        def_text = text[start:end].strip()
        defs.append((term, def_text))

    return defs


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~1.4 BPE tokens per word for legal text.

    The old 0.75 multiplier UNDERestimated by ~2x, producing "384-token"
    chunks of 700+ real tokens — far beyond the model's document_length
    of 300 (parity_test.py finding, 2026-07-31).
    """
    return int(len(text.split()) * 1.4)


def split_long_sentence(sent: str, max_tokens: int, overlap_words: int = 15) -> list[str]:
    """Word-boundary fallback for a single sentence exceeding max_tokens.

    Legal text contains run-on sentences (300+ words) that sentence splitting
    cannot break; without this they exceed the model's document_length and get
    truncated at encode time, silently making the tail unretrievable.
    """
    words = sent.split()
    step = max(int(max_tokens * 0.9 / 1.4), overlap_words + 1)  # words per piece
    pieces = []
    start = 0
    while start < len(words):
        pieces.append(" ".join(words[start : start + step]))
        if start + step >= len(words):
            break
        start += step - overlap_words
    return pieces


def split_at_sentences(text: str, max_tokens: int, overlap_sentences: int = 2) -> list[str]:
    """Split text at sentence boundaries with overlap when it exceeds max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split on sentence endings
    sentences = re.split(r"(?<=[.;])\s+", text)
    # Word-level fallback for atomic sentences that alone exceed the limit
    expanded = []
    for sent in sentences:
        if estimate_tokens(sent) > max_tokens:
            expanded.extend(split_long_sentence(sent, max_tokens))
        else:
            expanded.append(sent)
    sentences = expanded
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N sentences, but never more than 1/4 of the
            # token budget (fallback-split pieces are near max_tokens each,
            # so a fixed sentence count can blow past the cap)
            overlap = []
            overlap_tokens = 0
            for s in reversed(current[-overlap_sentences:] if overlap_sentences else []):
                s_tokens = estimate_tokens(s)
                if overlap_tokens + s_tokens > max_tokens // 4:
                    break
                overlap.insert(0, s)
                overlap_tokens += s_tokens
            current = overlap
            current_tokens = overlap_tokens
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Semantic chunking (for messy / unstructured text)
# ---------------------------------------------------------------------------

_sentence_model = None


def _get_sentence_model():
    """Lazy-load a lightweight sentence embedding model."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling legal text patterns.

    Splits on period/semicolon followed by space, but avoids splitting
    on abbreviations like 'Inc.', 'Ltd.', 'No.', section refs like 'Section 1.01.'.
    """
    # Protect common abbreviations
    protected = text
    abbrevs = [r"Inc\.", r"Ltd\.", r"Corp\.", r"No\.", r"Nos\.", r"Mr\.", r"Ms\.",
               r"Dr\.", r"Jr\.", r"Sr\.", r"St\.", r"vs\.", r"etc\.", r"i\.e\.",
               r"e\.g\.", r"U\.S\.", r"Sec\.", r"\d+\.\d+\."]
    for abbr in abbrevs:
        protected = re.sub(abbr, lambda m: m.group().replace(".", "∎"), protected)

    # Split on sentence boundaries
    raw_sentences = re.split(r"(?<=[.;:!?])\s+", protected)

    # Restore protected periods
    sentences = [s.replace("∎", ".") for s in raw_sentences]

    # Filter empty
    return [s.strip() for s in sentences if s.strip()]


def compute_similarities(sentences: list[str]) -> np.ndarray:
    """Compute cosine similarity between consecutive sentences."""
    model = _get_sentence_model()
    embeddings = model.encode(sentences, show_progress_bar=False, normalize_embeddings=True)

    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        similarities.append(float(sim))

    return np.array(similarities)


def find_breakpoints(
    similarities: np.ndarray,
    threshold_percentile: int = 25,
    min_chunk_sentences: int = 3,
) -> list[int]:
    """Find chunk boundaries where semantic similarity drops.

    Uses a percentile-based threshold: breakpoints are where similarity
    falls below the Nth percentile of all consecutive similarities.

    Args:
        similarities: array of cosine similarities between consecutive sentences
        threshold_percentile: break when similarity is below this percentile
        min_chunk_sentences: minimum sentences per chunk to avoid tiny fragments
    """
    if len(similarities) == 0:
        return []

    threshold = np.percentile(similarities, threshold_percentile)
    breakpoints = []
    last_break = 0

    for i, sim in enumerate(similarities):
        if sim < threshold and (i - last_break) >= min_chunk_sentences:
            breakpoints.append(i + 1)  # break AFTER sentence i
            last_break = i + 1

    return breakpoints


def semantic_chunk(
    text: str,
    max_tokens: int = 256,
    threshold_percentile: int = 25,
    min_chunk_sentences: int = 3,
) -> list[Chunk]:
    """Chunk text using sentence embedding similarity.

    1. Split into sentences
    2. Compute embedding similarity between consecutive sentences
    3. Break at points where similarity drops below threshold
    4. Merge small chunks, split large chunks to respect max_tokens
    """
    text = clean_text(text)
    sentences = split_sentences(text)

    if len(sentences) <= 3:
        return [Chunk(text=text, section_path=[], chunk_type="clause")]

    # Compute similarities and find breakpoints
    similarities = compute_similarities(sentences)
    breakpoints = find_breakpoints(similarities, threshold_percentile, min_chunk_sentences)

    # Create initial chunks from breakpoints
    raw_chunks = []
    start = 0
    for bp in breakpoints:
        chunk_text = " ".join(sentences[start:bp])
        raw_chunks.append(chunk_text)
        start = bp
    # Last chunk
    if start < len(sentences):
        raw_chunks.append(" ".join(sentences[start:]))

    # Merge small chunks and split large ones to respect max_tokens
    final_chunks = []
    buffer = ""
    for raw in raw_chunks:
        combined = (buffer + " " + raw).strip() if buffer else raw
        if estimate_tokens(combined) > max_tokens:
            if buffer:
                final_chunks.append(buffer.strip())
            # If this single chunk is too large, split at sentences
            if estimate_tokens(raw) > max_tokens:
                parts = split_at_sentences(raw, max_tokens, overlap_sentences=1)
                final_chunks.extend(parts[:-1])
                buffer = parts[-1]  # carry last part forward for potential merge
            else:
                buffer = raw
        else:
            buffer = combined

    if buffer.strip():
        final_chunks.append(buffer.strip())

    # Convert to Chunk objects
    return [
        Chunk(text=t, section_path=[], chunk_type="semantic")
        for t in final_chunks
        if len(t.split()) >= 5
    ]


def has_structural_markers(text: str) -> bool:
    """Detect if text has enough structural markers for regex-based chunking.

    Returns True if the text has Article/Section patterns typical of
    well-formatted legal documents.
    """
    article_count = len(RE_ARTICLE.findall(text))
    section_count = len(RE_SECTION.findall(text))
    # Need at least 2 articles or 3 sections to trust structural parsing
    return article_count >= 2 or section_count >= 3


def chunk_document(
    text: str,
    max_tokens: int = 256,
    overlap_sentences: int = 2,
    offset_to_page: dict[int, int] | None = None,
) -> list[Chunk]:
    """
    Chunk a legal document into semantically coherent pieces.

    Always uses semantic chunking (embedding similarity between consecutive
    sentences) for consistent results on both structured and messy text.

    Args:
        text: full document text
        max_tokens: target maximum tokens per chunk
        overlap_sentences: number of sentences to overlap when splitting
        offset_to_page: char-offset-to-page mapping from PDF extraction

    Returns:
        list of Chunk objects
    """
    return semantic_chunk(clean_text(text), max_tokens=max_tokens)


def chunk_pdf(pdf_path: str, max_tokens: int = 256) -> list[Chunk]:
    """Convenience: extract text from PDF and chunk it."""
    text, offset_to_page = extract_text_from_pdf(pdf_path)
    return chunk_document(text, max_tokens=max_tokens, offset_to_page=offset_to_page)


if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "example_contract.pdf"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 384

    chunks = chunk_pdf(pdf_path, max_tokens=max_tokens)

    print(f"Total chunks: {len(chunks)}\n")
    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk.text)
        print(f"--- Chunk {i+1} [{chunk.chunk_type}] ~{tokens} tokens (page {chunk.page}) ---")
        print(f"Path: {chunk.header}")
        print(chunk.text[:300])
        if len(chunk.text) > 300:
            print("...")
        print()
