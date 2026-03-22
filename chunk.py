"""
Hierarchical chunker for legal documents (contracts, credit agreements, etc.).

Parses document structure (Articles, Sections, subsections, bullet items),
groups semantically related content, and produces chunks sized for ColBERT
retrieval with section-path context headers.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path


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
    """Rough token estimate: ~0.75 tokens per word for legal text."""
    return int(len(text.split()) * 0.75)


def split_at_sentences(text: str, max_tokens: int, overlap_sentences: int = 2) -> list[str]:
    """Split text at sentence boundaries with overlap when it exceeds max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return [text]

    # Split on sentence endings
    sentences = re.split(r"(?<=[.;])\s+", text)
    chunks = []
    current = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            chunks.append(" ".join(current))
            # Overlap: keep last N sentences
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(estimate_tokens(s) for s in current)
        current.append(sent)
        current_tokens += sent_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def chunk_document(
    text: str,
    max_tokens: int = 384,
    overlap_sentences: int = 2,
    offset_to_page: dict[int, int] | None = None,
) -> list[Chunk]:
    """
    Chunk a legal document into semantically coherent pieces.

    Strategy:
    1. Parse into Articles → Sections → Subsections
    2. Definitions get individual chunks per term
    3. Subsections ((a), (b), etc.) stay with their parent if under max_tokens
    4. Long sections are split at sentence boundaries with overlap
    5. Every chunk gets a section-path header for context

    Args:
        text: full document text
        max_tokens: target maximum tokens per chunk
        overlap_sentences: number of sentences to overlap when splitting
        offset_to_page: char-offset-to-page mapping from PDF extraction

    Returns:
        list of Chunk objects
    """
    text = clean_text(text)
    chunks = []

    # Separate TOC/preamble from body
    preamble, body = strip_toc(text)
    if preamble:
        chunks.append(Chunk(
            text=preamble[:2000],  # Cap preamble size
            section_path=["PREAMBLE"],
            chunk_type="preamble",
            page=1,
        ))

    articles = split_into_articles(body)

    for article_header, article_body, article_offset in articles:
        # Skip TOC
        if is_toc_section(article_body):
            continue

        sections = split_into_sections(article_body)

        for section_header, section_body, section_offset in sections:
            section_body_clean = clean_text(section_body)
            if not section_body_clean or is_toc_section(section_body_clean):
                continue

            # Build section path
            path = []
            if article_header:
                path.append(article_header)
            if section_header:
                path.append(section_header)

            # Determine page
            abs_offset = article_offset + section_offset
            page = get_page_for_offset(abs_offset, offset_to_page) if offset_to_page else None

            # --- Definitions section: chunk per term ---
            if detect_definitions_section(section_body_clean):
                defs = split_definitions(section_body_clean)
                # Group small definitions together up to max_tokens
                group_text = ""
                group_terms = []
                for term, def_text in defs:
                    if estimate_tokens(group_text + "\n" + def_text) > max_tokens and group_text:
                        chunks.append(Chunk(
                            text=group_text.strip(),
                            section_path=path + ([f"Definitions: {', '.join(group_terms)}"] if group_terms else []),
                            chunk_type="definition",
                            page=page,
                        ))
                        group_text = ""
                        group_terms = []
                    group_text += "\n" + def_text
                    if term:
                        group_terms.append(term)
                if group_text.strip():
                    chunks.append(Chunk(
                        text=group_text.strip(),
                        section_path=path + ([f"Definitions: {', '.join(group_terms)}"] if group_terms else []),
                        chunk_type="definition",
                        page=page,
                    ))
                continue

            # --- Regular section ---
            # Try keeping whole section as one chunk
            if estimate_tokens(section_body_clean) <= max_tokens:
                chunks.append(Chunk(
                    text=section_body_clean,
                    section_path=path,
                    chunk_type="clause",
                    page=page,
                ))
                continue

            # Try subsection-level grouping
            subsections = split_subsections(section_body_clean)
            if len(subsections) > 1:
                # Try grouping: parent intro + subsections that fit together
                intro = subsections[0] if not RE_SUBSECTION.match(subsections[0]) else ""
                items = subsections[1:] if intro else subsections

                group_text = intro
                for item in items:
                    combined = (group_text + "\n" + item).strip() if group_text else item
                    if estimate_tokens(combined) > max_tokens and group_text:
                        # Flush current group
                        chunks.append(Chunk(
                            text=group_text.strip(),
                            section_path=path,
                            chunk_type="clause",
                            page=page,
                        ))
                        # Start new group with intro context for continuity
                        group_text = f"{intro}\n{item}".strip() if intro else item
                    else:
                        group_text = combined

                if group_text.strip():
                    chunks.append(Chunk(
                        text=group_text.strip(),
                        section_path=path,
                        chunk_type="clause",
                        page=page,
                    ))
            else:
                # No subsections — split at sentence boundaries
                parts = split_at_sentences(section_body_clean, max_tokens, overlap_sentences)
                for part in parts:
                    chunks.append(Chunk(
                        text=part.strip(),
                        section_path=path,
                        chunk_type="clause",
                        page=page,
                    ))

    return chunks


def chunk_pdf(pdf_path: str, max_tokens: int = 384) -> list[Chunk]:
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
