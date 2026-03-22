"""
Scrape contract clause examples from Justia using Cloudflare Browser Rendering.

Fetches clause collections across multiple categories (termination, indemnification,
governing law, etc.) and extracts clean clause text with contract attribution.

Usage:
    python scrape_justia.py                    # Scrape all categories
    python scrape_justia.py termination        # Scrape one category
    CLOUDFLARE_CRAWL_KEY=... python scrape_justia.py

Output: justia_dataset.json
"""

import json
import os
import re
import sys
import time

import requests

ACCOUNT_ID = "4ea1ce574208203469c1d217c1152d03"
CF_API_BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/browser-rendering"

# Clause categories to scrape (URL slug → display name)
CATEGORIES = {
    "termination": "Termination",
    "indemnification": "Indemnification",
    "governing-law": "Governing Law",
    "limitation-of-liability": "Limitation of Liability",
    "confidentiality": "Confidentiality",
    "force-majeure": "Force Majeure",
    "assignment": "Assignment",
    "representations-and-warranties-of-the-company": "Representations and Warranties",
    "events-of-default": "Events of Default",
    "waiver-of-jury-trial": "Waiver of Jury Trial",
    "change-of-control": "Change of Control",
    "non-competition": "Non-Competition",
    "severability": "Severability",
    "entire-agreement": "Entire Agreement",
    "dispute-resolution": "Dispute Resolution",
}


def get_api_key() -> str:
    key = os.environ.get("CLOUDFLARE_CRAWL_KEY", "")
    if not key:
        raise ValueError("Set CLOUDFLARE_CRAWL_KEY environment variable")
    return key


def fetch_page(url: str, api_key: str) -> str:
    """Fetch a page's rendered HTML via Cloudflare Browser Rendering."""
    resp = requests.post(
        f"{CF_API_BASE}/content",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "url": url,
            "rejectResourceTypes": ["image", "font", "stylesheet", "media"],
            "gotoOptions": {"waitUntil": "networkidle2"},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"CF API error: {data.get('errors')}")
    return data["result"]


def extract_collection_links(html: str, category_slug: str) -> list[tuple[str, int]]:
    """Extract 'View All N Similar Clauses' links from a category index page."""
    pattern = rf'<a[^>]*href="(https://contracts\.justia\.com/contract-clauses/{re.escape(category_slug)}/\d+/)"[^>]*>.*?View All ([\d,]+) Similar Clauses'
    matches = re.findall(pattern, html, re.DOTALL)
    return [(url, int(count.replace(",", ""))) for url, count in matches]


def extract_clauses_from_collection(html: str, clause_type: str) -> list[dict]:
    """Extract individual clause texts from a collection page."""
    # Each clause ends with a "clause-found-in" div containing the contract link
    blocks = re.split(r'<div class="clause-found-in">', html)

    results = []
    for i, block in enumerate(blocks[1:], 1):
        # Contract info
        contract_match = re.search(r'<a[^>]*href="([^"]*)"[^>]*>\s*([^<]+?)\s*</a>', block)
        contract_name = contract_match.group(2).strip() if contract_match else "Unknown"
        contract_url = contract_match.group(1).strip() if contract_match else ""

        # Clause text is in the previous block
        prev_block = blocks[i - 1] if i - 1 < len(blocks) else ""

        # Strip diffs: remove <del>, keep <ins> content
        clause_html = re.sub(r"<del[^>]*>.*?</del>", "", prev_block, flags=re.DOTALL)
        clause_html = re.sub(r"<ins[^>]*>(.*?)</ins>", r"\1", clause_html, flags=re.DOTALL)

        # Find clause text starting from the bold title
        # Try common patterns: <strong>Title</strong>. text...
        clause_match = re.search(
            r"<strong>([^<]+)</strong>\.\s*(.*)", clause_html, re.DOTALL
        )
        if not clause_match:
            continue

        title = clause_match.group(1).strip()
        text = clause_match.group(2)

        # Clean HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Trim at navigation elements
        for marker in ["View More", "View All", "Show Differences"]:
            idx = text.find(marker)
            if idx > 0:
                text = text[:idx].strip()

        text = f"{title}. {text}"

        if len(text) > 80:
            results.append({
                "clause_text": text[:3000],
                "contract": contract_name,
                "contract_url": contract_url,
                "clause_type": clause_type,
            })

    return results


def scrape_category(category_slug: str, category_name: str, api_key: str) -> list[dict]:
    """Scrape all clause collections for a category."""
    print(f"\n{'='*60}")
    print(f"Category: {category_name} ({category_slug})")
    print(f"{'='*60}")

    # Fetch category index page
    index_url = f"https://contracts.justia.com/contract-clauses/{category_slug}/"
    print(f"Fetching index: {index_url}")
    html = fetch_page(index_url, api_key)

    # Get collection links
    collections = extract_collection_links(html, category_slug)
    print(f"Found {len(collections)} collections")

    # Also extract clauses from the index page itself
    all_clauses = extract_clauses_from_collection(html, category_name)
    print(f"  Index page: {len(all_clauses)} clauses")

    # Fetch top collections (limit to avoid rate limiting)
    for url, count in collections[:5]:
        time.sleep(1)  # Rate limit
        print(f"  Fetching collection ({count} clauses): {url}")
        try:
            coll_html = fetch_page(url, api_key)
            clauses = extract_clauses_from_collection(coll_html, category_name)
            print(f"    Extracted {len(clauses)} clauses")
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"    Error: {e}")

    # Deduplicate by clause text (first 200 chars)
    seen = set()
    unique = []
    for c in all_clauses:
        key = c["clause_text"][:200]
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"Total unique clauses for {category_name}: {len(unique)}")
    return unique


def scrape_all(categories: dict[str, str] | None = None) -> list[dict]:
    """Scrape clauses across all categories."""
    api_key = get_api_key()
    if categories is None:
        categories = CATEGORIES

    all_clauses = []
    for slug, name in categories.items():
        try:
            clauses = scrape_category(slug, name, api_key)
            all_clauses.extend(clauses)
        except Exception as e:
            print(f"  ERROR scraping {name}: {e}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_clauses)} clauses across {len(categories)} categories")
    print(f"{'='*60}")

    return all_clauses


if __name__ == "__main__":
    if len(sys.argv) > 1:
        slug = sys.argv[1]
        name = CATEGORIES.get(slug, slug.replace("-", " ").title())
        categories = {slug: name}
    else:
        categories = CATEGORIES

    clauses = scrape_all(categories)

    output_path = "justia_dataset.json"
    with open(output_path, "w") as f:
        json.dump(clauses, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Summary by type
    from collections import Counter
    type_counts = Counter(c["clause_type"] for c in clauses)
    for t, n in type_counts.most_common():
        print(f"  {t}: {n}")
