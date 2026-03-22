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

    # Fetch ALL collections
    for url, count in collections:
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


def extract_contract_text(html: str) -> str:
    """Extract the full contract text from a Justia contract page."""
    # Contract filing text is the bulk of the page between nav and footer
    # Find content starting from EX- filing marker
    match = re.search(r"(EX-\d+.*?)(?:<footer|class=\"footer\"|© \d{4})", html, re.DOTALL)
    if not match:
        # Try from the first SEC exhibit marker
        match = re.search(r"(Exhibit \d+.*?)(?:<footer|class=\"footer\"|© \d{4})", html, re.DOTALL)
    if not match:
        return ""

    text = match.group(1)
    # Clean HTML
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&#\d+;", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def fetch_contracts(clauses: list[dict], api_key: str, max_contracts: int = 50) -> dict[str, str]:
    """Fetch full contract texts for unique contracts referenced in clauses.

    Returns dict mapping contract URL (without fragment) to contract text.
    """
    # Deduplicate contract URLs (strip #clause-id fragment)
    unique_urls = {}
    for c in clauses:
        url = c.get("contract_url", "")
        if not url:
            continue
        base_url = url.split("#")[0]
        if base_url not in unique_urls:
            unique_urls[base_url] = c["contract"]

    print(f"\nFetching {min(len(unique_urls), max_contracts)} contracts (of {len(unique_urls)} unique)...")
    contracts = {}
    for i, (url, name) in enumerate(list(unique_urls.items())[:max_contracts]):
        time.sleep(1)
        print(f"  [{i+1}] {name}: {url}")
        try:
            html = fetch_page(url, api_key)
            text = extract_contract_text(html)
            if len(text) > 500:
                contracts[url] = text
                print(f"    {len(text)} chars")
            else:
                print(f"    Too short ({len(text)} chars), skipping")
        except Exception as e:
            print(f"    Error: {e}")

    print(f"Fetched {len(contracts)} contracts")
    return contracts


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
    import argparse

    parser = argparse.ArgumentParser(description="Scrape Justia contract clauses")
    parser.add_argument("category", nargs="?", help="Single category slug to scrape")
    parser.add_argument("--fetch-contracts", action="store_true", help="Also fetch full contract texts")
    parser.add_argument("--max-contracts", type=int, default=50, help="Max contracts to fetch")
    args = parser.parse_args()

    if args.category:
        slug = args.category
        name = CATEGORIES.get(slug, slug.replace("-", " ").title())
        categories = {slug: name}
    else:
        categories = CATEGORIES

    clauses = scrape_all(categories)

    output_path = "justia_dataset.json"
    with open(output_path, "w") as f:
        json.dump(clauses, f, indent=2)
    print(f"\nSaved {len(clauses)} clauses to {output_path}")

    if args.fetch_contracts:
        api_key = get_api_key()
        contracts = fetch_contracts(clauses, api_key, max_contracts=args.max_contracts)
        contracts_path = "justia_contracts.json"
        with open(contracts_path, "w") as f:
            json.dump(contracts, f, indent=2)
        print(f"Saved {len(contracts)} contracts to {contracts_path}")

    # Summary by type
    from collections import Counter
    type_counts = Counter(c["clause_type"] for c in clauses)
    for t, n in type_counts.most_common():
        print(f"  {t}: {n}")
