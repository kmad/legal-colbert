"""Scrape a bounded Justia contract-clause sample for V2 hard negatives.

This intentionally samples representative clauses from category pages rather
than trying to mirror Justia's full corpus. The output schema matches
prepare_v2_data.py:

[
  {"clause_type": "Termination", "clause_text": "...", "source_url": "..."}
]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://contracts.justia.com"
INDEX_URL = f"{BASE_URL}/contract-clauses/"
DEFAULT_LABELS = {
    "Assignments",
    "Change of Control",
    "Choice of Law",
    "Confidentiality",
    "Dispute Resolution",
    "Events of Default",
    "Force Majeure",
    "Governing Law",
    "Indemnification",
    "Intellectual Property",
    "Limitation of Liability",
    "Non-Competition",
    "Payments",
    "Severability",
    "Termination",
    "Waiver of Jury Trial",
}


def clean_clause(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^Show Differences\s+", "", text)
    text = re.sub(r"\s*Found in\s+.*$", "", text)
    return text.strip()


def get_soup(url: str, timeout: int = 30) -> BeautifulSoup:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def discover_categories(labels: set[str]) -> dict[str, str]:
    soup = get_soup(INDEX_URL)
    found = {}
    for a in soup.find_all("a"):
        name = a.get_text(" ", strip=True)
        name = re.sub(r"\s*\(\d[\d,]*.*?\)\s*$", "", name).strip()
        if name in labels:
            found[name] = urljoin(BASE_URL, a.get("href", ""))
    return found


def extract_clause_blocks(soup: BeautifulSoup, max_clauses: int) -> list[str]:
    # Justia category pages render representative clause text before repeated
    # "View More" markers. Parsing the text stream is more stable than relying
    # on private CSS classes.
    text = soup.get_text("\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    blocks = []
    current = []
    in_content = False
    for line in lines:
        if line.startswith("This page contains"):
            in_content = True
            continue
        if not in_content:
            continue
        if line in {"Justia Legal Resources", "View Variations"}:
            break
        if line == "View More":
            clause = clean_clause(" ".join(current))
            current = []
            words = clause.split()
            if len(words) >= 30 and "~~" not in clause:
                blocks.append(clause)
                if len(blocks) >= max_clauses:
                    break
            continue
        if line.startswith("Found in") or line.startswith("View All ") or line == "Show Differences":
            continue
        if line.startswith("Grouped Into ") or line.startswith("Contract Clauses"):
            continue
        current.append(line)
    return list(dict.fromkeys(blocks))


def scrape(max_per_category: int, delay: float, labels: set[str]) -> list[dict]:
    categories = discover_categories(labels)
    missing = sorted(labels - set(categories))
    if missing:
        print(f"Warning: did not find categories: {', '.join(missing)}")

    rows = []
    for label, url in sorted(categories.items()):
        print(f"Scraping {label}: {url}")
        soup = get_soup(url)
        clauses = extract_clause_blocks(soup, max_per_category)
        for clause in clauses:
            rows.append({
                "clause_type": label,
                "clause_text": clause,
                "source_url": url,
            })
        print(f"  {len(clauses)} clauses")
        time.sleep(delay)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="justia_dataset.json")
    parser.add_argument("--max-per-category", type=int, default=150)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--labels", nargs="*", default=sorted(DEFAULT_LABELS))
    args = parser.parse_args()

    rows = scrape(args.max_per_category, args.delay, set(args.labels))
    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote {len(rows)} clauses to {out}")


if __name__ == "__main__":
    main()
