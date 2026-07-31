"""Download a fresh SEC EDGAR Exhibit 10 contract pool for blind evaluation.

This script deliberately only builds an unlabeled contract pool. Labels are
derived later from contract section headings, so the current ColBERT model never
creates its own evaluation targets.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO", "JPM",
    "LLY", "V", "UNH", "XOM", "MA", "COST", "WMT", "HD", "PG", "NFLX",
    "CRM", "ADBE", "ORCL", "CSCO", "ACN", "IBM", "INTC", "AMD", "QCOM",
    "NOW", "SHOP", "UBER", "ABNB", "SNOW", "PLTR", "COIN", "RBLX", "SQ",
    "ROKU", "ZM", "DOCU", "OKTA", "DDOG", "CRWD", "NET", "MDB", "TWLO",
    "TEAM", "ZS", "PANW", "FTNT", "HUBS", "WDAY", "PAYC", "FSLY", "U",
    "PATH", "BILL", "TOST", "RIVN", "LCID", "CVNA", "DASH", "LYFT",
]


class TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "tr", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def fetch_json(url: str, user_agent: str, delay: float) -> object:
    time.sleep(delay)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=30) as response:
        return json.load(response)


def fetch_text(url: str, user_agent: str, delay: float) -> str:
    time.sleep(delay)
    req = Request(url, headers={"User-Agent": user_agent})
    with urlopen(req, timeout=45) as response:
        data = response.read()
    return data.decode("utf-8", errors="ignore")


def html_to_text(raw: str) -> str:
    raw = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", " ", raw)
    parser = TextHTMLParser()
    parser.feed(raw)
    text = html.unescape(parser.text())
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode("utf-8", errors="ignore")).hexdigest()


def load_ticker_map(user_agent: str, delay: float) -> dict[str, int]:
    data = fetch_json("https://www.sec.gov/files/company_tickers_exchange.json", user_agent, delay)
    rows = data.get("data", []) if isinstance(data, dict) else []
    return {str(row[2]).upper(): int(row[0]) for row in rows if len(row) >= 3}


def accession_path(accession: str) -> str:
    return accession.replace("-", "")


def is_ex10_document(name: str, description: str) -> bool:
    haystack = f"{name} {description}".lower()
    if "graphic" in haystack or "calculation" in haystack:
        return False
    return bool(
        re.search(r"\bex(?:hibit)?[-_\s]?10(?:[\.\-_]\d+)?\b", haystack)
        or re.search(r"\b10(?:[\.\-_]\d+)?\b", description.lower())
    )


def iter_recent_filings(submissions: dict, forms: set[str]) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    keys = ["accessionNumber", "filingDate", "form", "primaryDocument"]
    values = [recent.get(k, []) for k in keys]
    filings = []
    for row in zip(*values):
        filing = dict(zip(keys, row))
        if filing["form"] in forms:
            filings.append(filing)
    return filings


def iter_current_feed_filings(user_agent: str, delay: float, form: str, count: int, pages: int) -> list[dict]:
    ns = {"a": "http://www.w3.org/2005/Atom"}
    filings = []
    for page in range(pages):
        start = page * count
        url = (
            "https://www.sec.gov/cgi-bin/browse-edgar"
            f"?action=getcurrent&type={form}&count={count}&start={start}&output=atom"
        )
        raw = ""
        for attempt in range(3):
            try:
                raw = fetch_text(url, user_agent, delay)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    print(f"skipping feed page start={start}: {exc!r}", file=sys.stderr, flush=True)
                    raw = ""
                    break
                time.sleep(2 * (attempt + 1))
            except (URLError, TimeoutError) as exc:
                if attempt == 2:
                    print(f"skipping feed page start={start}: {exc!r}", file=sys.stderr, flush=True)
                    raw = ""
                    break
                time.sleep(2 * (attempt + 1))
        if not raw:
            continue
        root = ET.fromstring(raw)
        for entry in root.findall("a:entry", ns):
            title = entry.findtext("a:title", default="", namespaces=ns)
            updated = entry.findtext("a:updated", default="", namespaces=ns)
            link = next((node.attrib.get("href", "") for node in entry.findall("a:link", ns) if node.attrib.get("href")), "")
            match = re.search(r"/data/(\d+)/(\d+)/", link)
            accession_match = re.search(r"accession-number=([0-9-]+)", entry.findtext("a:id", default="", namespaces=ns))
            if not match or not accession_match:
                continue
            company_match = re.search(r"-\s*(.*?)\s*\(\d{10}\)", title)
            filings.append({
                "cik": int(match.group(1)),
                "ticker": "",
                "company": company_match.group(1).strip() if company_match else title,
                "form": form,
                "filingDate": updated[:10],
                "accessionNumber": accession_match.group(1),
            })
    return filings


def collect_contracts(args: argparse.Namespace) -> tuple[list[dict], dict]:
    user_agent = args.sec_user_agent
    forms = set(args.forms)
    contracts: list[dict] = []
    seen_hashes: set[str] = set()
    attempts = 0
    errors: list[dict] = []

    if args.from_current_feed:
        feed_filings = []
        for form in args.feed_forms:
            feed_filings.extend(iter_current_feed_filings(user_agent, args.delay, form, args.feed_count, args.feed_pages))
        for filing in feed_filings:
            if len(contracts) >= args.max_contracts:
                break
            cik = filing["cik"]
            accession = filing["accessionNumber"]
            acc_path = accession_path(accession)
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/index.json"
            try:
                index = fetch_json(index_url, user_agent, args.delay)
            except (HTTPError, URLError, TimeoutError) as exc:
                errors.append({"cik": cik, "accession": accession, "error": repr(exc)})
                continue
            items = index.get("directory", {}).get("item", []) if isinstance(index, dict) else []
            for item in items:
                if len(contracts) >= args.max_contracts:
                    break
                name = str(item.get("name", ""))
                description = str(item.get("description", ""))
                if not name.lower().endswith((".htm", ".html", ".txt")):
                    continue
                if not is_ex10_document(name, description):
                    continue
                attempts += 1
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/{name}"
                try:
                    raw = fetch_text(doc_url, user_agent, args.delay)
                except (HTTPError, URLError, TimeoutError) as exc:
                    errors.append({"cik": cik, "accession": accession, "document": name, "error": repr(exc)})
                    continue
                text = html_to_text(raw) if "<html" in raw[:5000].lower() or "<document" in raw[:5000].lower() else raw
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                words = len(re.findall(r"\b\w+\b", text))
                if words < args.min_words or words > args.max_words:
                    continue
                digest = stable_hash(text)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                company = filing.get("company") or str(cik)
                contracts.append({
                    "contract_id": f"edgar-feed-{cik}-{accession}-{name}",
                    "ticker": filing.get("ticker", ""),
                    "company": company,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filingDate"],
                    "accession": accession,
                    "document": name,
                    "description": description,
                    "url": doc_url,
                    "sha1": digest,
                    "words": words,
                    "text": text,
                })
                if args.verbose:
                    print(
                        f"found {len(contracts)}/{args.max_contracts}: {company} {accession} {name} ({words} words)",
                        file=sys.stderr,
                        flush=True,
                    )
        manifest = {
            "source": "SEC EDGAR current filings feed and public Archives Exhibit 10 documents",
            "contract_count": len(contracts),
            "attempted_exhibit_documents": attempts,
            "feed_forms": args.feed_forms,
            "feed_count": args.feed_count,
            "feed_pages": args.feed_pages,
            "min_words": args.min_words,
            "max_words": args.max_words,
            "sec_user_agent": user_agent,
            "errors": errors[:50],
        }
        return contracts, manifest

    ticker_map = load_ticker_map(user_agent, args.delay)
    tickers = [ticker.upper() for ticker in args.tickers]

    for ticker in tickers:
        cik = ticker_map.get(ticker)
        if not cik:
            errors.append({"ticker": ticker, "error": "ticker not found"})
            continue
        padded = f"{cik:010d}"
        try:
            submissions = fetch_json(
                f"https://data.sec.gov/submissions/CIK{padded}.json",
                user_agent,
                args.delay,
            )
        except (HTTPError, URLError, TimeoutError) as exc:
            errors.append({"ticker": ticker, "error": repr(exc)})
            continue

        filings = iter_recent_filings(submissions, forms)[: args.max_filings_per_ticker]
        ticker_found = 0
        for filing in filings:
            if len(contracts) >= args.max_contracts:
                break
            accession = filing["accessionNumber"]
            acc_path = accession_path(accession)
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/index.json"
            try:
                index = fetch_json(index_url, user_agent, args.delay)
            except (HTTPError, URLError, TimeoutError) as exc:
                errors.append({"ticker": ticker, "accession": accession, "error": repr(exc)})
                continue
            items = index.get("directory", {}).get("item", []) if isinstance(index, dict) else []
            for item in items:
                if len(contracts) >= args.max_contracts:
                    break
                name = str(item.get("name", ""))
                description = str(item.get("description", ""))
                if not name.lower().endswith((".htm", ".html", ".txt")):
                    continue
                if not is_ex10_document(name, description):
                    continue
                attempts += 1
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_path}/{name}"
                try:
                    raw = fetch_text(doc_url, user_agent, args.delay)
                except (HTTPError, URLError, TimeoutError) as exc:
                    errors.append({"ticker": ticker, "accession": accession, "document": name, "error": repr(exc)})
                    continue
                text = html_to_text(raw) if "<html" in raw[:5000].lower() or "<document" in raw[:5000].lower() else raw
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                words = len(re.findall(r"\b\w+\b", text))
                if words < args.min_words or words > args.max_words:
                    continue
                digest = stable_hash(text)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                contracts.append({
                    "contract_id": f"edgar-{ticker}-{accession}-{name}",
                    "ticker": ticker,
                    "cik": cik,
                    "form": filing["form"],
                    "filing_date": filing["filingDate"],
                    "accession": accession,
                    "document": name,
                    "description": description,
                    "url": doc_url,
                    "sha1": digest,
                    "words": words,
                    "text": text,
                })
                ticker_found += 1
                if args.verbose:
                    print(
                        f"found {len(contracts)}/{args.max_contracts}: {ticker} {accession} {name} ({words} words)",
                        file=sys.stderr,
                        flush=True,
                    )
        if args.verbose:
            print(
                f"done {ticker}: filings_checked={len(filings)} contracts_found={ticker_found}",
                file=sys.stderr,
                flush=True,
            )
        if len(contracts) >= args.max_contracts:
            break

    manifest = {
        "source": "SEC EDGAR public Archives Exhibit 10 documents",
        "contract_count": len(contracts),
        "attempted_exhibit_documents": attempts,
        "forms": sorted(forms),
        "tickers": tickers,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "sec_user_agent": user_agent,
        "errors": errors[:50],
    }
    return contracts, manifest


def write_outputs(contracts: list[dict], manifest: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "raw_contracts.jsonl", "w") as f:
        for row in contracts:
            f.write(json.dumps(row) + "\n")
    with open(output_dir / "pool_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="eval_blind_edgar")
    parser.add_argument("--sec-user-agent", default=None)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--forms", nargs="+", default=["8-K", "10-K", "10-Q", "S-1", "S-1/A", "F-1", "F-1/A", "20-F", "6-K"])
    parser.add_argument("--from-current-feed", action="store_true")
    parser.add_argument("--feed-forms", nargs="+", default=["8-K"])
    parser.add_argument("--feed-count", type=int, default=100)
    parser.add_argument("--feed-pages", type=int, default=3)
    parser.add_argument("--max-contracts", type=int, default=30)
    parser.add_argument("--max-filings-per-ticker", type=int, default=12)
    parser.add_argument("--min-words", type=int, default=800)
    parser.add_argument("--max-words", type=int, default=35000)
    parser.add_argument("--delay", type=float, default=0.11)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.sec_user_agent:
        raise SystemExit("Pass --sec-user-agent with a descriptive contact per SEC access guidance.")

    contracts, manifest = collect_contracts(args)
    write_outputs(contracts, manifest, Path(args.output_dir))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
