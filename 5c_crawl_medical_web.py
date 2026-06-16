"""RoseMed Step 5c: Persistent BFS crawl of whitelisted medical web domains."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from tqdm import tqdm

from crawl_config import (
    CRAWL_ALLOWED_DOMAINS,
    CRAWL_DELAY_SECONDS,
    CRAWL_SEED_URLS,
    CRAWL_SKIP_PATTERNS,
    DEFAULT_PAGES_PER_RUN,
    DOMAIN_ORG,
)

console = Console()
STATE_DIR = Path("./data/sources/crawl")
PAGES_DIR = STATE_DIR / "pages"
VISITED_FILE = STATE_DIR / "visited.json"
QUEUE_FILE = STATE_DIR / "queue.json"
STATS_FILE = STATE_DIR / "stats.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RoseMed/1.0; medical-research)",
    "Accept": "text/html,application/xhtml+xml,application/pdf",
    "Accept-Language": "en,bg;q=0.9",
}


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    if parsed.query and len(parsed.query) < 120:
        clean += f"?{parsed.query}"
    return clean


def _domain_allowed(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    for allowed in CRAWL_ALLOWED_DOMAINS:
        a = allowed[4:] if allowed.startswith("www.") else allowed
        if host == a or host.endswith("." + a):
            return True
    return False


def _should_skip(url: str) -> bool:
    lower = url.lower()
    return any(p in lower for p in CRAWL_SKIP_PATTERNS)


def _org_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    for domain, org in DOMAIN_ORG.items():
        if domain in host:
            return org
    return "Medical Web"


def _lang_for_org(org: str) -> str:
    if org in ("NHIF", "BDA", "MoH Bulgaria", "NCPHP Bulgaria"):
        return "bg"
    return "en"


def _load_json_set(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return set()


def _load_json_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()
    return re.sub(r"\s+", " ", main.get_text(separator=" ")).strip()


def _extract_links(html: str, base_url: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith("#"):
            continue
        full = _normalize_url(urljoin(base_url, href))
        if full.startswith("http") and _domain_allowed(full) and not _should_skip(full):
            links.append(full)
    return links


def _page_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _fetch(url: str, timeout: int = 45) -> Optional[tuple[str, str]]:
    """Return (content_type, body_text_or_bytes_marker)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "pdf" in ctype.lower() or url.lower().endswith(".pdf"):
            pdf_dir = STATE_DIR / "pdfs"
            pdf_dir.mkdir(parents=True, exist_ok=True)
            name = _page_id(url) + ".pdf"
            (pdf_dir / name).write_bytes(r.content)
            return ("pdf", str(pdf_dir / name))
        r.encoding = r.apparent_encoding or "utf-8"
        return ("html", r.text)
    except requests.RequestException:
        return None


def crawl_medical_web(max_pages: int = DEFAULT_PAGES_PER_RUN) -> Dict[str, int]:
    """Crawl up to max_pages new URLs; resume from saved queue."""
    console.print("[bold cyan]RoseMed 5c: Medical Web Crawl[/bold cyan]")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    visited = _load_json_set(VISITED_FILE)
    queue = _load_json_list(QUEUE_FILE)

    if not queue:
        queue = list(CRAWL_SEED_URLS)
        console.print(f"Seeded queue with {len(queue)} URLs")

    stats = {"fetched": 0, "skipped": 0, "errors": 0, "new_links": 0}
    batch = 0

    pbar = tqdm(total=max_pages, desc="Crawl")
    while queue and batch < max_pages:
        url = queue.pop(0)
        if url in visited or _should_skip(url) or not _domain_allowed(url):
            stats["skipped"] += 1
            continue

        visited.add(url)
        result = _fetch(url)
        batch += 1
        pbar.update(1)

        if not result:
            stats["errors"] += 1
            time.sleep(CRAWL_DELAY_SECONDS)
            continue

        ctype, body = result
        org = _org_for_url(url)
        record: Dict[str, Any] = {
            "url": url,
            "org": org,
            "lang": _lang_for_org(org),
            "content_type": ctype,
            "title": urlparse(url).path.split("/")[-1].replace("-", " ").title() or org,
        }

        if ctype == "html":
            text = _extract_text(body)
            record["text"] = text[:50000]
            record["text_len"] = len(text)
            for link in _extract_links(body, url):
                if link not in visited and link not in queue:
                    queue.append(link)
                    stats["new_links"] += 1
        else:
            record["pdf_path"] = body
            record["text"] = ""
            record["text_len"] = 0

        pid = _page_id(url)
        (PAGES_DIR / f"{pid}.json").write_text(
            json.dumps(record, ensure_ascii=False), encoding="utf-8"
        )
        stats["fetched"] += 1
        time.sleep(CRAWL_DELAY_SECONDS)

    pbar.close()
    _save_json(VISITED_FILE, sorted(visited))
    _save_json(QUEUE_FILE, queue)

    cumulative: Dict[str, Any] = {}
    if STATS_FILE.exists():
        try:
            cumulative = json.loads(STATS_FILE.read_text(encoding="utf-8"))
            if not isinstance(cumulative, dict):
                cumulative = {}
        except (json.JSONDecodeError, OSError):
            cumulative = {}
    cumulative["total_visited"] = len(visited)
    cumulative["queue_remaining"] = len(queue)
    cumulative["last_run"] = stats
    _save_json(STATS_FILE, cumulative)

    console.print(
        f"[green]✓[/green] Fetched {stats['fetched']} | "
        f"Queue: {len(queue)} | Visited: {len(visited)} | "
        f"New links: {stats['new_links']}"
    )
    return stats


def fetch_pubmed_abstracts(max_results: int = 2000) -> int:
    """Bulk-fetch PubMed abstracts (Bulgaria + general medicine terms)."""
    console.print("[bold]PubMed bulk fetch...[/bold]")
    terms = [
        "bulgaria medicine",
        "bulgaria health",
        "clinical guidelines",
        "rare disease",
        "pharmacology",
        "cardiology",
        "oncology",
        "diabetes treatment",
        "hypertension management",
        "infectious disease",
    ]
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    for term in terms:
        per_term = max_results // len(terms)
        try:
            sr = requests.get(
                f"{base}/esearch.fcgi",
                params={"db": "pubmed", "term": term, "retmax": per_term, "retmode": "json"},
                timeout=60,
            )
            sr.raise_for_status()
            ids = sr.json().get("esearchresult", {}).get("idlist", [])
        except (requests.RequestException, json.JSONDecodeError):
            continue

        for i in range(0, len(ids), 100):
            batch_ids = ids[i : i + 100]
            try:
                fr = requests.get(
                    f"{base}/efetch.fcgi",
                    params={"db": "pubmed", "id": ",".join(batch_ids), "retmode": "xml"},
                    timeout=90,
                )
                fr.raise_for_status()
                xml = fr.text
            except requests.RequestException:
                continue

            for pmid in batch_ids:
                chunk = xml
                if f'<PMID Version="1">{pmid}</PMID>' in xml:
                    start = xml.find(f'<PMID Version="1">{pmid}</PMID>')
                    end = xml.find("</PubmedArticle>", start)
                    if end > start:
                        chunk = xml[start : end + 16]

                title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", chunk, re.DOTALL)
                abs_m = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", chunk, re.DOTALL)
                title = re.sub(r"<[^>]+>", "", title_m.group(1)) if title_m else f"PubMed {pmid}"
                abstract = re.sub(r"<[^>]+>", " ", abs_m.group(1)) if abs_m else ""
                text = f"{title}. {abstract}".strip()
                if len(text) < 80:
                    continue

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                pid = f"pubmed_{pmid}"
                record = {
                    "url": url,
                    "org": "PubMed",
                    "lang": "en",
                    "content_type": "pubmed",
                    "title": title[:200],
                    "text": text[:8000],
                    "text_len": len(text),
                }
                (PAGES_DIR / f"{pid}.json").write_text(
                    json.dumps(record, ensure_ascii=False), encoding="utf-8"
                )
                count += 1
            time.sleep(0.4)

    console.print(f"[green]✓[/green] PubMed abstracts: {count}")
    return count


if __name__ == "__main__":
    try:
        pages = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PAGES_PER_RUN
        crawl_medical_web(max_pages=pages)
        if "--pubmed" in sys.argv or len(sys.argv) == 1:
            fetch_pubmed_abstracts(max_results=3000)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted — state saved.[/yellow]")
        sys.exit(1)
