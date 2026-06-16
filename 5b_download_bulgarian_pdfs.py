"""Download Bulgarian public medical PDFs into data/sources/raw/ for full ingest."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from tqdm import tqdm

console = Console()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RoseMed/1.0; medical-research)",
}

# Seed pages that link to PDFs (MoH, NHIF, NCPHP)
CRAWL_SEEDS = [
    "https://www.mh.government.bg/bg/politiki/",
    "https://www.mh.government.bg/bg/deynosti/imunizacii/",
    "https://www.mh.government.bg/bg/deynosti/speshna-pomosht/",
    "https://ncphp.government.bg/",
    "https://www.nhif.bg/bg/",
    "https://www.nhif.bg/bg/medicines",
]

OUTPUT_DIR = Path("./data/sources/raw/bulgarian_pdfs")


def _fetch(url: str, timeout: int = 45) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text
    except requests.RequestException as exc:
        console.print(f"[yellow]Skip[/yellow] {url}: {exc}")
        return None


def _find_pdf_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        full = urljoin(base_url, href)
        if full.lower().endswith(".pdf") or ".pdf?" in full.lower():
            links.add(full.split("?")[0])
    return links


def _download_pdf(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)
        return dest.stat().st_size > 500
    except requests.RequestException as exc:
        console.print(f"[yellow]PDF fail[/yellow] {url}: {exc}")
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def download_bulgarian_pdfs(max_pdfs: int = 200) -> None:
    """Crawl Bulgarian health sites and download linked PDFs."""
    console.print("[bold cyan]RoseMed: Download Bulgarian medical PDFs[/bold cyan]\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_urls: set[str] = set()
    for seed in CRAWL_SEEDS:
        html = _fetch(seed)
        if html:
            pdf_urls.update(_find_pdf_links(html, seed))
        time.sleep(0.5)

    console.print(f"Found {len(pdf_urls)} PDF links")
    downloaded = 0

    for url in tqdm(list(pdf_urls)[:max_pdfs], desc="PDFs"):
        name = urlparse(url).path.split("/")[-1] or "document.pdf"
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)[:120]
        dest = OUTPUT_DIR / safe
        if _download_pdf(url, dest):
            downloaded += 1
        time.sleep(0.3)

    console.print(f"\n[green]✓[/green] Downloaded {downloaded} PDFs → {OUTPUT_DIR}")
    console.print("Run: python3 5_ingest_knowledge_base.py")


if __name__ == "__main__":
    try:
        limit = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        download_bulgarian_pdfs(max_pdfs=limit)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
