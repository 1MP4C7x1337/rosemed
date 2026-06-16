"""RoseMed Step 5: Ingest WHO and Bulgarian medical sources into knowledge base."""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import track
from tqdm import tqdm

from medical_sources import (
    BULGARIAN_URL_SOURCES,
    RAW_SOURCES_DIR_NAME,
    SUPPORTED_RAW_EXTENSIONS,
    WHO_URL_SOURCES,
)

console = Console()
HEADER = """
══════ RoseMed Step 5: Ingest Medical Knowledge ══════
"""

REQUEST_HEADERS = {
    "User-Agent": "RoseMed-27B-BG/1.0 (medical knowledge ingestion; research)",
    "Accept": "text/html,application/xhtml+xml,application/json",
}


def _clean_text(text: str) -> str:
    """Normalize whitespace in extracted text."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks by character count."""
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _fetch_url(url: str, timeout: int = 30) -> Optional[str]:
    """Fetch URL and return HTML or text content."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        console.print(f"[yellow]WARNING:[/yellow] Failed to fetch {url}: {exc}")
        return None


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return _clean_text(soup.get_text(separator=" "))
    return _clean_text(main.get_text(separator=" "))


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return _clean_text(" ".join(pages))
    except ImportError:
        console.print(
            "[yellow]WARNING:[/yellow] pypdf not installed — skipping PDF. "
            "Run: pip install pypdf"
        )
        return ""
    except Exception as exc:
        console.print(f"[yellow]WARNING:[/yellow] PDF read failed {path.name}: {exc}")
        return ""


def _read_local_file(path: Path) -> str:
    """Read text content from a local source file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".json", ".jsonl"}:
        try:
            if suffix == ".jsonl":
                lines = []
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        obj = json.loads(line)
                        lines.append(json.dumps(obj, ensure_ascii=False))
                return "\n".join(lines)
            return path.read_text(encoding="utf-8")
        except (OSError, json.JSONDecodeError) as exc:
            console.print(f"[yellow]WARNING:[/yellow] JSON read failed {path.name}: {exc}")
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        console.print(f"[yellow]WARNING:[/yellow] File read failed {path.name}: {exc}")
        return ""


def _make_chunk_records(
    text: str,
    source_id: str,
    title: str,
    org: str,
    lang: str,
    url: str = "",
) -> List[Dict[str, Any]]:
    """Create knowledge-base chunk records from text."""
    records: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(_chunk_text(text)):
        if len(chunk) < 80:
            continue
        records.append(
            {
                "id": f"{source_id}_chunk_{idx}",
                "source_id": source_id,
                "title": title,
                "org": org,
                "lang": lang,
                "url": url,
                "text": chunk,
            }
        )
    return records


def _ingest_url_source(source: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetch and chunk a single URL source."""
    html = _fetch_url(source["url"])
    if not html:
        return []
    text = _html_to_text(html)
    if len(text) < 100:
        return []
    return _make_chunk_records(
        text=text,
        source_id=source["id"],
        title=source["title"],
        org=source["org"],
        lang=source["lang"],
        url=source["url"],
    )


def _ingest_local_sources(raw_dir: Path) -> List[Dict[str, Any]]:
    """Ingest all supported files from data/sources/raw/."""
    records: List[Dict[str, Any]] = []
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        return records

    files = [
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_RAW_EXTENSIONS
    ]

    for path in track(files, description="Local files"):
        text = _read_local_file(path)
        if len(text) < 80:
            continue
        source_id = f"local_{path.stem.lower().replace(' ', '_')}"
        org = "Bulgaria Local" if any(
            x in path.name.lower() for x in ("nzok", "nhif", "bda", "moh", "bg")
        ) else "Local Source"
        lang = "bg" if org == "Bulgaria Local" else "en"
        records.extend(
            _make_chunk_records(
                text=text,
                source_id=source_id,
                title=path.name,
                org=org,
                lang=lang,
                url=str(path),
            )
        )
    return records


def _generate_qa_from_chunks(chunks: List[Dict[str, Any]], max_samples: int = 500) -> List[Dict[str, Any]]:
    """Generate supplementary Q&A training samples from knowledge chunks."""
    samples: List[Dict[str, Any]] = []
    for chunk in chunks[:max_samples]:
        title = chunk["title"]
        org = chunk["org"]
        text = chunk["text"][:1200]
        instruction = f"Какво казва {org} относно {title}?"
        output = (
            f"Според {org} ({title}):\n\n{text}\n\n"
            "Препоръчва се консултация с лекар за индивидуална оценка."
        )
        samples.append(
            {
                "instruction": instruction,
                "input": "",
                "output": output,
                "category": f"Източник: {org}",
                "language": "bg" if chunk["lang"] == "bg" else "bg",
                "source_id": chunk["source_id"],
            }
        )
    return samples


def ingest_knowledge_base(
    output_dir: Path = Path("./data/knowledge_base"),
    raw_dir: Path = Path("./data/sources/raw"),
    generate_training: bool = True,
) -> None:
    """Download WHO/Bulgarian sources and build the RoseMed knowledge base."""
    console.print(HEADER, style="bold cyan")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []

    console.print("[bold]Fetching WHO sources...[/bold]")
    for source in tqdm(WHO_URL_SOURCES, desc="WHO"):
        chunks = _ingest_url_source(source)
        all_chunks.extend(chunks)
        time.sleep(0.5)

    console.print("[bold]Fetching Bulgarian public sources...[/bold]")
    for source in tqdm(BULGARIAN_URL_SOURCES, desc="Bulgaria"):
        chunks = _ingest_url_source(source)
        all_chunks.extend(chunks)
        time.sleep(0.5)

    console.print("[bold]Ingesting local files from data/sources/raw/...[/bold]")
    all_chunks.extend(_ingest_local_sources(raw_dir))

    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    by_org: Dict[str, int] = {}
    for c in all_chunks:
        by_org[c["org"]] = by_org.get(c["org"], 0) + 1

    manifest = {
        "total_chunks": len(all_chunks),
        "by_org": by_org,
        "who_sources": len(WHO_URL_SOURCES),
        "bg_sources": len(BULGARIAN_URL_SOURCES),
        "raw_dir": str(raw_dir),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print(f"\n[green]✓[/green] Knowledge base saved: {chunks_path}")
    console.print(f"  Total chunks: {len(all_chunks)}")
    for org, count in sorted(by_org.items()):
        console.print(f"  {org}: {count}")

    if generate_training and all_chunks:
        qa_samples = _generate_qa_from_chunks(all_chunks)
        qa_path = Path("./data/rosemed_sources_train.jsonl")
        with qa_path.open("w", encoding="utf-8") as fh:
            for sample in qa_samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
        console.print(f"\n[green]✓[/green] Supplementary training data: {qa_path} ({len(qa_samples)} samples)")
        console.print(
            "  Merge into fine-tune: cat data/rosemed_sources_train.jsonl >> data/rosemed_train.jsonl"
        )

    console.print(f"\n[bold]Add more data:[/bold] drop PDF/HTML/TXT files into {raw_dir}")
    console.print("  Examples: NZOK drug lists, MoH guidelines, WHO PDFs, Orphanet exports")
    console.print("\nNext: restart API server — RAG will use the knowledge base automatically.")


if __name__ == "__main__":
    try:
        ingest_knowledge_base()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
