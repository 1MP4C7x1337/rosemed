"""RoseMed Step 5: Full WHO + Bulgarian medical knowledge ingestion (no compromises)."""

from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from medical_sources import (
    BULK_DOWNLOAD_SOURCES,
    BULGARIAN_URL_SOURCES,
    SUPPORTED_RAW_EXTENSIONS,
    build_who_sources,
)

console = Console()
HEADER = """
══════ RoseMed Step 5: Full Medical Knowledge Ingest ══════
"""

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RoseMed/1.0; medical-research)",
    "Accept": "text/html,application/xhtml+xml,application/json,text/xml",
    "Accept-Language": "en,bg;q=0.9",
}


def _clean_text(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text else []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _fetch_url(url: str, timeout: int = 45) -> Optional[str]:
    """Fetch URL content."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text
    except requests.RequestException:
        return None


def _discover_who_fact_sheets() -> List[Dict[str, str]]:
    """Crawl WHO fact sheets index for additional pages."""
    index_urls = [
        "https://www.who.int/news-room/fact-sheets",
        "https://www.who.int/news-room/fact-sheets?health_topic=Noncommunicable+diseases",
        "https://www.who.int/news-room/fact-sheets?health_topic=Communicable+diseases",
    ]
    discovered: List[Dict[str, str]] = []
    seen: Set[str] = set()

    for index_url in index_urls:
        html = _fetch_url(index_url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full = urljoin("https://www.who.int", href)
            if "/fact-sheets/detail/" not in full or full in seen:
                continue
            seen.add(full)
            slug = full.rstrip("/").split("/")[-1]
            title = link.get_text(strip=True) or slug.replace("-", " ").title()
            discovered.append({
                "id": f"who_disc_{slug[:35]}",
                "title": f"WHO: {title}",
                "url": full,
                "org": "WHO",
                "lang": "en",
            })
        time.sleep(0.3)

    return discovered


def _html_to_text(html: str) -> str:
    """Extract main text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return _clean_text(soup.get_text(separator=" "))
    return _clean_text(main.get_text(separator=" "))


def _orphanet_disorder_texts(xml_content: str) -> List[str]:
    """Extract one text block per Orphanet disorder (full dataset, no cap)."""
    texts: List[str] = []
    try:
        root = ET.fromstring(xml_content.encode("utf-8", errors="ignore"))
    except ET.ParseError:
        return [_clean_text(re.sub(r"<[^>]+>", " ", xml_content))]

    for node in root.iter():
        tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
        if tag.lower() != "disorder":
            continue
        parts: List[str] = []
        for child in node.iter():
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            text = (child.text or "").strip()
            if text and ctag.lower() in (
                "name", "orpha_code", "disorder_type", "definition", "synonym",
            ):
                parts.append(f"{ctag}: {text}")
        if parts:
            texts.append(" | ".join(parts))
    return texts


def _xml_to_text(xml_content: str) -> str:
    """Flatten generic XML to text."""
    texts = _orphanet_disorder_texts(xml_content)
    if texts:
        return "\n".join(texts)
    return _clean_text(re.sub(r"<[^>]+>", " ", xml_content))


def _read_pdf(path: Path) -> str:
    """Extract PDF text."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return _clean_text(" ".join(page.extract_text() or "" for page in reader.pages))
    except Exception:
        return ""


def _read_local_file(path: Path) -> str:
    """Read local source file."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".xml":
        try:
            return _xml_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            return ""
    if suffix == ".jsonl":
        lines = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    lines.append(json.dumps(json.loads(line), ensure_ascii=False))
            return "\n".join(lines)
        except (OSError, json.JSONDecodeError):
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _make_chunks(
    text: str, source_id: str, title: str, org: str, lang: str, url: str = ""
) -> List[Dict[str, Any]]:
    """Build chunk records."""
    out: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(_chunk_text(text)):
        if len(chunk) < 60:
            continue
        out.append({
            "id": f"{source_id}_c{idx}",
            "source_id": source_id,
            "title": title,
            "org": org,
            "lang": lang,
            "url": url,
            "text": chunk,
        })
    return out


def _ingest_url(source: Dict[str, str]) -> List[Dict[str, Any]]:
    """Fetch and chunk one URL."""
    content = _fetch_url(source["url"])
    if not content:
        return []
    if source.get("type") == "xml" or source["url"].endswith(".xml"):
        text = _xml_to_text(content)
    else:
        text = _html_to_text(content)
    if len(text) < 80:
        return []
    return _make_chunks(text, source["id"], source["title"], source["org"], source["lang"], source["url"])


def _ingest_bulk_sources(cache_dir: Path) -> List[Dict[str, Any]]:
    """Download and ingest large bulk datasets (Orphanet XML, etc.)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []

    for source in tqdm(BULK_DOWNLOAD_SOURCES, desc="Bulk datasets"):
        cache_file = cache_dir / f"{source['id']}.cache"
        content = _fetch_url(source["url"])
        if not content:
            if cache_file.exists():
                content = cache_file.read_text(encoding="utf-8", errors="ignore")
            else:
                continue
        else:
            cache_file.write_text(content, encoding="utf-8", errors="ignore")

        if source.get("type") == "xml" or source["url"].endswith(".xml"):
            disorders = _orphanet_disorder_texts(content)
            for idx, block in enumerate(disorders):
                records.extend(_make_chunks(
                    block,
                    f"{source['id']}_{idx}",
                    f"{source['title']} #{idx + 1}",
                    source["org"],
                    source["lang"],
                    source["url"],
                ))
            continue

        text = _html_to_text(content)
        records.extend(_make_chunks(
            text, source["id"], source["title"], source["org"], source["lang"], source["url"]
        ))
        time.sleep(1)

    return records


def _ingest_crawl_cache(crawl_dir: Path) -> List[Dict[str, Any]]:
    """Ingest pages saved by 5c_crawl_medical_web.py."""
    pages_dir = crawl_dir / "pages"
    pdfs_dir = crawl_dir / "pdfs"
    records: List[Dict[str, Any]] = []

    if pages_dir.exists():
        files = list(pages_dir.glob("*.json"))
        for path in tqdm(files, desc="Crawl cache"):
            try:
                page = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            text = page.get("text", "")
            if len(text) < 80 and page.get("content_type") == "pdf":
                pdf_path = page.get("pdf_path")
                if pdf_path and Path(pdf_path).exists():
                    text = _read_pdf(Path(pdf_path))
            if len(text) < 80:
                continue
            sid = f"crawl_{path.stem[:24]}"
            records.extend(_make_chunks(
                text,
                sid,
                page.get("title", "Web page")[:120],
                page.get("org", "Medical Web"),
                page.get("lang", "en"),
                page.get("url", ""),
            ))

    if pdfs_dir.exists():
        for pdf in tqdm(list(pdfs_dir.glob("*.pdf")), desc="Crawl PDFs"):
            text = _read_pdf(pdf)
            if len(text) < 80:
                continue
            sid = f"crawl_pdf_{pdf.stem[:20]}"
            records.extend(_make_chunks(text, sid, pdf.name, "Medical Web PDF", "en", str(pdf)))

    return records


def _ingest_local(raw_dir: Path) -> List[Dict[str, Any]]:
    """Ingest all files from data/sources/raw/."""
    records: List[Dict[str, Any]] = []
    if not raw_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        return records

    files = [p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_RAW_EXTENSIONS]
    for path in tqdm(files, desc="Local files"):
        text = _read_local_file(path)
        if len(text) < 80:
            continue
        org = "Bulgaria Local"
        if not any(x in path.name.lower() for x in ("nzok", "nhif", "bda", "moh", "bg", "bulgar")):
            org = "Local Source"
        lang = "bg" if org == "Bulgaria Local" else "en"
        sid = f"local_{path.stem.lower().replace(' ', '_')[:30]}"
        records.extend(_make_chunks(text, sid, path.name, org, lang, str(path)))
    return records


def _generate_qa_all(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate Q&A from ALL chunks (no limit)."""
    samples: List[Dict[str, Any]] = []
    for chunk in chunks:
        org = chunk["org"]
        title = chunk["title"]
        text = chunk["text"][:1500]
        samples.append({
            "instruction": f"Предоставете медицинска информация от {org} относно: {title}",
            "input": "",
            "output": (
                f"Източник: {org} — {title}\n\n{text}\n\n"
                "Консултирайте се с лекар за индивидуална оценка."
            ),
            "category": f"Източник: {org}",
            "language": "bg",
            "source_id": chunk["source_id"],
        })
    return samples


def ingest_knowledge_base(
    output_dir: Path = Path("./data/knowledge_base"),
    raw_dir: Path = Path("./data/sources/raw"),
    bulk_dir: Path = Path("./data/sources/bulk"),
    crawl_dir: Path = Path("./data/sources/crawl"),
) -> None:
    """Full ingestion — WHO, Orphanet, Bulgaria, local files."""
    console.print(HEADER, style="bold cyan")

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []

    who_sources = build_who_sources()
    console.print(f"[bold]WHO catalog:[/bold] {len(who_sources)} URLs")

    console.print("[bold]Discovering extra WHO fact sheets...[/bold]")
    discovered = _discover_who_fact_sheets()
    console.print(f"  Discovered: {len(discovered)} additional pages")

    combined_who = {s["url"]: s for s in who_sources}
    for s in discovered:
        combined_who.setdefault(s["url"], s)

    console.print(f"[bold]Fetching {len(combined_who)} WHO pages...[/bold]")
    for source in tqdm(list(combined_who.values()), desc="WHO"):
        all_chunks.extend(_ingest_url(source))
        time.sleep(0.25)

    console.print(f"[bold]Fetching {len(BULGARIAN_URL_SOURCES)} Bulgarian sources...[/bold]")
    for source in tqdm(BULGARIAN_URL_SOURCES, desc="Bulgaria"):
        all_chunks.extend(_ingest_url(source))
        time.sleep(0.25)

    console.print("[bold]Downloading bulk datasets (Orphanet rare diseases)...[/bold]")
    all_chunks.extend(_ingest_bulk_sources(bulk_dir))

    console.print("[bold]Ingesting web crawl cache...[/bold]")
    all_chunks.extend(_ingest_crawl_cache(crawl_dir))

    console.print("[bold]Ingesting local files...[/bold]")
    all_chunks.extend(_ingest_local(raw_dir))

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
        "who_urls_attempted": len(combined_who),
        "bg_urls": len(BULGARIAN_URL_SOURCES),
        "bulk_sources": len(BULK_DOWNLOAD_SOURCES),
        "crawl_pages": len(list((crawl_dir / "pages").glob("*.json"))) if (crawl_dir / "pages").exists() else 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    table = Table(title="RoseMed Knowledge Base — Full Ingest")
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", justify="right", style="green")
    for org, count in sorted(by_org.items(), key=lambda x: -x[1]):
        table.add_row(org, str(count))
    table.add_row("─" * 25, "─" * 8)
    table.add_row("TOTAL", str(len(all_chunks)), style="bold green")
    console.print(table)

    if all_chunks:
        qa = _generate_qa_all(all_chunks)
        qa_path = Path("./data/rosemed_sources_train.jsonl")
        with qa_path.open("w", encoding="utf-8") as fh:
            for s in qa:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

        merged_path = Path("./data/rosemed_train_full.jsonl")
        base_train = Path("./data/rosemed_train.jsonl")
        with merged_path.open("w", encoding="utf-8") as out:
            if base_train.exists():
                out.write(base_train.read_text(encoding="utf-8"))
            for s in qa:
                out.write(json.dumps(s, ensure_ascii=False) + "\n")

        console.print(f"\n[green]✓[/green] {len(all_chunks)} chunks → {chunks_path}")
        console.print(f"[green]✓[/green] {len(qa)} source Q&A → {qa_path}")
        console.print(f"[green]✓[/green] Merged train set → {merged_path}")
        console.print("\n[bold]For v2 fine-tune after current run:[/bold]")
        console.print("  cp data/rosemed_train_full.jsonl data/rosemed_train.jsonl")
        console.print("  python3 3_finetune.py")

    console.print(f"\n[bold]Add MORE:[/bold] drop PDFs into {raw_dir} and re-run this script.")
    console.print("  NZOK lists, MoH PDFs, BDA exports, WHO PDFs, hospital protocols")


if __name__ == "__main__":
    try:
        ingest_knowledge_base()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
