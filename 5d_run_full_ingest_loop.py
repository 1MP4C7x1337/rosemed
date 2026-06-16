"""RoseMed: one loop iteration — crawl web + PDFs + full ingest."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def run_step(label: str, cmd: list[str]) -> bool:
    console.print(f"\n[bold cyan]▶ {label}[/bold cyan]")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        console.print(f"[yellow]Warning:[/yellow] {label} exited {result.returncode}")
        return False
    return True


def main() -> None:
    py = sys.executable
    console.print("[bold green]RoseMed Full Ingest Loop — iteration start[/bold green]")

    run_step("Crawl medical web (800 pages + PubMed)", [py, "5c_crawl_medical_web.py", "800"])
    run_step("Download Bulgarian PDFs", [py, "5b_download_bulgarian_pdfs.py", "150"])
    run_step("Full knowledge base ingest", [py, "5_ingest_knowledge_base.py"])

    manifest = Path("data/knowledge_base/manifest.json")
    if manifest.exists():
        import json
        m = json.loads(manifest.read_text(encoding="utf-8"))
        console.print(f"\n[bold green]Done.[/bold green] Total chunks: {m.get('total_chunks', '?')}")


if __name__ == "__main__":
    main()
