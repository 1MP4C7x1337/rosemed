"""Retrieval-augmented generation over WHO and Bulgarian knowledge base."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import get_config


class KnowledgeRetriever:
    """Simple BM25-style retriever over ingested medical chunks."""

    def __init__(self, chunks_path: Optional[Path] = None) -> None:
        cfg = get_config()
        self.chunks_path = chunks_path or Path("./data/knowledge_base/chunks.jsonl")
        self.chunks: List[Dict[str, Any]] = []
        self._loaded = False
        self._avg_doc_len = 1.0
        self._doc_freq: Dict[str, int] = {}
        self._doc_tokens: List[List[str]] = []

    @property
    def is_available(self) -> bool:
        """Return True if knowledge base file exists and has chunks."""
        return self.chunks_path.exists()

    def load(self) -> int:
        """Load chunks from disk. Returns number of chunks loaded."""
        self.chunks = []
        self._doc_tokens = []
        self._doc_freq = {}

        if not self.chunks_path.exists():
            self._loaded = True
            return 0

        with self.chunks_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.chunks.append(json.loads(line))

        self._doc_tokens = [self._tokenize(c["text"]) for c in self.chunks]
        total_len = 0
        for tokens in self._doc_tokens:
            total_len += len(tokens)
            seen = set(tokens)
            for tok in seen:
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1

        self._avg_doc_len = total_len / max(len(self._doc_tokens), 1)
        self._loaded = True
        return len(self.chunks)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for retrieval scoring."""
        text = text.lower()
        text = re.sub(r"[^\w\s\u0400-\u04FF-]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def _bm25_score(self, query_tokens: List[str], doc_idx: int, k1: float = 1.5, b: float = 0.75) -> float:
        """Compute BM25 score for a document."""
        doc_tokens = self._doc_tokens[doc_idx]
        if not doc_tokens:
            return 0.0
        doc_len = len(doc_tokens)
        n_docs = len(self._doc_tokens)
        tf_map: Dict[str, int] = {}
        for tok in doc_tokens:
            tf_map[tok] = tf_map.get(tok, 0) + 1

        score = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            df = self._doc_freq.get(term, 0)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            tf = tf_map[term]
            denom = tf + k1 * (1 - b + b * doc_len / self._avg_doc_len)
            score += idf * (tf * (k1 + 1)) / denom
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant chunks for a query."""
        if not self._loaded:
            self.load()
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored: List[Tuple[float, int]] = []
        for idx in range(len(self.chunks)):
            score = self._bm25_score(query_tokens, idx)
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, idx in scored[:top_k]:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(score, 4)
            results.append(chunk)
        return results

    def build_context(self, query: str, top_k: int = 5) -> str:
        """Build a context block for injection into the LLM prompt."""
        hits = self.retrieve(query, top_k=top_k)
        if not hits:
            return ""

        parts = [
            "=== Проверени медицински източници (WHO / България) ===",
            "Използвайте САМО тази информация за факти. Отговорете на български.",
        ]
        for i, hit in enumerate(hits, 1):
            parts.append(
                f"\n[Източник {i}: {hit['org']} — {hit['title']}]\n{hit['text'][:900]}"
            )
        parts.append("\n=== Край на източниците ===\n")
        return "\n".join(parts)

    def format_citations(self, hits: List[Dict[str, Any]]) -> str:
        """Format source citations for API responses."""
        if not hits:
            return ""
        lines = ["\n📚 Източници:"]
        for hit in hits:
            url = hit.get("url", "")
            line = f"- {hit['org']}: {hit['title']}"
            if url.startswith("http"):
                line += f" ({url})"
            lines.append(line)
        return "\n".join(lines)
