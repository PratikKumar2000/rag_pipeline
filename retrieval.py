"""Hybrid retrieval: FAISS semantic search plus SQLite FTS5 keyword search, fused and reranked."""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort

# Windows without Developer Mode cannot symlink; the cache falls back to copies, which is fine.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from llm import embed

DENSE_K = 50
LEXICAL_K = 50
RRF_K = 60
RERANK_K = 50
FINAL_K = 5
RERANKER_REPO = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANKER_MAX_TOKENS = 512
# Arctic Embed retrieves well only when queries (not documents) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = set(
    "a an and are as at be by can do does for from how i in is it of on or that the this to "
    "was what when where which who why will with you your me my we our there their".split()
)


@dataclass
class Match:
    id: int
    source: str
    location: str
    chunk: str
    text: str
    rerank_score: float = 0.0
    found_by: list[str] = field(default_factory=list)


class Reranker:
    """Cross-encoder that reads the question and each excerpt together, on the CPU."""

    def __init__(self, cache_dir: Path) -> None:
        model_path = hf_hub_download(RERANKER_REPO, "onnx/model.onnx", cache_dir=cache_dir)
        tokenizer_path = hf_hub_download(RERANKER_REPO, "tokenizer.json", cache_dir=cache_dir)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(RERANKER_MAX_TOKENS, strategy="only_second")
        self.tokenizer.enable_padding()
        options = ort.SessionOptions()
        # Leave CPU cores free for indexing and the answer model.
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def score(self, query: str, texts: list[str], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch([(query, text) for text in texts[start : start + batch_size]])
            inputs = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
            }
            logits = self.session.run(None, {k: v for k, v in inputs.items() if k in self.input_names})[0]
            scores.extend(float(value) for value in logits[:, 0])
        return scores


def _index_version(index_dir: Path) -> str:
    progress = index_dir / "progress.json"
    if not progress.exists():
        return ""
    return json.loads(progress.read_text(encoding="utf-8")).get("config_hash", "")


def sync_lexical_index(index_dir: Path) -> Path:
    """Bring the keyword index up to date with the chunk metadata, adding only new rows."""
    lexical_path = index_dir / "lexical.sqlite3"
    version = _index_version(index_dir)
    with sqlite3.connect(lexical_path) as lexical:
        lexical.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        stored = lexical.execute("SELECT value FROM state WHERE key = 'version'").fetchone()
        if stored is None or stored[0] != version:
            lexical.execute("DROP TABLE IF EXISTS chunks")
            lexical.execute("DELETE FROM state")
        # Contentless table: the text stays in metadata.sqlite3, only the search index lives here.
        lexical.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "text, source, content='', tokenize='porter unicode61')"
        )
        lexical.execute("INSERT OR REPLACE INTO state VALUES ('version', ?)", (version,))
        row = lexical.execute("SELECT value FROM state WHERE key = 'last_id'").fetchone()
        last_id = int(row[0]) if row else -1
        with sqlite3.connect(f"file:{index_dir / 'metadata.sqlite3'}?mode=ro", uri=True) as metadata:
            while True:
                rows = metadata.execute(
                    "SELECT id, text, source FROM records WHERE id > ? ORDER BY id LIMIT 5000",
                    (last_id,),
                ).fetchall()
                if not rows:
                    break
                # Index CSV fields and path segments as separate words.
                lexical.executemany(
                    "INSERT INTO chunks (rowid, text, source) VALUES (?, ?, ?)",
                    [(i, text, re.sub(r"[\\/_.]", " ", source)) for i, text, source in rows],
                )
                last_id = rows[-1][0]
                lexical.execute("INSERT OR REPLACE INTO state VALUES ('last_id', ?)", (str(last_id),))
                lexical.commit()
    return lexical_path


def _fts_query(question: str) -> str:
    terms = dict.fromkeys(
        term for term in re.findall(r"\w+", question.lower()) if len(term) > 1 and term not in STOPWORDS
    )
    return " OR ".join(f'"{term}"' for term in terms)


def lexical_search(lexical_path: Path, question: str, k: int = LEXICAL_K) -> list[int]:
    query = _fts_query(question)
    if not query:
        return []
    with sqlite3.connect(lexical_path) as lexical:
        rows = lexical.execute(
            "SELECT rowid FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
            (query, k),
        ).fetchall()
    return [row[0] for row in rows]


def dense_search(
    index: faiss.Index, question: str, model: str, base_url: str, k: int = DENSE_K
) -> list[int]:
    if index.ntotal == 0:
        return []
    vector = np.asarray(embed([QUERY_PREFIX + question], model, base_url), dtype="float32")
    faiss.normalize_L2(vector)
    _, positions = index.search(vector, min(k, index.ntotal))
    return [int(position) for position in positions[0] if position >= 0]


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = RRF_K) -> list[tuple[int, list[str]]]:
    scores: dict[int, float] = {}
    found_by: dict[int, list[str]] = {}
    for name, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            found_by.setdefault(chunk_id, []).append(name)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [(chunk_id, found_by[chunk_id]) for chunk_id in ordered]


def hybrid_search(
    question: str,
    index: faiss.Index,
    index_dir: Path,
    lexical_path: Path,
    reranker: Reranker,
    model: str,
    base_url: str,
    final_k: int = FINAL_K,
) -> list[Match]:
    rankings = {
        "semantic": dense_search(index, question, model, base_url),
        "keyword": lexical_search(lexical_path, question),
    }
    candidates = reciprocal_rank_fusion(rankings)[:RERANK_K]
    if not candidates:
        return []
    with sqlite3.connect(f"file:{index_dir / 'metadata.sqlite3'}?mode=ro", uri=True) as metadata:
        matches = []
        seen_texts = set()
        for chunk_id, found_by in candidates:
            row = metadata.execute(
                "SELECT source, location, chunk, text FROM records WHERE id = ?", (chunk_id,)
            ).fetchone()
            if not row:
                continue
            # Versioned bundles repeat the same documents; keep one copy of each excerpt.
            key = " ".join(row[3].split())
            if key not in seen_texts:
                seen_texts.add(key)
                matches.append(Match(chunk_id, *row, found_by=found_by))
    for match, score in zip(matches, reranker.score(question, [m.text for m in matches])):
        match.rerank_score = score
    return sorted(matches, key=lambda m: m.rerank_score, reverse=True)[:final_k]
