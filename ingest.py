"""Extract, chunk, embed, and persist STOV-ATOLL source documents."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator

import faiss
import numpy as np
from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
from pptx import Presentation

from llm import embed, split_to_context

SUPPORTED = {".pdf", ".docx", ".xlsx", ".xlsm", ".ppt", ".pptx", ".txt", ".md", ".csv"}


def _text_pages(path: Path) -> Iterator[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        for number, page in enumerate(PdfReader(str(path)).pages, start=1):
            yield f"page {number}", page.extract_text() or ""
    elif suffix == ".docx":
        document = Document(str(path))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text.strip() for cell in row.cells))
        yield "document", "\n".join(paragraphs)
    elif suffix in {".xlsx", ".xlsm"}:
        workbook = load_workbook(str(path), read_only=True, data_only=True)
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value is not None and str(value).strip()]
                if values:
                    rows.append(" | ".join(values))
            yield f"sheet {sheet.title}", "\n".join(rows)
    elif suffix == ".pptx":
        presentation = Presentation(str(path))
        for number, slide in enumerate(presentation.slides, start=1):
            text = "\n".join(
                shape.text for shape in slide.shapes if hasattr(shape, "text") and shape.text.strip()
            )
            yield f"slide {number}", text
    elif suffix == ".ppt":
        office = shutil.which("soffice") or shutil.which("libreoffice")
        if office is None:
            raise RuntimeError(
                "Legacy .ppt requires LibreOffice (soffice) installed and available on PATH."
            )
        with tempfile.TemporaryDirectory(prefix="stov_ppt_") as temporary_dir:
            subprocess.run(
                [
                    office,
                    "--headless",
                    "--convert-to",
                    "pptx",
                    "--outdir",
                    temporary_dir,
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
            converted = Path(temporary_dir) / f"{path.stem}.pptx"
            if not converted.exists():
                raise RuntimeError(f"LibreOffice did not convert legacy PowerPoint file: {path}")
            yield from _text_pages(converted)
    else:
        yield "document", path.read_text(encoding="utf-8", errors="ignore")


def extract_documents(source_roots: list[tuple[Path, str]]) -> Iterator[dict[str, str]]:
    for source_dir, source_prefix in source_roots:
        for path in sorted(source_dir.rglob("*")):
            if (
                not path.is_file()
                or path.name.startswith("~$")
                or path.suffix.lower() not in SUPPORTED
            ):
                continue
            try:
                for location, text in _text_pages(path):
                    clean = re.sub(r"[ \t]+", " ", text).strip()
                    if clean:
                        relative_path = path.relative_to(source_dir)
                        source_name = Path(source_prefix) / relative_path if source_prefix else relative_path
                        yield {
                            "source": str(source_name),
                            "location": location,
                            "text": clean,
                        }
            except Exception as exc:
                print(
                    f"Skipping unavailable or unreadable file: {path} ({exc})",
                    file=sys.stderr,
                )


def _extract_archive(archive: Path, run_dir: Path, number: int) -> Path | None:
    archive_root = run_dir / f"archive_{number:04d}"
    try:
        return _expand_zip(archive, archive_root)
    except (OSError, zipfile.BadZipFile) as exc:
        print(f"Skipping unreadable ZIP file: {archive} ({exc})", file=sys.stderr)
        return None


def prepare_sources(source: Path, extract_dir: Path) -> list[tuple[Path, str]]:
    """Find ZIPs below a folder and expand them into a new run directory."""
    run_dir = extract_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if source.is_file() and source.suffix.lower() == ".zip":
        expanded = _extract_archive(source, run_dir, 1)
        return [(expanded, "archive_0001")] if expanded else []
    if source.is_dir():
        roots: list[tuple[Path, str]] = [(source, "")]
        archives = sorted(
            path
            for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() == ".zip"
        )
        for number, archive in enumerate(archives, start=1):
            expanded = _extract_archive(archive, run_dir, number)
            if expanded:
                roots.append((expanded, f"archive_{number:04d}"))
        if archives:
            print(f"Discovered {len(archives)} ZIP archive(s); expanded under {run_dir}.")
        return roots
    raise FileNotFoundError(f"Source folder or ZIP file does not exist: {source}")


def chunk_text(text: str, size: int = 180, overlap: int = 30, max_chars: int = 1000) -> list[str]:
    # CSV rows, paths and XML have no spaces, so one "word" can be hundreds of tokens.
    text = re.sub(r"([;|])(?=\S)", r"\1 ", text)
    piece = max(1, max_chars // 4)
    words = [
        word[offset : offset + piece]
        for word in text.split()
        for offset in range(0, len(word), piece)
    ]
    chunks = []
    start = 0
    while start < len(words):
        end, length = start, 0
        while end < len(words) and end - start < size:
            if end > start and length + len(words[end]) + 1 > max_chars:
                break
            length += len(words[end]) + 1
            end += 1
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(start + 1, end - min(overlap, (end - start) // 4))
    return chunks


def _iter_batches(
    source_roots: list[tuple[Path, str]],
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    max_chars: int,
) -> Iterator[tuple[int, list[dict[str, str]]]]:
    batch: list[dict[str, str]] = []
    batch_id = 0
    for document in extract_documents(source_roots):
        for chunk_number, chunk in enumerate(
            chunk_text(document["text"], chunk_size, chunk_overlap, max_chars), start=1
        ):
            batch.append(
                {
                    "source": document["source"],
                    "location": document["location"],
                    "chunk": str(chunk_number),
                    "text": chunk,
                }
            )
            if len(batch) >= batch_size:
                yield batch_id, batch
                batch_id += 1
                batch = []
    if batch:
        yield batch_id, batch


def _expand_zip(archive: Path, extract_dir: Path) -> Path:
    """Expand one archive under a short, stable Windows path."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (extract_dir / member.filename).resolve()
            if not target.is_relative_to(root):
                print(f"Skipping unsafe archive entry: {member.filename}", file=sys.stderr)
                continue
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
    return extract_dir


def _replace_with_retry(source: Path, target: Path, attempts: int = 30) -> None:
    """Windows denies renames while scanners or file watchers briefly hold the target."""
    for attempt in range(attempts):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(min(0.2 * (attempt + 1), 2.0))


def _temporary_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.tmp")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = _temporary_path(path)
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _replace_with_retry(temporary, path)


def _open_metadata(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY,
            batch_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            location TEXT NOT NULL,
            chunk TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    connection.commit()
    return connection


def _embed_records(
    records: list[dict[str, str]], model: str, url: str, api: str
) -> tuple[list[dict[str, str]], list[list[float]]]:
    if api == "openai":
        expanded = []
        for record, pieces in zip(records, split_to_context([r["text"] for r in records], url)):
            if len(pieces) == 1:
                expanded.append(record)
                continue
            expanded.extend(
                {**record, "chunk": f"{record['chunk']}.{number}", "text": piece}
                for number, piece in enumerate(pieces, start=1)
            )
        records = expanded
    return records, embed([record["text"] for record in records], model, url, api)


def build_index(
    source_dir: Path,
    index_dir: Path,
    embedding_model: str,
    ollama_url: str,
    chunk_size: int = 180,
    chunk_overlap: int = 30,
    batch_size: int = 2,
    workers: int = 4,
    extract_dir: Path = Path(r"C:\rag_source"),
    fresh: bool = False,
    checkpoint_seconds: float = 60.0,
    embed_api: str = "ollama",
    max_chars: int = 1000,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    if batch_size < 1:
        raise ValueError("batch-size must be at least 1")

    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "stov_atoll.faiss"
    metadata_path = index_dir / "metadata.sqlite3"
    progress_path = index_dir / "progress.json"
    config = {
        "source": str(source_dir),
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_chars": max_chars,
        "batch_size": batch_size,
        "workers": workers,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    checkpoint = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {}
    )
    resume = (
        not fresh
        and checkpoint.get("config_hash") == config_hash
        and index_path.exists()
        and metadata_path.exists()
    )
    if not resume:
        for path in (index_path, metadata_path, progress_path):
            if path.exists():
                path.unlink()
        checkpoint = {"next_batch": 0}

    source_roots = prepare_sources(source_dir, extract_dir)
    connection = _open_metadata(metadata_path)
    index = faiss.read_index(str(index_path)) if resume else None
    database_count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    index_count = index.ntotal if index is not None else 0
    if database_count > index_count:
        connection.execute("DELETE FROM records WHERE id >= ?", (index_count,))
        connection.commit()
        database_count = index_count
    if database_count < index_count:
        connection.close()
        raise RuntimeError("FAISS and metadata checkpoint are inconsistent; use --fresh.")

    next_batch = int(
        connection.execute(
            "SELECT COALESCE(MAX(batch_id) + 1, 0) FROM records"
        ).fetchone()[0]
    )
    batches = _iter_batches(source_roots, chunk_size, chunk_overlap, batch_size, max_chars)
    pending: dict[int, Any] = {}
    processed = 0
    last_checkpoint = time.monotonic()

    def save_checkpoint() -> None:
        if index is None:
            return
        temporary_index = _temporary_path(index_path)
        faiss.write_index(index, str(temporary_index))
        _replace_with_retry(temporary_index, index_path)
        _write_json_atomic(
            progress_path,
            {
                "config": config,
                "config_hash": config_hash,
                "next_batch": next_batch,
                "chunks": index.ntotal,
            },
        )

    def submit_more(executor: ThreadPoolExecutor) -> None:
        while len(pending) < workers:
            try:
                batch_id, records = next(batches)
            except StopIteration:
                return
            if batch_id < next_batch:
                continue
            pending[batch_id] = executor.submit(
                _embed_records, records, embedding_model, ollama_url, embed_api
            )

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            submit_more(executor)
            while pending:
                if next_batch not in pending:
                    raise RuntimeError(f"Missing batch {next_batch} in resume queue.")
                records, vectors = pending.pop(next_batch).result()
                matrix = np.asarray(vectors, dtype="float32")
                faiss.normalize_L2(matrix)
                if index is None:
                    index = faiss.IndexFlatIP(matrix.shape[1])
                start_id = index.ntotal
                connection.executemany(
                    "INSERT INTO records (id, batch_id, source, location, chunk, text) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (
                            start_id + offset,
                            next_batch,
                            record["source"],
                            record["location"],
                            record["chunk"],
                            record["text"],
                        )
                        for offset, record in enumerate(records)
                    ],
                )
                connection.commit()
                index.add(matrix)
                next_batch += 1
                processed += len(records)
                if time.monotonic() - last_checkpoint >= checkpoint_seconds:
                    save_checkpoint()
                    last_checkpoint = time.monotonic()
                    print(f"Checkpoint: {index.ntotal} chunks indexed.", flush=True)
                submit_more(executor)
    finally:
        connection.close()
        try:
            save_checkpoint()
        except OSError as exc:
            print(f"Final checkpoint failed; rerun to resume: {exc}", file=sys.stderr)

    total_chunks = index.ntotal if index is not None else 0
    connection = sqlite3.connect(metadata_path)
    document_count = connection.execute(
        "SELECT COUNT(DISTINCT source) FROM records"
    ).fetchone()[0]
    connection.close()
    return {
        "documents": document_count,
        "chunks": total_chunks,
        "processed_this_run": processed,
        "resumed": resume,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the STOV-ATOLL FAISS index.")
    parser.add_argument("--source", default="data/source")
    parser.add_argument("--index", default="data/index")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--embed-api",
        choices=["ollama", "openai"],
        default="ollama",
        help="Use 'openai' with --ollama-url http://127.0.0.1:8081 for start_embed_server.ps1.",
    )
    parser.add_argument("--chunk-size", type=int, default=180)
    parser.add_argument("--chunk-overlap", type=int, default=30)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Character cap per chunk so text without spaces stays within the model's token limit.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        default=60.0,
        help="How often the FAISS index and progress file are saved to disk.",
    )
    parser.add_argument(
        "--extract-dir",
        default=r"C:\rag_source",
        help="Short directory used when --source is a ZIP archive.",
    )
    args = parser.parse_args()
    result = build_index(
        Path(args.source),
        Path(args.index),
        args.embedding_model,
        args.ollama_url,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        workers=args.workers,
        extract_dir=Path(args.extract_dir),
        fresh=args.fresh,
        checkpoint_seconds=args.checkpoint_seconds,
        embed_api=args.embed_api,
        max_chars=args.max_chars,
    )
    print(
        f"Indexed {result['documents']} documents into {result['chunks']} chunks "
        f"({result['processed_this_run']} processed this run; resumed={result['resumed']})."
    )


if __name__ == "__main__":
    main()
