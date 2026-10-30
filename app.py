"""Streamlit UI for local STOV-ATOLL semantic search and Q&A."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from ingest import build_index
from llm import OllamaError, embed, generate

ROOT = Path(__file__).parent
SOURCE_DIR = ROOT / "data" / "source"
INDEX_DIR = ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "stov_atoll.faiss"
METADATA_FILE = INDEX_DIR / "metadata.sqlite3"

st.set_page_config(page_title="STOV-ATOLL Assistant", page_icon="📡", layout="wide")
st.title("STOV-ATOLL Assistant")
st.caption("Local SharePoint-document retrieval with FAISS and Ollama")

with st.sidebar:
    st.header("Settings")
    embedding_model = st.text_input("Embedding model", "all-minilm")
    generation_model = st.text_input("Answer model", "llama3.2:3b")
    ollama_url = st.text_input("Ollama URL", "http://127.0.0.1:11434")
    top_k = st.slider("Retrieved chunks", 1, 10, 5)
    st.divider()
    st.write(f"Source folder: `{SOURCE_DIR}`")
    if st.button("Rebuild FAISS index", type="primary"):
        with st.spinner("Extracting documents and creating embeddings..."):
            try:
                result = build_index(
                    SOURCE_DIR,
                    INDEX_DIR,
                    embedding_model,
                    ollama_url,
                )
                st.cache_resource.clear()
                st.success(f"Indexed {result['documents']} documents and {result['chunks']} chunks.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


@st.cache_resource
def load_index() -> faiss.Index | None:
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        return None
    return faiss.read_index(str(INDEX_FILE))


def retrieve(question: str, top_k: int, model: str, base_url: str) -> list[tuple[float, dict[str, str]]]:
    loaded = load_index()
    if loaded is None:
        return []
    index = loaded
    vector = np.asarray(embed([question], model, base_url), dtype="float32")
    faiss.normalize_L2(vector)
    scores, positions = index.search(vector, min(top_k, index.ntotal))
    matches: list[tuple[float, dict[str, str]]] = []
    with sqlite3.connect(METADATA_FILE) as connection:
        for score, position in zip(scores[0], positions[0]):
            if position < 0:
                continue
            row = connection.execute(
                "SELECT source, location, chunk, text FROM records WHERE id = ?",
                (int(position),),
            ).fetchone()
            if row:
                matches.append(
                    (
                        float(score),
                        {
                            "source": row[0],
                            "location": row[1],
                            "chunk": row[2],
                            "text": row[3],
                        },
                    )
                )
    return matches


def make_prompt(question: str, matches: list[tuple[float, dict[str, str]]]) -> str:
    context = "\n\n".join(
        f"[{number}] {record['source']} ({record['location']}, chunk {record['chunk']})\n{record['text']}"
        for number, (_, record) in enumerate(matches, start=1)
    )
    return f"""You answer questions about STOV-ATOLL using only the supplied document excerpts.
If the excerpts do not contain the answer, say that the indexed documents do not provide enough information.
Do not invent facts. Keep the response concise and cite excerpt numbers such as [1] or [2].

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}
"""


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not INDEX_FILE.exists():
    st.info(
        "Copy SharePoint documents into `data/source`, install Ollama models, then click "
        "**Rebuild FAISS index**."
    )

question = st.chat_input("Ask a question about STOV-ATOLL")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            matches = retrieve(question, top_k, embedding_model, ollama_url)
            if not matches:
                answer = "No indexed documents are available yet."
            else:
                answer = generate(make_prompt(question, matches), generation_model, ollama_url)
                answer += "\n\n**Sources**\n" + "\n".join(
                    f"- `{record['source']}` — {record['location']} (similarity {score:.3f})"
                    for score, record in matches
                )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except OllamaError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Query failed: {exc}")
