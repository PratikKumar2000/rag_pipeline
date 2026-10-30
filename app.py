"""Streamlit UI for local STOV-ATOLL question answering."""

from __future__ import annotations

import sqlite3
import traceback
from pathlib import Path

import faiss
import numpy as np
import streamlit as st

from llm import OllamaError, embed, generate

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index_arctic33"
INDEX_FILE = INDEX_DIR / "stov_atoll.faiss"
METADATA_FILE = INDEX_DIR / "metadata.sqlite3"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "snowflake-arctic-embed:33m"
ANSWER_MODEL = "llama3.2:3b"
TOP_K = 10
# Arctic Embed retrieves well only when queries (not documents) carry this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

st.set_page_config(page_title="STOV-ATOLL Assistant", page_icon="📡", layout="wide")
st.title("STOV-ATOLL Assistant")
st.caption("Ask a question about STOV-ATOLL. Answers come only from the indexed project documents.")


@st.cache_resource
def load_index() -> faiss.Index | None:
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        return None
    return faiss.read_index(str(INDEX_FILE))


def retrieve(question: str) -> list[tuple[float, dict[str, str]]]:
    index = load_index()
    if index is None or index.ntotal == 0:
        return []
    vector = np.asarray(embed([QUERY_PREFIX + question], EMBEDDING_MODEL, OLLAMA_URL), dtype="float32")
    faiss.normalize_L2(vector)
    scores, positions = index.search(vector, min(TOP_K, index.ntotal))
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
                        {"source": row[0], "location": row[1], "chunk": row[2], "text": row[3]},
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


def show_matches(matches: list[tuple[float, dict[str, str]]]) -> None:
    with st.expander(f"Retrieved excerpts ({len(matches)})"):
        for number, (score, record) in enumerate(matches, start=1):
            st.markdown(
                f"**[{number}] `{record['source']}`** — {record['location']}, "
                f"chunk {record['chunk']} (similarity {score:.3f})"
            )
            st.text(record["text"])


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("matches"):
            show_matches(message["matches"])

if load_index() is None:
    st.info("The document index is not available yet. Please try again once indexing has finished.")

question = st.chat_input("Ask a question about STOV-ATOLL")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents..."):
                matches = retrieve(question)
            if not matches:
                answer = "No indexed documents are available yet."
            else:
                with st.spinner("Writing the answer..."):
                    answer = generate(make_prompt(question, matches), ANSWER_MODEL, OLLAMA_URL)
            st.markdown(answer)
            if matches:
                show_matches(matches)
            st.session_state.messages.append({"role": "assistant", "content": answer, "matches": matches})
        except OllamaError:
            st.error("The assistant is not available right now. Please check that Ollama is running.")
        except Exception as exc:
            traceback.print_exc()
            st.error(f"Query failed ({type(exc).__name__}): {exc}")
