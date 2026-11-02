"""Streamlit UI for local STOV-ATOLL question answering."""

from __future__ import annotations

import traceback
from pathlib import Path

import faiss
import streamlit as st

from llm import OllamaError, generate
from retrieval import Match, Reranker, hybrid_search, rewrite_queries, sync_lexical_index

ROOT = Path(__file__).parent
INDEX_DIR = ROOT / "data" / "index_arctic33"
INDEX_FILE = INDEX_DIR / "stov_atoll.faiss"
METADATA_FILE = INDEX_DIR / "metadata.sqlite3"
MODEL_DIR = ROOT / "data" / "models"
OLLAMA_URL = "http://127.0.0.1:11434"
EMBEDDING_MODEL = "snowflake-arctic-embed:33m"
ANSWER_MODEL = "llama3.2:3b"
TOP_K = 10

st.set_page_config(page_title="STOV-ATOLL Assistant", page_icon="📡", layout="wide")
st.title("STOV-ATOLL Assistant")
st.caption("Ask a question about STOV-ATOLL. Answers come only from the indexed project documents.")


@st.cache_resource
def load_index() -> faiss.Index | None:
    if not INDEX_FILE.exists() or not METADATA_FILE.exists():
        return None
    return faiss.read_index(str(INDEX_FILE))


@st.cache_resource
def load_reranker() -> Reranker:
    return Reranker(MODEL_DIR)


def retrieve(question: str, queries: list[str]) -> list[Match]:
    index = load_index()
    if index is None:
        return []
    lexical_path = sync_lexical_index(INDEX_DIR)
    return hybrid_search(
        question, queries, index, INDEX_DIR, lexical_path, load_reranker(), EMBEDDING_MODEL, OLLAMA_URL, TOP_K
    )


def make_prompt(question: str, matches: list[Match]) -> str:
    context = "\n\n".join(
        f"[{number}] {match.source} ({match.location}, chunk {match.chunk})\n{match.text}"
        for number, match in enumerate(matches, start=1)
    )
    return f"""You answer questions about STOV-ATOLL using only the supplied document excerpts.
If the excerpts do not contain the answer, say that the indexed documents do not provide enough information.
Do not invent facts. Keep the response concise and cite excerpt numbers such as [1] or [2].

DOCUMENT EXCERPTS:
{context}

QUESTION:
{question}
"""


def show_queries(queries: list[str]) -> None:
    with st.expander(f"Searched with {len(queries)} queries"):
        st.markdown("\n".join(f"- {query}" for query in queries))


def show_matches(matches: list[Match]) -> None:
    with st.expander(f"Retrieved excerpts ({len(matches)})"):
        for number, match in enumerate(matches, start=1):
            st.markdown(
                f"**[{number}] `{match.source}`** — {match.location}, chunk {match.chunk} "
                f"(relevance {match.rerank_score:.2f}; found by {' + '.join(match.found_by)} search)"
            )
            st.text(match.text)


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("queries"):
            show_queries(message["queries"])
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
            with st.spinner("Understanding the question..."):
                queries = [question] + rewrite_queries(question, ANSWER_MODEL, OLLAMA_URL)
            with st.spinner("Searching the documents..."):
                matches = retrieve(question, queries)
            if not matches:
                answer = "No indexed documents are available yet."
            else:
                with st.spinner("Writing the answer..."):
                    answer = generate(make_prompt(question, matches), ANSWER_MODEL, OLLAMA_URL)
            st.markdown(answer)
            show_queries(queries)
            if matches:
                show_matches(matches)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "queries": queries, "matches": matches}
            )
        except OllamaError:
            st.error("The assistant is not available right now. Please check that Ollama is running.")
        except Exception as exc:
            traceback.print_exc()
            st.error(f"Query failed ({type(exc).__name__}): {exc}")
