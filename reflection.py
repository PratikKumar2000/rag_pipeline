"""Self-review loop: the model critiques its own draft and may query the chunk database read-only."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm import chat

MAX_ROUNDS = 5
MAX_ROWS = 10
MAX_CELL_CHARS = 300
MAX_RESULT_CHARS = 3000
QUERY_SECONDS = 5
FINAL_MARK = "FINAL:"

TOOL_NAME = "run_sql"
QUERY_IN_TEXT = re.compile(r"(?is)\b(SELECT\b.*?)(?:\"\s*}|$)")
REPEATED_QUERY = "This query was already run; its result is above. Use it instead of running it again."

READ_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}

SQL_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Run one read-only SQLite SELECT against the indexed documents. "
            "Table records(id INTEGER, batch_id INTEGER, source TEXT file path, "
            "location TEXT page/sheet/slide, chunk INTEGER, text TEXT chunk content). "
            "Search with LIKE, for example: SELECT source, location, text FROM records "
            "WHERE text LIKE '%9105%' LIMIT 5. Returns at most 10 rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"],
        },
    },
}

REVIEW_PROMPT = f"""Review your answer above against the document excerpts and the question.
Check every claim. If a fact is missing, wrong or uncertain, call run_sql to look it up in the documents
and cite facts found that way as [DB].
The only table is records with columns id, batch_id, source, location, chunk, text. There are no other columns.
Search the text column, for example: SELECT source, text FROM records WHERE text LIKE '%keyword%' LIMIT 5
If the answer is correct and complete, reply with {FINAL_MARK} followed by the final answer.
Otherwise reply with an improved answer."""

LAST_ROUND_PROMPT = f"Reply now with {FINAL_MARK} followed by your best final answer. Do not call tools."


@dataclass
class Step:
    kind: str
    text: str


def _authorize(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in READ_ACTIONS else sqlite3.SQLITE_DENY


def run_sql(metadata_path: Path, query: str) -> str:
    """Run a model-written query; anything other than reading is rejected by SQLite itself."""
    deadline = time.monotonic() + QUERY_SECONDS
    try:
        with sqlite3.connect(f"file:{metadata_path}?mode=ro", uri=True) as connection:
            connection.set_authorizer(_authorize)
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
            cursor = connection.execute(query)
            columns = [column[0] for column in cursor.description or []]
            rows = cursor.fetchmany(MAX_ROWS)
    except sqlite3.Error as exc:
        return f"SQL error: {exc}"
    if not rows:
        return "No rows."
    lines = [" | ".join(columns)]
    for row in rows:
        lines.append(" | ".join(str(value)[:MAX_CELL_CHARS].replace("\n", " ") for value in row))
    return "\n".join(lines)[:MAX_RESULT_CHARS]


def _query_argument(call: dict[str, Any]) -> str:
    arguments = call.get("function", {}).get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return str(arguments.get("query", ""))


def _mentions_tool(text: str) -> bool:
    return TOOL_NAME in text


def _text_tool_calls(text: str) -> list[dict[str, Any]]:
    """Small models sometimes write the tool call as plain, even malformed, JSON instead of a structured call."""
    if not _mentions_tool(text):
        return []
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start:end + 1]) if start != -1 and end > start else None
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and data.get("name") == TOOL_NAME:
        arguments = data.get("parameters") or data.get("arguments") or {}
    else:
        query = QUERY_IN_TEXT.search(text)
        if not query:
            return []
        arguments = {"query": query.group(1).strip()}
    return [{"function": {"name": TOOL_NAME, "arguments": arguments}}]


def _strip_final(text: str) -> tuple[str, bool]:
    text = text.strip()
    position = text.upper().find(FINAL_MARK)
    if position == -1:
        return text, False
    return text[position + len(FINAL_MARK):].strip(), True


def refine_answer(prompt: str, metadata_path: Path, model: str, base_url: str) -> tuple[str, list[Step]]:
    """Draft an answer, then let the model review it, calling run_sql as needed, for at most MAX_ROUNDS calls."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    draft = chat(messages, model, base_url).get("content", "").strip()
    answer = draft
    steps = [Step("Draft", draft)]
    messages += [{"role": "assistant", "content": draft}, {"role": "user", "content": REVIEW_PROMPT}]
    queries_run: set[str] = set()

    for round_number in range(2, MAX_ROUNDS + 1):
        last_round = round_number == MAX_ROUNDS
        if last_round:
            messages.append({"role": "user", "content": LAST_ROUND_PROMPT})
        # The tool stays declared on the last round so Ollama parses any call instead of returning it as text.
        reply = chat(messages, model, base_url, tools=[SQL_TOOL])
        content = reply.get("content", "")
        calls = reply.get("tool_calls") or _text_tool_calls(content)
        if calls and last_round:
            steps.append(Step("Stopped", "The model asked for another lookup after the last round; the previous answer is kept."))
            break
        messages.append({"role": "assistant", "content": "", "tool_calls": calls} if calls else reply)

        for call in calls:
            query = _query_argument(call)
            result = REPEATED_QUERY if query in queries_run else run_sql(metadata_path, query)
            queries_run.add(query)
            steps.append(Step("SQL query", query))
            steps.append(Step("SQL result", result))
            messages.append({"role": "tool", "tool_name": TOOL_NAME, "content": result})
        if calls:
            continue

        text, final = _strip_final(content)
        if text and not _mentions_tool(text):
            answer = text
        steps.append(Step("Final" if final else "Revision", text))
        if final:
            break
        messages.append({"role": "user", "content": REVIEW_PROMPT})

    return answer, steps
