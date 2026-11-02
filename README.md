# STOV-ATOLL local RAG

A question-answering assistant over the STOV-ATOLL project documents. Documents,
embeddings, search indexes and models all stay on the laptop; nothing is sent
to a cloud service.

See [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) for the problems met
while building it and why each design choice was made.

## How it works

**Indexing** (`ingest.py`, run once and resumed as needed):

1. Reads every supported file under the source folder, plus the contents of
   every ZIP archive found there.
2. Splits text into chunks of up to 100 words and 1,000 characters, and splits
   any chunk that is still over the embedding model's 512-token limit, so no
   text is cut off.
3. Embeds the chunks with `snowflake-arctic-embed:33m` and stores the vectors in
   FAISS, with each chunk's text and source in SQLite.

**Answering** (`app.py` with `retrieval.py`, for every question):

1. **Query rewriting:** `llama3.2:3b` turns the question into 5 alternative
   search queries (rephrased, keywords only, abbreviations expanded, likely
   answer wording, broader topic).
2. **Hybrid search:** the original question and each rewrite run both semantic
   search (FAISS, top 50) and keyword search (SQLite FTS5 with BM25, top 50).
3. **Fusion:** reciprocal rank fusion merges all 12 result lists; duplicate
   excerpts from repeated document versions are removed.
4. **Reranking:** the cross-encoder `ms-marco-MiniLM-L6-v2` scores the best 50
   candidates against the original question, on the CPU.
5. **Release history:** when the question mentions versions, releases, builds
   or "latest", the 10 newest releases are added, sorted by the version in the
   release folder names (`STOV9105RC2_...` is 9.10.5 RC2). Similarity search
   cannot tell which release is newest, so this list answers "latest" questions.
6. **Answer:** the 10 best excerpts go to `llama3.2:3b`, which drafts an answer
   from them and cites excerpt numbers.
7. **Self-review:** the model reviews its own draft. It can call a `run_sql`
   tool to search the chunk database (cited as [DB]), and stops when it marks
   the answer `FINAL:`, or after 5 model calls at most. The tool is read-only:
   the database opens in read-only mode, an SQLite authorizer allows only
   reads, each query is limited to 5 seconds and 10 rows, and a repeated query
   is answered from the earlier result instead of running again. A tool call
   that the model writes as plain text, even malformed JSON, is run as a call
   and never shown as the answer; a call on the last round is ignored and the
   previous answer kept. The app shows the queries searched, the review steps
   and the excerpts used.

## Files

| File | Purpose |
|---|---|
| `stage_documents.py` | Builds a curated copy of the raw documents, excluding `Package` and `Artifacts` folders and `lib.zip` |
| `ingest.py` | Extracts, chunks, embeds and indexes documents; resumable |
| `start_embed_server.ps1` | Starts Ollama's bundled llama-server with parallel slots for fast embedding |
| `retrieval.py` | Query rewriting, hybrid search, fusion and reranking |
| `reflection.py` | Self-review loop and the read-only SQL tool |
| `llm.py` | Client for Ollama and llama-server |
| `app.py` | Streamlit question-and-answer UI |

## Setup

Python 3.14 is used because the FAISS wheel installs cleanly on it.

```powershell
cd C:\apps\cursor-code\stov_atoll_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install [Ollama](https://ollama.com), switch it to the Vulkan backend so it uses
the integrated GPU, and pull the two models:

```powershell
ollama pull snowflake-arctic-embed:33m
ollama pull llama3.2:3b
```

The reranker (about 90 MB) downloads automatically from Hugging Face into
`data\models` the first time a question is asked.

## Getting the documents

The SharePoint REST API on this tenant returns malformed responses, so sync the
folder with OneDrive instead:

`https://amdocs.sharepoint.com/sites/OSSFactory_Cramer/Shared Documents/ATOLL STOV`

Copy the synced files to a short path such as `C:\rag_docs` (long OneDrive
paths break ZIP extraction), then build the curated copy:

```powershell
python stage_documents.py --source C:\rag_docs --destination C:\rag_docs\curated
```

Supported formats: PDF, DOCX, XLSX/XLSM, PPTX, PPT, TXT, Markdown and CSV.
Legacy `.ppt` files need LibreOffice (`soffice`) on `PATH`; `.doc`, `.xls` and
`.rtf` are not read. Office lock files (`~$...`) and unreadable OneDrive
placeholders are skipped with a message.

## Building the index

1. Start the embedding server in its own window and leave it open. It runs
   8 chunks at a time on the GPU, which is about twice as fast as Ollama:

   ```powershell
   .\start_embed_server.ps1 -Model snowflake-arctic-embed:33m -Parallel 8 -SlotContext 512
   ```

2. In another window, run the indexer:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python ingest.py --source "C:\rag_docs\curated" --extract-dir "C:\rag_source" `
     --index data/index_arctic33 --embedding-model snowflake-arctic-embed:33m `
     --chunk-size 100 --chunk-overlap 20 --batch-size 8 --workers 1 `
     --embed-api openai --ollama-url http://127.0.0.1:8081
   ```

- **Resuming:** the index is saved every 60 seconds and on exit. Rerunning the
  same command continues from the last saved batch.
- **Rebuilding:** changing the source, model, chunk size, overlap, `--max-chars`,
  batch size or worker count starts the index again automatically. Use
  `--fresh` to force a rebuild with unchanged settings.
- **Progress:** `Checkpoint: N chunks indexed.` lines in the indexer window,
  slot activity in the server window, or `data\index_arctic33\progress.json`.
- **Without llama-server:** omit `--embed-api` and `--ollama-url` to embed
  through Ollama. It is slower, and Ollama silently truncates any chunk over
  512 tokens instead of letting the indexer split it.

## Running the app

Ollama must be running; the embedding server is not needed for questions.

```powershell
streamlit run app.py
```

Open the local URL it prints and type a question. Restart the app after
indexing finishes so it loads the complete index.

Settings are constants, not UI options:

| Setting | File | Default |
|---|---|---|
| Index folder, embedding and answer models, excerpts sent to the answer model (`TOP_K`) | `app.py` | `data/index_arctic33`, `snowflake-arctic-embed:33m`, `llama3.2:3b`, 10 |
| Query rewrites (`QUERY_VARIANTS`) | `retrieval.py` | 5 |
| Semantic and keyword results per query (`DENSE_K`, `LEXICAL_K`) | `retrieval.py` | 50 each |
| Candidates reranked (`RERANK_K`) | `retrieval.py` | 50 |
| Model calls per answer, rows and seconds per SQL query (`MAX_ROUNDS`, `MAX_ROWS`, `QUERY_SECONDS`) | `reflection.py` | 5, 10, 5 |
| Context window for every answer-model call (`CONTEXT_TOKENS`) | `llm.py` | 8,192 |

On this laptop a question takes roughly 10 s for rewriting, 5–6 s for search
and reranking, and 40–110 s for drafting and self-review. Lowering
`MAX_ROUNDS`, `RERANK_K` or `TOP_K` makes answers faster.

### Context window and memory

`llama3.2:3b` supports 131,072 tokens, but Ollama runs it with 4,096 unless a
request sets `num_ctx`. One self-review conversation holds at most about
7,000 tokens (excerpts and instructions about 3,500, up to three SQL results
of about 750 each, plus drafts), so every call uses 8,192. All calls use the
same value so Ollama never reloads the model between them.

Ollama reserves the KV cache for the whole window up front, about 112 KB per
token for this model:

| `num_ctx` | KV cache | With the 2 GB of weights |
|---|---|---|
| 4,096 | 0.45 GB | 2.5 GB |
| 8,192 | 0.9 GB | 2.9 GB |
| 16,384 | 1.8 GB | 3.8 GB |
| 131,072 | 14.7 GB | about 17 GB, too large for this laptop |

- `num_ctx` is the window of one conversation: prompt plus generated text.
- `num_batch` (default 512) is how many prompt tokens are processed per step
  for that same conversation. All steps write into the same KV cache, so it
  does not multiply memory; it only sizes a temporary compute buffer.
- Parallel slots (`OLLAMA_NUM_PARALLEL`, or `-np` for llama-server) do
  multiply it: each slot has its own KV cache, while the weights are shared.
  llama-server splits `-c` across slots, so the embedding server's
  `-c 4096 -np 8` gives each slot 512 tokens.

Size a slot as longest prompt plus longest output; total KV memory is about
slot size × slots × KV bytes per token. Setting `OLLAMA_FLASH_ATTENTION=1` and
`OLLAMA_KV_CACHE_TYPE=q8_0` roughly halves the cache if more room is needed.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ConnectionResetError: [WinError 10054]` in the Streamlit terminal | The browser closed its connection (refresh or closed tab). Harmless. |
| "The assistant is not available right now" | Ollama is not running, or a model is missing; check `ollama list`. |
| `exceed_context_size_error` from llama-server | A chunk is over 512 tokens; only happens when the token check is bypassed. Rerun with `--embed-api openai`. |
| `Access is denied` while saving the index | A scanner briefly locked the file; the indexer retries automatically. If it still fails, rerun and it resumes. |
| Two `python.exe` processes per run | Normal: the venv launcher starts the real interpreter as a child. |
| Answer is raw JSON such as `{"name":"run_sql",...}` | The 3B model wrote a tool call as text. `reflection.py` now runs such calls and never uses them as the answer; restart Streamlit if it still appears. |
| Self-review steps show `SQL error: no such column` | The model guessed a column; the error goes back to it and it usually corrects the query in the next round. |

The authenticated SharePoint session is available to the development tools, but
this tenant currently returns malformed XML for its folder/search REST response.
Download the files from the authenticated browser session and place them in
`data\source`; ingestion is then fully local and repeatable.

