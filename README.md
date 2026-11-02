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
5. **Answer:** the 10 best excerpts go to `llama3.2:3b`, which answers only from
   them and cites excerpt numbers. The app shows the queries searched and the
   excerpts used.

## Files

| File | Purpose |
|---|---|
| `stage_documents.py` | Builds a curated copy of the raw documents, excluding `Package` and `Artifacts` folders and `lib.zip` |
| `ingest.py` | Extracts, chunks, embeds and indexes documents; resumable |
| `start_embed_server.ps1` | Starts Ollama's bundled llama-server with parallel slots for fast embedding |
| `retrieval.py` | Query rewriting, hybrid search, fusion and reranking |
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

On this laptop a question takes roughly 10 s for rewriting, 5–6 s for search
and reranking, and 25–80 s for the answer while indexing shares the GPU.
Lowering `RERANK_K` or `TOP_K` makes answers faster.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ConnectionResetError: [WinError 10054]` in the Streamlit terminal | The browser closed its connection (refresh or closed tab). Harmless. |
| "The assistant is not available right now" | Ollama is not running, or a model is missing; check `ollama list`. |
| `exceed_context_size_error` from llama-server | A chunk is over 512 tokens; only happens when the token check is bypassed. Rerun with `--embed-api openai`. |
| `Access is denied` while saving the index | A scanner briefly locked the file; the indexer retries automatically. If it still fails, rerun and it resumes. |
| Two `python.exe` processes per run | Normal: the venv launcher starts the real interpreter as a child. |
