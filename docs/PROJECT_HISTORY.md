# STOV-ATOLL local RAG: project history

This document records how the STOV-ATOLL retrieval system was built, which
problems blocked progress, what caused each one, and how it was resolved. It is
meant for anyone who has to run, extend, or debug the system later.

## 1. Goal

Answer questions about STOV-ATOLL from the project's own documents, entirely on
a 16 GB Windows laptop with no cloud services:

- collect the documents from the `ATOLL STOV` SharePoint folder;
- extract text, split it into chunks, and embed the chunks with a free Ollama model;
- store the vectors in FAISS;
- answer questions in a Streamlit app using retrieved excerpts and a local LLM.

## 2. Where it ended up

```mermaid
flowchart LR
    SP[SharePoint folder] -->|OneDrive sync| RAW[C:\rag_docs]
    RAW -->|stage_documents.py| CUR[C:\rag_docs\curated]
    CUR -->|prepare_sources: files + ZIPs expanded to C:\rag_source| EXT[extract_documents]
    EXT --> CH[chunk_text: words + character cap]
    CH --> SPLIT[token check and lossless split]
    SPLIT --> EMB[llama-server, 8 slots, snowflake-arctic-embed:33m]
    EMB --> IDX[(FAISS + SQLite metadata)]
    IDX --> APP[Streamlit app.py]
    APP -->|answer| LLM[Ollama llama3.2:3b]
```

| Component | File | Role |
|---|---|---|
| Staging | `stage_documents.py` | Builds a curated copy: every non-ZIP file plus ZIP contents, excluding `Package`, `Artifacts` and `lib.zip` |
| Ingestion | `ingest.py` | Discovers files and nested ZIPs, extracts text, chunks, embeds, writes FAISS and SQLite, resumes after failures |
| Embedding client | `llm.py` | Talks to Ollama (`/api/embed`, `/api/generate`) or llama-server (`/v1/embeddings`, `/tokenize`) |
| Embedding server | `start_embed_server.ps1` | Starts Ollama's bundled `llama-server.exe` with several parallel slots on the GPU |
| Application | `app.py` | Streamlit question-and-answer UI over the index |

## 3. Blockers and how each was resolved

### 3.1 SharePoint could not be scraped

- **Symptom:** the SharePoint REST API returned HTTP 400 for the sharing link,
  then malformed, non-JSON XML for folder listing and search.
- **Cause:** the tenant is fronted by a proxy that rewrites responses; the API
  could not be used reliably from tooling.
- **Resolution:** the folder was synced with OneDrive to
  `C:\Users\pratikk\OneDrive - AMDOCS\Vodafone OSS Factory - ATOLL STOV`, and
  ingestion reads local files only. This also makes ingestion repeatable offline.

### 3.2 Python environment

- **Symptom:** creating a virtual environment with the Anaconda Python 3.12
  install failed.
- **Resolution:** a project virtual environment on Python 3.14 in
  `stov_atoll_rag\.venv`, with `faiss-cpu` 1.15.1. On Windows, a venv's
  `python.exe` is a launcher that starts the real interpreter as a child, so two
  `python` processes per run is normal, not a bug.

### 3.3 Unreadable files stopped the whole run

- **Symptom:** `PackageNotFoundError` partway through ingestion.
- **Cause:** OneDrive "online-only" placeholder files, and Office lock files
  named `~$...` that look like documents but are not.
- **Resolution:** every file is read inside its own `try`/`except`; failures are
  reported as "Skipping unavailable or unreadable file" and the run continues.
  Names starting with `~$` are skipped outright.

### 3.4 ZIP archives and Windows path length

- **Symptom:** most delivery material lives inside ZIP bundles, and extracting
  them under the OneDrive folder produced paths beyond Windows' length limit.
- **Resolution:** `prepare_sources` reads every file in the source folder and
  additionally discovers every ZIP below it. Each ZIP is expanded under the short
  path `C:\rag_source\run_<timestamp>\archive_NNNN` and added as another root, so
  loose files and ZIP contents are indexed together. Extraction rejects entries
  that would escape the target folder.

### 3.5 Re-extraction failed on locked folders

- **Symptom:** `WinError 32` from `shutil.rmtree` when a rerun tried to delete
  the previous extraction.
- **Cause:** Windows Search, Defender or the editor's file watcher holding files
  open in the old folder.
- **Resolution:** each run extracts into a new timestamped folder instead of
  deleting the old one.

### 3.6 Ollama errors, timeouts and a batch-size bug

- **Symptom:** HTTP 404 on `/api/embed`, then `ReadTimeout` after 300 seconds.
- **Causes:** Ollama was not fully running; batches of 32 chunks were too large
  for the laptop; and the batch slice was hard-coded as `start + 32`, so the
  `--batch-size` option had no effect.
- **Resolution:** the slice uses `batch_size`, defaults became small, and every
  Ollama request sends `keep_alive: -1` so the model is not unloaded and
  reloaded between requests.

### 3.7 CPU at 100% and the laptop hung

- **Symptom:** the whole system froze during embedding.
- **Cause:** Ollama was running inference on the CPU.
- **Resolution:** Ollama was switched to its Vulkan backend, which runs on the
  Intel Iris Xe integrated GPU. `ollama ps` then reported 100% GPU.

### 3.8 The full corpus would take a long time and any failure lost everything

- **Resolution:** ingestion became streaming, concurrent and resumable:
  - files are read and chunked lazily, so memory stays flat;
  - up to four embedding requests run at once, and results are committed
    strictly in batch order by the main thread;
  - each chunk's text and source are stored in SQLite keyed by its FAISS row id;
  - `progress.json` stores a hash of the settings (source, model, chunk size,
    overlap, character cap, batch size, workers). A rerun with the same settings
    continues from the last committed batch; different settings start again;
  - on resume, any SQLite rows beyond the saved FAISS index are removed, so the
    two stores always agree.

### 3.9 Legacy PowerPoint files

- **Resolution:** `.pptx` is read with `python-pptx`. `.ppt` is converted to
  `.pptx` with LibreOffice (`soffice --headless`) first; LibreOffice is not
  installed yet, so `.ppt` files are currently skipped with a message.

### 3.10 Too much irrelevant material

- **Symptom:** about 9 GB of source, much of it build output and libraries.
- **Resolution:** `stage_documents.py` creates `C:\rag_docs\curated` containing
  every non-ZIP file plus the contents of each ZIP, excluding any `Package` or
  `Artifacts` folder and any `lib.zip`. The first run copied 650 files and
  expanded 54 ZIP archives.

### 3.11 "Access is denied" when saving the index

- **Symptom:** `WinError 5` on `os.replace` when swapping in the new
  `stov_atoll.faiss` or `progress.json`, which killed long runs.
- **Cause:** not a second ingestion process (the two PIDs were the venv
  launcher), but a scanner or file watcher briefly holding the file just as it
  was replaced. Saving after every batch made the collision near-certain.
- **Resolution:** replacements retry with back-off for several seconds;
  temporary files carry the process id; the index is checkpointed every 60
  seconds instead of every batch, plus once on exit. A simulated crash with three
  forced denials resumed to the same 66 chunks as a clean run.

### 3.12 Ollama would not run embeddings in parallel

- **Symptom:** the server log showed only `slot id 0` even with
  `OLLAMA_NUM_PARALLEL=4`.
- **Cause:** the setting was read (Ollama requested `num_ctx=8192`, which is
  4 × 2048), but Ollama starts every embedding model with `-np 1` and caps the
  context at the model's training length. No Ollama setting changes this.
- **Resolution:** `start_embed_server.ps1` runs the `llama-server.exe` bundled
  with Ollama directly on the same model file, with several slots. It must set
  `GGML_BACKEND_PATH` to the Vulkan DLL, otherwise the server finds no GPU.
  Vectors from llama-server and Ollama were compared: cosine similarity 1.0.

### 3.13 Parallel slots did not help with large chunks

Measured on the Iris Xe GPU:

| Test | Result |
|---|---|
| `nomic-embed-text`, 2 chunks of about 1,700 tokens, one request | 8.8 s |
| Same, four requests at once on a 4-slot server | 37.8 s (no gain) |
| Same model on the CPU only | 73.2 s vs 10.5 s on the GPU |

Large chunks already saturate the GPU, and the CPU is about seven times slower,
so neither more slots nor a CPU worker helps. The model is already stored in
FP16, so lowering precision was not an option either.

### 3.14 Switching to a smaller embedding model

- **Decision:** `snowflake-arctic-embed:33m` (33M parameters, 384 dimensions,
  512-token limit). Its published retrieval score is close to
  `nomic-embed-text` (137M) and well above `all-minilm`.
- **Consequence:** chunks must fit 512 tokens, so chunk size dropped to 100
  words with 20 words of overlap.
- **Result:** small chunks leave the GPU under-used, so parallel slots now pay
  off. A batch of 8 chunks took 0.85 s on an 8-slot llama-server versus 1.59 s
  through Ollama, roughly 1.9× faster.

### 3.15 Chunks silently losing data

- **Symptom:** llama-server rejected a batch with
  `input (... tokens) is larger than the max context size (512 tokens)`.
- **Cause:** `chunk_text` counted whitespace-separated words, and much of the
  corpus has long runs with no spaces: semicolon-separated CSV rows (one 383
  character "word" in `PartnerCellData.csv`), Java source paths, XML. One
  100-word chunk could be several thousand tokens. Ollama had been truncating
  such chunks without any warning: 96 of the first 3,592 chunks (2.7%) had lost
  text.
- **Resolution:**
  - a space is inserted after `;` and `|`, so CSV fields become separate words;
  - each chunk is capped at `--max-chars` (default 1,000) as well as the word
    count, and any single word over 250 characters is cut into pieces;
  - with llama-server, each chunk's exact token count is checked through
    `/tokenize`, and a chunk that is still too long is split in half at a space,
    repeatedly, until every piece fits. Pieces are stored as `5.1`, `5.2`, ...
    with the same source and location, so no text is dropped;
  - the earlier truncation fallback was removed, so any future overflow raises
    an error instead of losing text.
- **Verification:** `PartnerCellData.csv` became 17 chunks with a maximum of
  388 tokens and every word kept; a forced chunk of 400 IP addresses split into
  8 pieces with every character kept.

### 3.16 Retrieval quality: hybrid search with reranking

- **Problem:** semantic search alone misses exact identifiers (`LTE_CELL_ID`,
  `GreySpot.jar`, table names), and its similarity score is a weak judge of
  which excerpt actually answers the question.
- **Resolution (`retrieval.py`):**
  1. **Semantic search:** FAISS returns the 50 nearest chunks to the question.
  2. **Keyword search:** a SQLite FTS5 index with BM25 ranking returns the 50
     best keyword matches. It lives in `lexical.sqlite3` next to the FAISS
     index, is built automatically, and only adds new chunks on each query, so
     it keeps up with an indexing run in progress (first build of 43,726 chunks:
     2.5 s; later syncs: about 0.02 s).
  3. **Fusion:** reciprocal rank fusion merges both lists by rank, so the two
     different score scales never need to be compared.
  4. **Deduplication:** identical excerpts from repeated document versions are
     kept once.
  5. **Reranking:** the top 50 candidates are scored by the cross-encoder
     `cross-encoder/ms-marco-MiniLM-L6-v2` (22M parameters, ONNX on the CPU),
     which reads the question and each excerpt together. The 10 best go to the
     answer model.
- **Result:** for "what is cell compound" the top excerpts became the cell and
  cell-group definitions from the KT summary document. A query takes about
  4–8 seconds for retrieval and reranking on the CPU while indexing runs.

### 3.17 Too many "not enough information" answers: query transformation

- **Problem:** users type short, informal questions ("what is cell compound"),
  while the documents use formal, varied wording. A single query often missed
  the passages that held the answer.
- **Resolution (multi-query retrieval):**
  1. `llama3.2:3b` rewrites the question into 5 search queries (JSON output,
     capped at 250 tokens): a clear rephrasing, a keyword version, one with
     abbreviations expanded and synonyms, a sentence the answering passage might
     contain, and a broader variant. A worked example in the prompt was needed;
     without it the 3B model produced only near-identical keyword variants.
  2. The original question plus the 5 rewrites each run semantic and keyword
     search: 12 ranked lists. All query embeddings go in one request and one
     batched FAISS search.
  3. Reciprocal rank fusion merges all 12 lists, so chunks found by several
     queries rise to the top; the best 50 are deduplicated.
  4. The cross-encoder reranks them against the user's original question, and
     the best 10 go to the answer model.
  5. If rewriting fails, the search continues with the original question only.
- **Cost:** query rewriting adds about 8–13 s per question on this laptop while
  indexing shares the GPU.

## 4. Running it today

1. Start the embedding server in its own window and leave it open:

   ```powershell
   cd C:\apps\cursor-code\stov_atoll_rag
   .\start_embed_server.ps1 -Model snowflake-arctic-embed:33m -Parallel 8 -SlotContext 512
   ```

2. Build or resume the index:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   python ingest.py --source "C:\rag_docs\curated" --extract-dir "C:\rag_source" `
     --index data/index_arctic33 --embedding-model snowflake-arctic-embed:33m `
     --chunk-size 100 --chunk-overlap 20 --batch-size 8 --workers 1 `
     --embed-api openai --ollama-url http://127.0.0.1:8081
   ```

   Rerunning the same command resumes. Changing any chunking, model, batch or
   worker option starts the index again.

3. Watch progress: `Checkpoint: N chunks indexed.` lines in the ingest terminal,
   slot activity in the server window, or `data\index_arctic33\progress.json`.

## 5. Lessons learned

- Measure before optimising. Parallelism, precision and CPU offload all looked
  promising and none helped with large chunks; the real lever was the number of
  tokens embedded.
- Words are not tokens. Technical documents can produce many tokens per word,
  and a silent truncation is worse than an error.
- On Windows, expect transient file locks from scanners and watchers; design
  file replacement and cleanup to retry or avoid deletion.
- Make long jobs resumable from the start, with a settings hash so a resumed
  run can never mix incompatible chunks.

## 6. Open items

- Many bundles are repeated versions of the same documents; skipping files with
  identical content would shorten indexing considerably.
- `.doc`, `.xls` and `.rtf` files are not read; `.ppt` needs LibreOffice.
- When embedding through Ollama instead of llama-server, token counts cannot be
  checked, so the character cap is the only protection against truncation.
- `C:\rag_docs\Knowledge Repository` and `C:\rag_docs\STOV - ST Delivery` could
  not be deleted because of file locks; they are harmless while `--source`
  points at `curated`.
