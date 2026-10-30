# STOV-ATOLL local RAG

This application indexes local copies of the STOV-ATOLL SharePoint documents and
answers questions using FAISS retrieval plus Ollama. The files and embeddings
remain on the laptop.

## Setup

Python 3.14 is supported by the installed FAISS wheel on this machine.

```powershell
cd stov_atoll_rag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install Ollama, then pull models sized for a 16 GB laptop:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

Copy the downloaded SharePoint files into `data\source`. Supported formats are
PDF, DOCX, XLSX/XLSM, PPT/PPTX, TXT, Markdown, and CSV. Legacy `.ppt` files
require LibreOffice (`soffice`) on PATH; modern `.pptx` files need no extra tool.

ZIP archives found anywhere below the source folder are discovered automatically.
They are expanded under `C:\rag_source\archive_*` to avoid long OneDrive paths.
You can also pass one ZIP directly:

```powershell
python ingest.py `
  --source "C:\path\to\STOV-ATOLL-documents.zip" `
  --extract-dir "C:\rag_source"
```

The recommended setup embeds through Ollama's bundled llama-server, which runs
several chunks in parallel on the GPU and checks exact token counts so no text
is truncated. Start it in its own window:

```powershell
ollama pull snowflake-arctic-embed:33m
.\start_embed_server.ps1 -Model snowflake-arctic-embed:33m -Parallel 8 -SlotContext 512
```

Then build the index. It checkpoints every 60 seconds and resumes automatically
when rerun with the same options:

```powershell
python ingest.py --source "C:\rag_docs\curated" --extract-dir "C:\rag_source" `
  --index data/index_arctic33 --embedding-model snowflake-arctic-embed:33m `
  --chunk-size 100 --chunk-overlap 20 --batch-size 8 --workers 1 `
  --embed-api openai --ollama-url http://127.0.0.1:8081
```

Use `--fresh` only when intentionally rebuilding the entire index from zero.
See [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) for why these settings
were chosen and the problems they solve.

Start the application:

```powershell
streamlit run app.py
```

Open the displayed local URL in a browser. The sidebar can rebuild the index
after source files change.

## SharePoint download note

The supplied folder is:

`https://amdocs.sharepoint.com/sites/OSSFactory_Cramer/Shared Documents/ATOLL STOV`

The authenticated SharePoint session is available to the development tools, but
this tenant currently returns malformed XML for its folder/search REST response.
Download the files from the authenticated browser session and place them in
`data\source`; ingestion is then fully local and repeatable.
