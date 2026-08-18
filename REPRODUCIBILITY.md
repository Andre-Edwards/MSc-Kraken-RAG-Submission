# Reproducibility check

This file gives the shortest route for checking the submitted application and
evaluation. Run the commands from a clean clone with Python 3.12 and Node.js 20
or later.

## 1. Install the project

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd ..\frontend
npm ci
npm run build
```

## 2. Run the automated tests

```powershell
cd ..\backend
python -m unittest discover -s tests -v
```

The tests cover both chunking strategies, backend startup, seeded login and
the local frontend CORS origins.

## 3. Verify the human-labelled datasets

```powershell
python evaluation/build_gold_standards.py --check
```

This rebuilds the three JSON representations in memory from their Excel
workbooks and checks that the committed JSON is identical.

## 4. Verify the submitted metric calculations

```powershell
python evaluation/verify_evaluation_scores.py
```

This does not call OpenAI or perform retrieval. It recomputes every saved
summary from the per-question rows in:

- `evaluation/results/original_source_level_at5.json`;
- `evaluation/results/expanded_source_level_at5.json`; and
- `evaluation/results/chunk_level_at5.json`.

It should finish with `All evaluation summaries verified successfully.`

## 5. Run the application

Create `backend/.env` from `.env.example`, add an OpenAI API key, and then run:

```powershell
python -m app.scripts.ingest --force
python -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
npm run dev
```

Open `http://127.0.0.1:5173`.

## Reproduction boundary

The PDF files and their hashes are committed. The web snapshot is represented
by `data/web_sources/web_corpus_manifest.csv`, which contains source URLs,
crawl timestamps, extracted-text lengths and hashes. The full copied website
text is not committed. Recrawling the seed URLs can therefore produce different
retrieval rankings if the public pages have changed.

The submitted result JSON preserves the complete retrieved rows used for the
reported calculations, so its arithmetic and relevance classifications remain
independently checkable even when a later live crawl differs.
