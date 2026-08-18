# Kraken Policy Assistant

This repository contains the practical part of my MSc dissertation project. I built a retrieval-augmented generation (RAG) application that answers questions using a controlled collection of public Kraken policy documents and selected Kraken web pages.

The aim was not simply to build another chatbot. I wanted to investigate how the way documents are divided and retrieved affects whether an answer is complete, grounded in the source material and supported by useful citations.

My research question is:

> How do chunking and retrieval design choices affect citation-grounded answers in a policy-document RAG assistant?

This is an independent academic prototype. It is not an official Kraken product and should not be used as legal, financial, compliance or investment advice.

For a shorter checking sequence, see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). The evaluation method is described in [`backend/evaluation/README.md`](backend/evaluation/README.md), and the submitted corpus is documented in [`data/CORPUS.md`](data/CORPUS.md).

## What the application does

Users can ask practical questions about the material in the corpus. The application retrieves relevant passages, sends a limited evidence set to an OpenAI generation model and returns an answer with citations.

The interface lets users inspect the evidence behind an answer rather than relying on the answer alone. It also provides feedback controls and stores chat activity for later review.

An administrator can:

- choose between fixed-size, structure-aware and hybrid retrieval;
- enable or disable metadata-aware reranking;
- upload or remove PDF documents;
- crawl selected public Kraken web pages;
- rebuild the vector indexes;
- inspect recent chats, evidence and evaluation information;
- review refused or low-scoring responses in an insights dashboard; and
- configure when users are asked to complete a human review form.

## Why I used Kraken as the case study

I chose a public crypto-exchange policy corpus because it contains structured legal, regulatory, privacy, risk and operational information. This made it a useful case study for testing whether a RAG system can find the right policy evidence and explain it in a more accessible way.

Only public official documents and public web pages are used. The project does not contain internal Kraken material.

## How the RAG pipeline works

The main pipeline is:

```text
Public PDFs and selected web pages
                 |
                 v
        Text extraction and cleaning
                 |
                 v
     Fixed-size and structure-aware chunks
                 |
                 v
     OpenAI text-embedding-3-small
                 |
                 v
       Separate ChromaDB collections
                 |
                 v
 Semantic retrieval and metadata reranking
                 |
                 v
       Top five evidence passages
                 |
                 v
        OpenAI answer generation
                 |
                 v
 Answer, citations, evidence and evaluation logs
```

Questions and chunks are represented as embeddings, and ChromaDB retrieves chunks using vector similarity. This means that a question does not need to use exactly the same wording as the source, although indirect or highly contextual questions can still be difficult.

## Retrieval strategies

### Fixed-size

The fixed-size baseline divides cleaned text into consistent word windows with overlap. It is simple and robust, but it can divide one policy section across several chunks.

### Structure-aware

The structure-aware method uses headings and section boundaries before applying size limits. This is intended to preserve more of the organisation and meaning of a policy document.

### Hybrid

The hybrid strategy retrieves candidates from both indexes, combines and removes duplicate evidence, and then applies metadata-aware reranking. It sends one final top-five evidence set to the generation model rather than generating two separate answers.

Hybrid retrieval became the default application strategy after it produced the strongest retrieval results in my evaluations.

## Technology used

### Frontend

- React
- TypeScript
- Vite
- Lucide icons

### Backend and retrieval

- Python 3.12.8
- FastAPI and Uvicorn
- PyMuPDF for PDF extraction
- ChromaDB for local vector storage
- OpenAI `text-embedding-3-small` for embeddings
- OpenAI models for answer generation and LLM-assisted evaluation

### Storage, evaluation and deployment

- SQLite for application data and local logs
- Supabase as an optional audit mirror
- pandas and openpyxl for evaluation workbooks
- Git and GitHub for version control
- Render for the deployed demonstration

## Repository structure

```text
MSc-Kraken-Rag/
|-- backend/
|   |-- app/                    FastAPI application and RAG pipeline
|   |-- evaluation/
|   |-- requirements.txt
|   `-- .env.example
|-- data/
|   `-- kraken_PDFs/            Public PDF corpus
|-- frontend/
|   |-- src/                    React and TypeScript interface
|   |-- package.json
|   `-- package-lock.json
`-- README.md
```

Generated databases, ChromaDB indexes, local logs, downloaded web content and API keys should not be committed.

## Requirements

The project was developed and tested with:

- Python 3.12.8;
- Node.js 20 or later;
- npm;
- an OpenAI API key; and
- Git.

The instructions below use Windows PowerShell because that is the environment in which I developed the project. The Python and npm commands can also be adapted for macOS or Linux.

## Running the project locally

### 1. Clone the repository

```powershell
git clone https://github.com/Andre-Edwards/MSc-Kraken-RAG-Submission.git
cd MSc-Kraken-RAG-Submission
```

### 2. Set up the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `backend/.env` and provide at least:

```env
OPENAI_API_KEY=replace-with-your-own-key
APP_SECRET_KEY=replace-with-a-long-random-secret
KRAKEN_PDF_DIR=data/kraken_PDFs
CHROMA_DIR=storage/chroma
SQLITE_PATH=storage/app.db
```

Do not commit the completed `.env` file.

### 3. Build the local indexes

From the `backend` directory, run:

```powershell
python -m app.scripts.ingest --force
```

This extracts the PDF text, creates both chunking representations, generates embeddings and writes the ChromaDB indexes.

### 4. Start the backend

```powershell
python -m uvicorn app.main:app --reload --port 8000
```

The health endpoint should return a successful response at:

```text
http://127.0.0.1:8000/api/health
```

Open that address before starting the frontend. If it does not return
`{"ok":true,"app":"kraken-rag-assistant"}`, resolve the backend error shown in
the terminal first.


### 5. Set up the frontend

Open a second terminal:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
```

The copied local environment file points the frontend to:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Then start the frontend:

```powershell
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The local accounts are controlled by the corresponding values in `backend/.env`.
The supplied local demonstration values are:

```text
User:  demo@example.com / demo1234
Admin: admin@example.com / admin1234
```

These credentials are for local reproduction only. Change the passwords and
`APP_SECRET_KEY` before deploying the application publicly.

## Adding selected web pages

The crawler is intended for selected public Kraken pages rather than an unrestricted crawl of the whole website.

From the `backend` directory:

```powershell
python -m app.scripts.scrape_web --seed https://www.kraken.com/legal --max-pages 10 --max-depth 1
python -m app.scripts.ingest --force
```

The crawler checks `robots.txt`, stays within configured Kraken domains and records the source URL with the extracted page. The local web-page output is treated as generated corpus data and is not required to be committed.

## How I evaluated it

I did not treat one automated score as proof that an answer was correct. Retrieval and generated answers were checked separately.

For retrieval, I created labelled question sets by reading the source material and recording which documents or passages should be relevant. Each strategy was then tested using the same questions and a top-five cutoff.

The main retrieval measures were:

- **Precision@5:** how many of the five retrieved items were relevant;
- **Recall@5:** how much of the labelled relevant material appeared in those five items; and
- **F1@5:** a balance between precision and recall.

The headline F1@5 results were:

| Evaluation | Fixed-size | Structure-aware | Hybrid |
| Original source-level set, n=59 | 0.813 | 0.789 | **0.889** |
| Expanded source-level set, n=46 | 0.764 | 0.798 | **0.879** |
| Chunk-level set, n=20 | 0.651 | 0.567 | **0.731** |

These results support using hybrid retrieval in this prototype, but they do not prove that every generated answer is correct.

I therefore also reviewed answers against the corpus using pass, partial and fail verdicts. The LLM-as-judge assessed groundedness, citation quality, relevance, completeness and clarity on a five-point scale, but I treated it as a supporting signal rather than ground truth. Human checks were especially important when an answer sounded convincing but missed a condition or cited the wrong passage.

Early usability testing also led to practical changes. These included metadata reranking, clearer evidence and scoring labels, a more polished interface, recent chats, an administrator dashboard, structured feedback and an insights page for reviewing repeated failures and possible corpus gaps.

Because generation models are probabilistic, exact answer wording and LLM-as-judge scores may vary when an evaluation is rerun. The labelled retrieval results and manually verified outputs are included so that the reported conclusions can still be inspected.

The three labelled Excel workbooks in `backend/evaluation/gold_standards/`
are the human-readable source of truth for the evaluation labels. Their JSON
versions can be regenerated or checked without calling OpenAI:

```powershell
cd backend
python evaluation/build_gold_standards.py
python evaluation/build_gold_standards.py --check
```

These JSON files drive the retrieval metrics. The answer-audit workbooks in
`backend/evaluation/audits/` are separate human assessments and are not needed
to calculate Precision@5, Recall@5 or F1@5.

The submitted row-level results can be checked without an OpenAI API key:

```powershell
cd backend
python evaluation/verify_evaluation_scores.py
```

The complete commands and the distinction between verifying saved results and
performing a new retrieval run are documented in
`backend/evaluation/README.md` and `REPRODUCIBILITY.md`.

## A quick reproduction check

After indexing and starting both services:

1. Sign in using the local demo credentials configured in `backend/.env`.
2. Select hybrid retrieval in the administrator settings.
3. Ask: `What risks are described in the MiCAR Risk Disclosure?`
4. Confirm that the response includes citation markers.
5. Open the evidence panel and confirm that the retrieved document title, excerpt and retrieval information are visible.

The exact wording can vary, but the response should primarily use evidence from the MiCAR Risk Disclosure.

## Known limitations

- The system uses a limited public case-study corpus rather than internal company knowledge.
- Image-only scanned PDFs require OCR. PyMuPDF cannot extract meaningful text from them by itself.
- Web pages can change after they have been crawled, so corpus dates and versions matter.
- Vector similarity does not guarantee that indirect questions will retrieve the intended policy relationship.
- The hybrid strategy performs more retrieval work than either individual strategy.
- LLM-as-judge required calibration and should not replace human review.
- The evaluation sets and employee-testing sample are useful but still limited in size.
- OpenAI API use creates a small running cost and requires an internet connection.

## Live demonstration

The deployed prototype is available at:

https://msc-kraken-rag-1.onrender.com/

```text
User:  demo@example.com / demo1234
Admin: admin@example.com / admin123
tester: tester15@example.com / tester15
```

## Data and privacy

The application is designed for public demonstration material. Users should not enter confidential information, personal customer data, internal policies or sensitive business information.

## Author

Andre Edwards
MSc Artificial Intelligence and Data Science
2026
