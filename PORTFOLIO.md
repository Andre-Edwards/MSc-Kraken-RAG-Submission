# Project walkthrough

Kraken Policy Assistant is my MSc dissertation prototype. It explores how document chunking and retrieval affect evidence quality in a policy-document assistant. I wrote substantial parts of the code and evaluated the retrieval strategies. The system uses public documents and has no affiliation with Kraken.

## Review the project in five minutes

1. Read the [retrieval pipeline](backend/app/services/rag.py) to see how evidence is selected for an answer.
2. Compare [fixed-size chunking](backend/app/services/chunking/fixed_size.py) with [structure-aware chunking](backend/app/services/chunking/section_aware.py).
3. Inspect the [chat API](backend/app/api/chat.py) and [frontend source](frontend/src) for the user-facing integration.
4. Read the [evaluation definitions](backend/evaluation/README.md) and inspect the saved per-question results.
5. Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) to verify the saved results or run the application.

## What I investigated

Policy answers can sound plausible while relying on incomplete or irrelevant evidence. I compared fixed-size word windows, structure-aware chunks and a hybrid of both indexes. The application exposes citations and retrieved evidence so users can inspect the source material.

| Retrieval evaluation | Questions in metric subset | Fixed-size F1@5 | Hybrid F1@5 |
|---|---:|---:|---:|
| Original source-level set | 59 | 0.813 | 0.889 |
| Expanded source-level set | 46 | 0.764 | 0.879 |
| Chunk-level set | 20 | 0.651 | 0.731 |

These are macro retrieval metrics on small labelled datasets. They do not measure overall answer accuracy, production reliability or performance on an unseen customer corpus. The evaluation documentation explains exclusions and the separate answer audits.

## Engineering scope

- Python/FastAPI API with PDF ingestion, selected web-page crawling and retrieval configuration.
- React/TypeScript frontend with cited answers, evidence inspection and separate user/admin workflows.
- ChromaDB indexes and SQLite application storage.
- Feedback, chat logs and an insights dashboard for reviewing failures and possible corpus gaps.
- A Render demonstration and documented local setup.

## Verify without model calls

From the repository root, with Python 3.12:

```sh
python backend/evaluation/verify_evaluation_scores.py
```

This recomputes metrics from saved rows using the Python standard library. It requires no API key and makes no model calls. A fresh retrieval run is a separate procedure requiring embeddings, indexes and an API key.

## Scope and limitations

This is an academic prototype, not a customer-deployed product. There are no production adoption claims. The small public corpus limits generalization; scanned documents need OCR; generation and judge scores can vary between runs. The hybrid strategy also adds retrieval work. The [main README](README.md#known-limitations) records further limitations.

Useful next engineering steps would be independent question sets, measured latency and cost, stronger automated integration tests, and deployment hardening. These are future improvements, not completed features.
