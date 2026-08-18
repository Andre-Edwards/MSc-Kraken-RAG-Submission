# Evaluation

The evaluation separates retrieval quality from generated-answer quality.
The JSON files in `gold_standards/` label expected sources or answer-bearing
passages. The Excel files in `audits/` contain the separate human answer audits.

## Submitted datasets

| Dataset | Labelled questions | Main metric subset |
|---|---:|---:|
| Original PDF source level | 60 | 59 |
| Expanded PDF and web source level | 50 | 46 |
| Chunk level | 20 | 20 |

Q060 in the original set is excluded from the main retrieval aggregate because
its scanned vendor document requires OCR. Four expanded-set boundary questions
have no expected retrieval source and are therefore excluded from source-level
Precision@5, Recall@5 and F1@5. EQ050 is a refusal/caution question with a
relevant rewards page, so it remains in the retrieval subset.

## Build the gold-standard JSON

The three Excel workbooks are the human-readable source of truth:

```powershell
cd backend
python evaluation/build_gold_standards.py
python evaluation/build_gold_standards.py --check
```

## Verify the submitted results

```powershell
python evaluation/verify_evaluation_scores.py
```

The verifier recomputes all macro summaries from the saved per-question rows.
No OpenAI key, Chroma index or model call is required.

To regenerate the source-level confusion-matrix summaries:

```powershell
python evaluation/display_confusion_matrices.py `
  --metrics-dir evaluation/results `
  --output-dir evaluation/runs/confusion_matrices
```

## Perform a new retrieval run

A new run requires an OpenAI API key and populated fixed-size and
structure-aware Chroma collections. Build them first with:

```powershell
python -m app.scripts.ingest --force
```

Then run both source-level datasets:

```powershell
python evaluation/run_gold_standard_evaluation.py `
  --dataset evaluation/gold_standards/old_60_gold_standard.json `
  --dataset-name old_60_gold_standard `
  --mode retrieval-only --top-k 5 `
  --output evaluation/results/original_source_level_at5.json

python evaluation/run_gold_standard_evaluation.py `
  --dataset evaluation/gold_standards/expanded_gold_standard_50.json `
  --dataset-name expanded_gold_standard_50 `
  --mode retrieval-only --top-k 5 `
  --output evaluation/results/expanded_source_level_at5.json
```

The chunk-level evaluation reuses the expanded run's saved top-five hits:

```powershell
python evaluation/run_chunk_level_evaluation.py `
  --evaluation-json evaluation/results/expanded_source_level_at5.json `
  --gold-standard evaluation/gold_standards/chunk_gold_standard_20.json `
  --processed-dir storage/processed `
  --top-k 5 `
  --output evaluation/results/chunk_level_at5.json
```

## Metric definitions

At source level, a retrieved chunk is relevant when its PDF filename, web-page
title or URL matches the question's labelled primary source. Precision@5 is the
proportion of the five retrieved chunks from that source. Recall@5 records
whether the labelled source was retrieved. Macro F1@5 is computed per question
and then averaged.

At chunk level, relevance requires both a source match and the minimum number of
human-defined answer-bearing keyword groups. Passage Recall@5 is one when at
least one relevant chunk appears in the top five and zero otherwise.

The source-level confusion matrices count unique retrieved sources, so their
micro precision and recall are supporting diagnostics rather than replacements
for the macro top-K metrics.

## Submitted results

| Dataset | Strategy | Precision@5 | Recall@5 | F1@5 |
|---|---|---:|---:|---:|
| Original | Structure-aware | 0.722 | 0.966 | 0.789 |
| Original | Fixed-size | 0.753 | 0.983 | 0.813 |
| Original | Hybrid | 0.841 | 1.000 | 0.889 |
| Expanded | Structure-aware | 0.717 | 0.978 | 0.798 |
| Expanded | Fixed-size | 0.678 | 0.978 | 0.764 |
| Expanded | Hybrid | 0.830 | 0.978 | 0.879 |

Chunk-level F1@5 is 0.567 for structure-aware, 0.651 for fixed-size and 0.731
for hybrid retrieval. The complete rows and summaries are in `results/`.

These retrieval metrics do not establish answer correctness. The separate human
audits and LLM-as-judge outputs assess grounding, citation quality, relevance,
completeness and clarity.
