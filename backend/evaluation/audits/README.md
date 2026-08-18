# Human audit workbooks

These workbooks contain the manual review material used alongside the automated
retrieval metrics. They are not inputs to Precision@5, Recall@5 or F1@5.

## Original 60-question answer audit

`Old_60 Questions Audit.xlsx` compares the structure-aware and fixed-size
answers. All 60 questions align with the original gold-standard question bank.
The recorded verdicts are 55 pass, four partial and one fail.

## Expanded 50-question answer audit

`expanded_50 Questions Audit.xlsx` compares structure-aware, fixed-size and
hybrid answers against the expected answer and expected source. All 50 question
IDs and question texts align with the expanded gold standard. Reviewer notes
were optional, so some passing rows do not contain a written note.

## Chunk-level audit

`Chunk level 20 questions (from expanded) Audit.xlsx` contains 300 judgements:
20 questions multiplied by three retrieval strategies and five ranks. Each row
records source matching, keyword-group coverage, the relevance decision and the
retrieved excerpt.

The relevant-chunk totals are:

| Strategy | Relevant top-five chunks |
|---|---:|
| Structure-aware | 44 |
| Fixed-size | 53 |
| Hybrid | 61 |

These totals produce the submitted chunk-level Precision@5 and F1@5 values.
The workbook records the reviewed retrieval run; a new embedding/index run can
change the order of tied or near-tied chunks even when aggregate metrics remain
the same.
