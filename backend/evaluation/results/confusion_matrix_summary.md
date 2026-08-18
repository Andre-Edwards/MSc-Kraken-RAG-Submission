# RAG Retrieval Confusion Matrix Summary

These matrices are source/document-level because the gold standard labels expected documents or URLs, not every relevant chunk. Precision@K, Recall@K and F1@K remain the main retrieval metrics; the matrices provide supporting detail for the dissertation.

### old_60 - core - Structure-aware

Questions: 59 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 57 | 2 |
| Gold source not relevant | 59 | 4307 |

Reported macro @K: Precision 0.722, Recall 0.966, F1 0.789.
Source-matrix micro: Precision 0.491, Recall 0.966, F1 0.651.
Top-K chunk counts: relevant 213, non-relevant 82.

### old_60 - core - Fixed-size

Questions: 59 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 58 | 1 |
| Gold source not relevant | 51 | 4315 |

Reported macro @K: Precision 0.753, Recall 0.983, F1 0.813.
Source-matrix micro: Precision 0.532, Recall 0.983, F1 0.690.
Top-K chunk counts: relevant 222, non-relevant 73.

### old_60 - core - Hybrid

Questions: 59 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 59 | 0 |
| Gold source not relevant | 35 | 4331 |

Reported macro @K: Precision 0.841, Recall 1.000, F1 0.889.
Source-matrix micro: Precision 0.628, Recall 1.000, F1 0.771.
Top-K chunk counts: relevant 248, non-relevant 47.

### expanded_50 - core - Structure-aware

Questions: 46 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 45 | 1 |
| Gold source not relevant | 52 | 3352 |

Reported macro @K: Precision 0.717, Recall 0.978, F1 0.798.
Source-matrix micro: Precision 0.464, Recall 0.978, F1 0.629.
Top-K chunk counts: relevant 165, non-relevant 65.

### expanded_50 - core - Fixed-size

Questions: 46 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 45 | 1 |
| Gold source not relevant | 54 | 3350 |

Reported macro @K: Precision 0.678, Recall 0.978, F1 0.764.
Source-matrix micro: Precision 0.455, Recall 0.978, F1 0.621.
Top-K chunk counts: relevant 156, non-relevant 74.

### expanded_50 - core - Hybrid

Questions: 46 | K: 5 | Source universe: 75

| Gold / retrieved source | Retrieved source | Not retrieved source |
|---|---:|---:|
| Gold source relevant | 45 | 1 |
| Gold source not relevant | 29 | 3375 |

Reported macro @K: Precision 0.830, Recall 0.978, F1 0.879.
Source-matrix micro: Precision 0.608, Recall 0.978, F1 0.750.
Top-K chunk counts: relevant 191, non-relevant 39.
