# Retrieval evaluation results

Eval set: 26 questions on the SALOME Mesh (SMESH) docs, each labelled with the
page(s) that contain the answer (`eval/questions.jsonl`).
Corpus: 330 HTML pages -> 761 chunks. Hardware: laptop CPU only (no GPU), 16 GB RAM.

| # | Embedding model | Reranker | Candidates | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|---|---|---|
| 1 | bge-small-en-v1.5 | none | - | 65 % | 81 % | 0.72 | < 0.1 s |
| 2 | bge-small-en-v1.5 | ms-marco-MiniLM-L-6-v2 | 20 | 73 % | 77 % | 0.75 | - |
| 3 | bge-small-en-v1.5 | bge-reranker-base | 20 | 65 % | 85 % | 0.74 | 13.3 s |
| 4 | bge-small-en-v1.5 | bge-reranker-base | 10 | 65 % | 85 % | 0.74 | 6.7 s |
| 5 | bge-base-en-v1.5 | bge-reranker-base | 20 | 62 % | 92 % | 0.75 | 14.6 s |
| **6** | **bge-base-en-v1.5** | **none** | - | **77 %** | **92 %** | **0.82** | **0.1 s** |

Configs 3-6 keep at most 1 chunk per page in the top 5 when reranking.

## Findings

- **A stronger embedding model beat reranking.** With bge-small, a cross-encoder
  reranker helped (hit@5 81 % -> 85 %). With bge-base, the same reranker *hurt*
  (hit@1 77 % -> 62 %, MRR 0.82 -> 0.75) while being ~150x slower: the generic
  reranker tends to promote the large API reference pages (`smeshBuilder.html`,
  `modules.html`) over the user-guide page that actually answers the question.
- **recall@20 was 96 % from the start**: most failures were ranking problems,
  not missing candidates - measuring it avoided optimizing the wrong stage.
- **Reranking 10 candidates instead of 20** kept the same quality at half the cost,
  because every correct page was already within the top 9 of the embedding search.
- **Remaining failures** are vocabulary gaps: "combine two meshes" vs the doc's
  "Build Compound", "finer mesh on one part" vs "sub-mesh".

## Caveats

26 questions is small: one question = ~4 points, so differences under ~8 points
are not conclusive. The set was written by one person who knows the docs.

Chosen default: **config 6** (bge-base-en-v1.5, no reranking).
