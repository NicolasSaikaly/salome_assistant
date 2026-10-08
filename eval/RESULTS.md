# Retrieval evaluation results

Eval set: 26 questions on the SALOME Mesh (SMESH) docs, each labelled with the
page(s) that contain the answer (`eval/questions.jsonl`).
Corpus: SMESH HTML docs - first the online docs (330 pages -> 761 chunks), then the docs
shipped with SALOME 9.16 (178 pages -> 796 chunks, same scores on config 6).
Hardware: laptop CPU only (no GPU), 16 GB RAM.

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

## Query rewriting by the LLM (27 questions)

A 27th question was added after a real failure in the app
("How can I view the interior of a mesh?" -> `clipping.html`).

| Config (bge-base, no rerank) | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|
| Original question only | 74 % | **89 %** | **0.79** | 0.1 s |
| + rewritten query (qwen2.5:3b), fused with RRF | 74 % | 78 % | 0.76 | 2.7 s |

- **Rewriting hurt on English questions**: it fixed none of the failures and broke three.
  The 3B model does not know SALOME's vocabulary (it rewrote "view the interior" as
  "internal mesh visualization", not "clipping"), and the first prompt's examples biased
  it towards API words ("smeshBuilder ... method", even an invented "AddLayer"), which
  pulled the search towards API reference pages.
- **Decision**: rewriting is now *automatic*, only for follow-up questions and non-English
  (e.g. French) questions, where the original wording cannot work. Prompt fixed (no API
  words in examples, "never invent function names"). The French case still needs its own
  evaluation set.

## French questions (27 questions translated, `eval/questions_fr.jsonl`)

The embedding model is English-only and the docs are in English.

| Config (bge-base, no rerank) | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|
| French question as is | 30 % | 52 % | 0.37 | 0.1 s |
| French question rewritten in English by qwen2.5:3b (auto mode) | **52 %** | **78 %** | **0.62** | 2.4 s |
| *(reference: English questions)* | *74 %* | *89 %* | *0.79* | *0.1 s* |

- **Rewriting pays off here**: +26 points of hit@5 (7 more questions answered), which
  justifies the *auto* mode: rewrite French and follow-up questions, not English ones.
- Without rewriting, French queries are pulled towards the few French pages shipped
  with SALOME (`usage_outil.html`, `presentation_base.html`), whatever the topic.
- Remaining gap vs English (78 % vs 89 %) comes from translation slips
  ("rapport d'aspect" -> "aspect report") and the same vocabulary gaps as in English.

## Caveats

26 questions is small: one question = ~4 points, so differences under ~8 points
are not conclusive. The set was written by one person who knows the docs.

Chosen default: **config 6** (bge-base-en-v1.5, no reranking).
