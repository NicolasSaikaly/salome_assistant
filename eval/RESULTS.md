# Evaluation log

Question sets: `questions.jsonl` (English) and `questions_fr.jsonl` (same questions in
French), each question labelled with the doc page(s) that contain the answer.
Corpus: SMESH HTML docs, first the online version (330 pages, 761 chunks), then the
docs shipped with SALOME 9.16 (178 pages, 796 chunks; config 6 gives the same scores).
Hardware: laptop CPU, 16 GB RAM, no GPU.

## 1. Embeddings and reranking (26 questions)

| # | Embedding model | Reranker | Candidates | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|---|---|---|
| 1 | bge-small-en-v1.5 | none | - | 65 % | 81 % | 0.72 | < 0.1 s |
| 2 | bge-small-en-v1.5 | ms-marco-MiniLM-L-6-v2 | 20 | 73 % | 77 % | 0.75 | - |
| 3 | bge-small-en-v1.5 | bge-reranker-base | 20 | 65 % | 85 % | 0.74 | 13.3 s |
| 4 | bge-small-en-v1.5 | bge-reranker-base | 10 | 65 % | 85 % | 0.74 | 6.7 s |
| 5 | bge-base-en-v1.5 | bge-reranker-base | 20 | 62 % | 92 % | 0.75 | 14.6 s |
| 6 | bge-base-en-v1.5 | none | - | 77 % | 92 % | 0.82 | 0.1 s |

Configs 3 to 5 keep at most one chunk per page in the top 5.

- With bge-small the cross-encoder helped (hit@5 81 % -> 85 %). With bge-base it made
  things worse (hit@1 77 % -> 62 %, MRR 0.82 -> 0.75) and was about 150x slower. It
  tends to rank the big API reference pages (`smeshBuilder.html`, `modules.html`) above
  the user-guide page that answers the question.
- recall@20 was already 96 %: most misses were ranking problems, not missing candidates.
- Reranking 10 candidates instead of 20 gave the same scores for half the time, since
  every correct page was within the top 9 of the embedding search.
- Remaining misses are vocabulary gaps: "combine two meshes" vs "Build Compound",
  "finer mesh on one part" vs "sub-mesh".

Kept: config 6.

## 2. Query rewriting (27 questions)

A 27th question was added after a wrong answer in the UI
("How can I view the interior of a mesh?", expected `clipping.html`).

| bge-base, no rerank | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|
| Original question | 74 % | 89 % | 0.79 | 0.1 s |
| + query rewritten by qwen2.5:3b, merged with RRF | 74 % | 78 % | 0.76 | 2.7 s |

- Rewriting fixed none of the misses and broke three. The 3B model does not know
  SALOME's vocabulary ("view the interior" became "internal mesh visualization", not
  "clipping"), and the examples in my first prompt pushed it towards API words, even an
  invented "AddLayer", which pulled the search towards API reference pages.
- Rewriting is now only used for follow-up questions and non-English questions. The
  prompt examples no longer contain API names.

## 3. French questions (27 questions)

| bge-base, no rerank | hit@1 | hit@5 | MRR | Time / question |
|---|---|---|---|---|
| French question as is | 30 % | 52 % | 0.37 | 0.1 s |
| Rewritten in English by qwen2.5:3b | 52 % | 78 % | 0.62 | 2.4 s |
| (English questions, for reference) | 74 % | 89 % | 0.79 | 0.1 s |

- Here rewriting helps: +26 points of hit@5.
- Without it, French questions are drawn to the few French pages shipped with SALOME
  (`usage_outil.html`, `presentation_base.html`) whatever the topic.
- The remaining gap with English comes from translation slips ("rapport d'aspect" ->
  "aspect report") and the same vocabulary gaps.

## 4. Answers (`eval_answers.py`, first 8 questions)

Checks on the generated answers: *intro* = starts with a one-sentence answer;
*full ex.* = points to the complete example; *no invented API* = every function called
in the code exists somewhere in the docs; *sourced API* = it also appears in the 5
passages given to the model.

| LLM | Intro | Full ex. | No invented API | Sourced API | Time / answer |
|---|---|---|---|---|---|
| qwen2.5:3b | 50 % | 100 % | 75 % | 88 % | 60 s |
| qwen2.5:7b | 100 % | 100 % | 100 % | 88 % | 35 s |

- On a first run the 3B invented `FaceGroups`, `MergingNodes` (the real one is
  `MergeNodes`), `FindHole` and `Add`. The 7B invented none.
- The first version of the check only looked at the 5 passages and flagged the 7B's
  `MergeNodes`, a real function that just wasn't in them. It now checks the whole docs
  and reports "not in the passages" separately.
- Timings depend on the machine load: the 7B took 116 s per answer on a first run and
  35 s later, after removing needless model reloads between calls (`NUM_CTX` in rag.py).
- A full example answer in the prompt was copied word for word by the 3B on an
  unrelated question; the prompt now only contains an empty layout.

Kept: qwen2.5:7b for answers, qwen2.5:3b for query rewriting.

## Caveats

These sets are small (one question is about 4 points), so differences of a few points
are not meaningful. The questions were written by someone who knows the docs.
