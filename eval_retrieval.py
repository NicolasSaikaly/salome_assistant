"""Step 5a - Measure retrieval quality on a set of questions with known answers.

For each question in eval/questions.jsonl we know which doc page(s) contain
the answer. We check where that page shows up in the retrieved results:

    hit@1     = % of questions where a correct page is ranked first
    hit@5     = % of questions where a correct page is in the top 5
                (what matters most: the LLM reads these 5 passages)
    MRR       = mean of 1/rank of the first correct page (0 if absent) -> 1.0 is perfect
    recall@20 = % of questions where a correct page is among the 20 candidates
                of the embedding search. The reranker can only re-order these 20:
                if the page is not there, only a better first stage can fix it.

No LLM is involved, so this runs quickly and can be re-run after every change.

Usage:  python eval_retrieval.py
        EMBED_MODEL=BAAI/bge-base-en-v1.5 python eval_retrieval.py
        RERANK=0 python eval_retrieval.py
"""

import json
import time
from pathlib import Path

import rag
from rag import CANDIDATES, retrieve

QUESTIONS_FILE = Path("eval/questions.jsonl")
K = 5


def page_of(url: str) -> str:
    """https://.../SMESH/creating_groups.html#standalone-group -> creating_groups.html"""
    return url.split("#")[0].rsplit("/", 1)[-1]


def first_rank(pages, expected):
    return next((i for i, p in enumerate(pages, 1) if p in expected), None)


def main():
    items = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")
             if line.strip()]
    retrieve("warm-up")  # load the models before timing
    hit1 = hit5 = rr_sum = recall = 0
    misses = []
    start = time.time()

    for item in items:
        expected = set(item["pages"])
        pages = [page_of(h["url"]) for h in retrieve(item["question"], k=K)]
        rank = first_rank(pages, expected)
        # Where is the right page in the raw embedding search (before reranking)?
        candidates = [page_of(h["url"])
                      for h in retrieve(item["question"], k=CANDIDATES, rerank=False)]
        cand_rank = first_rank(candidates, expected)
        recall += cand_rank is not None

        if rank:
            hit1 += rank == 1
            hit5 += 1
            rr_sum += 1 / rank
        else:
            misses.append((item["question"], expected, pages, cand_rank))
        print(f"{'OK  ' if rank else 'MISS'} rank={rank or '-':<2} "
              f"(embedding rank={cand_rank or f'>{CANDIDATES}'})  {item['question']}")

    n = len(items)
    per_q = (time.time() - start) / n
    setup = (f"rerank={rag.RERANK_MODEL} | max/page={rag.MAX_PER_PAGE or 'off'}"
             if rag.RERANK else "no rerank")
    print(f"\n[{rag.EMBED_MODEL} | {setup}]")
    print(f"{n} questions | hit@1 = {hit1 / n:.0%} | hit@{K} = {hit5 / n:.0%} | "
          f"MRR = {rr_sum / n:.2f} | recall@{CANDIDATES} = {recall / n:.0%} | "
          f"{per_q:.1f}s/question")

    if misses:
        print("\nMisses (expected -> got):")
        for question, expected, pages, cand_rank in misses:
            print(f"- {question}\n    expected {sorted(expected)}"
                  f"  (embedding rank: {cand_rank or f'not in top {CANDIDATES}'})\n    got      {pages}")


if __name__ == "__main__":
    main()
