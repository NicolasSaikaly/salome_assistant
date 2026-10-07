"""Step 5a - Measure retrieval quality on a set of questions with known answers.

For each question in eval/questions.jsonl we know which doc page(s) contain
the answer. We check where that page shows up in the retrieved results:

    hit@1  = % of questions where a correct page is ranked first
    hit@5  = % of questions where a correct page is in the top 5
    MRR    = mean of 1/rank of the first correct page (0 if absent) -> 1.0 is perfect

No LLM is involved, so this runs in a few seconds and can be re-run after
every change (chunk size, embedding model, top-k...).

Usage:  python eval_retrieval.py
"""

import json
from pathlib import Path

import rag
from rag import retrieve

QUESTIONS_FILE = Path("eval/questions.jsonl")
K = 5


def page_of(url: str) -> str:
    """https://.../SMESH/creating_groups.html#standalone-group -> creating_groups.html"""
    return url.split("#")[0].rsplit("/", 1)[-1]


def main():
    items = [json.loads(line) for line in QUESTIONS_FILE.open(encoding="utf-8")
             if line.strip()]
    hit1 = hit5 = rr_sum = 0
    misses = []

    for item in items:
        expected = set(item["pages"])
        pages = [page_of(h["url"]) for h in retrieve(item["question"], k=K)]
        rank = next((i for i, p in enumerate(pages, 1) if p in expected), None)
        if rank:
            hit1 += rank == 1
            hit5 += 1
            rr_sum += 1 / rank
        else:
            misses.append((item["question"], expected, pages))
        print(f"{'OK  ' if rank else 'MISS'} rank={rank or '-'}  {item['question']}")

    n = len(items)
    setup = f"rerank={rag.RERANK_MODEL}" if rag.RERANK else "no rerank"
    print(f"\n[{rag.EMBED_MODEL} | {setup}]")
    print(f"{n} questions | hit@1 = {hit1 / n:.0%} | hit@{K} = {hit5 / n:.0%} | MRR = {rr_sum / n:.2f}")

    if misses:
        print("\nMisses (expected -> got):")
        for question, expected, pages in misses:
            print(f"- {question}\n    expected {sorted(expected)}\n    got      {pages}")


if __name__ == "__main__":
    main()
