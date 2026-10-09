"""Check the ANSWERS written by the LLM (eval_retrieval.py only checks the search).

For a handful of questions, this script runs the full pipeline (search + answer) and
runs three automatic checks on each answer:

  intro      the answer starts with a sentence, not directly with a heading or a list
  full_ex    the answer ends with "Full example: [n]" when it shows Python code
  api        every function/method called in the code exists somewhere in the docs.
             A name that appears nowhere was invented (hallucinated).
  grounded   every function called also appears in the passages given to the model
             (otherwise it comes from the model's own knowledge: maybe right, not sourced)

All answers are also saved to eval/answers/<model>_<date>.md so you can read them.
Use it after every prompt change, so a change is judged on several questions, not one.

Usage:  python eval_answers.py                  (8 first questions, default model)
        LLM_MODEL=qwen2.5:7b python eval_answers.py
        N=27 python eval_answers.py             (all questions, slow on CPU)
Needs Ollama running.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import rag

QUESTIONS_FILE = Path(os.environ.get("QUESTIONS", "eval/questions.jsonl"))
N = int(os.environ.get("N", "8"))
OUT_DIR = Path("eval/answers")


def check(answer: str, hits):
    first_line = next((l.strip() for l in answer.splitlines() if l.strip()), "")
    codes = rag.code_blocks(answer)
    invented = rag.invented_calls(answer, hits)
    ungrounded = rag.ungrounded_calls(answer, hits)
    return {
        "intro": bool(first_line) and not first_line.startswith(("#", "**In", "1.", "-", "```")),
        "full_ex": (not codes) or bool(re.search(r"Full example:?\s*\[\d+\]", answer)),
        "api": not invented,          # no function that exists nowhere in the docs
        "grounded": not ungrounded,   # every function also appears in the given passages
        "invented": invented,
        "ungrounded": ungrounded,
    }


def main():
    items = [json.loads(l) for l in QUESTIONS_FILE.open(encoding="utf-8") if l.strip()][:N]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"{rag.LLM_MODEL.replace(':', '-')}_{datetime.now():%Y%m%d-%H%M}.md"
    totals = {"intro": 0, "full_ex": 0, "api": 0, "grounded": 0}
    report = [f"# Answers - {rag.LLM_MODEL} - {datetime.now():%Y-%m-%d %H:%M}\n"]
    start = time.time()

    for i, item in enumerate(items, 1):
        q = item["question"]
        hits = rag.retrieve(q)
        t0 = time.time()
        answer = "".join(rag.generate(q, hits))
        res = check(answer, hits)
        for key in totals:
            totals[key] += res[key]
        flags = " ".join(f"{k}={'ok' if res[k] else 'FAIL'}" for k in totals)
        extra = (f"  invented: {', '.join(res['invented'])}" if res["invented"] else "") + \
                (f"  not in passages: {', '.join(res['ungrounded'])}" if res["ungrounded"] else "")
        print(f"[{i}/{len(items)}] {time.time() - t0:4.0f}s  {flags}{extra}  {q}")
        report.append(f"## {q}\n\n{flags}{extra}\n\n{answer}\n\n---\n")

    n = len(items)
    summary = " | ".join(f"{k} {v / n:.0%}" for k, v in totals.items())
    print(f"\n[{rag.LLM_MODEL}] {n} answers | {summary} | "
          f"{(time.time() - start) / n:.0f}s/answer")
    print(f"Answers saved to {out_file}")
    report.insert(1, f"**{summary}**\n")
    out_file.write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
