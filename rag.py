"""Core of the assistant: retrieval (step 2b) + generation (step 3).

    retrieve(question)        -> the doc chunks closest to the question
    generate(question, hits)  -> the LLM answer, written from those chunks only

From the terminal:
    python rag.py "How do I create a group of faces?"
    python rag.py --search "How do I create a group of faces?"   (retrieval only)
"""

import json
import os
import sys
import time
from functools import lru_cache

import chromadb
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---- Settings shared by index.py, rag.py and later the app ----
# Embedding model, changeable without editing the code:
#   EMBED_MODEL=BAAI/bge-base-en-v1.5 python index.py
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
# One database per embedding model, so several models can be compared side by side
DB_DIR = "chroma_db/" + EMBED_MODEL.replace("/", "__")
COLLECTION = "smesh_docs"
# bge models work better when the *question* (not the documents) gets this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOP_K = 5

# Reranking: fetch more candidates with embeddings (fast, approximate), then let a
# cross-encoder re-read each (question, passage) pair and re-order them (slower, precise).
# Off by default: it did not help with bge-base (see eval/RESULTS.md).
# Turn it on to compare:  RERANK=1 python eval_retrieval.py
RERANK = os.environ.get("RERANK", "0") != "0"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
CANDIDATES = int(os.environ.get("CANDIDATES", "20"))
# Diversity: at most this many chunks from the same page in the final top-k,
# so one big page cannot fill all the slots.  MAX_PER_PAGE=0 disables the rule.
MAX_PER_PAGE = int(os.environ.get("MAX_PER_PAGE", "1"))

OLLAMA_URL = "http://localhost:11434/api/chat"
# Change the model without editing the code:  LLM_MODEL=qwen2.5:7b python rag.py "..."
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
SYSTEM_PROMPT = """You are an assistant for the SALOME Mesh module (SMESH).
Answer the question using ONLY the documentation excerpts provided.
Rules:
- Cite the excerpts you use with their number, like [1] or [2][3].
- If the question is about scripting, give a short Python example based on the excerpts
  (smeshBuilder API). Never invent functions that do not appear in the excerpts.
- If the excerpts do not contain the answer, say so clearly instead of guessing.
- Be concise and practical."""


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once (it takes a few seconds)."""
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_collection(COLLECTION)


def embed(texts, is_query=False):
    """Turn texts into normalized vectors (so cosine similarity = dot product)."""
    if is_query:
        texts = [QUERY_PREFIX + t for t in texts]
    return get_model().encode(texts, normalize_embeddings=True,
                              show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL)


def retrieve(question: str, k: int = TOP_K, rerank: bool = None):
    """Return the k chunks most relevant to the question, best first.

    rerank=None follows the RERANK setting; False gives the raw embedding search.
    """
    rerank = RERANK if rerank is None else rerank
    n = max(CANDIDATES, k) if rerank else k
    res = get_collection().query(query_embeddings=embed([question], is_query=True),
                                 n_results=n)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({
            "text": doc,
            "title": meta["title"],
            "section": meta["section"],
            "url": meta["url"],
            "score": round(1 - dist, 3),   # cosine distance -> similarity (1 = identical)
        })

    if rerank:
        # The cross-encoder reads question and passage together -> a relevance score
        scores = get_reranker().predict([(question, h["text"]) for h in hits])
        for hit, s in zip(hits, scores):
            hit["score"] = round(float(s), 2)   # relevance (higher = better, can be < 0)
        hits.sort(key=lambda h: h["score"], reverse=True)
    return diversify(hits, k) if rerank else hits[:k]


def diversify(hits, k):
    """Keep the best-ranked hits, with at most MAX_PER_PAGE chunks per page."""
    if MAX_PER_PAGE <= 0:
        return hits[:k]
    kept, per_page = [], {}
    for hit in hits:
        page = hit["url"].split("#")[0]
        if per_page.get(page, 0) < MAX_PER_PAGE:
            kept.append(hit)
            per_page[page] = per_page.get(page, 0) + 1
        if len(kept) == k:
            break
    return kept


# ---------------------------------------------------------------------------
# Step 3: generation - the LLM writes an answer from the retrieved chunks
# ---------------------------------------------------------------------------

def build_messages(question: str, hits):
    """Build the chat messages sent to the LLM: rules + numbered sources + question."""
    sources = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
    # Small models follow instructions better when they are repeated
    # right next to the question, at the end of the prompt.
    user_msg = (f"Documentation excerpts:\n\n{sources}\n\n"
                f"Question: {question}\n\n"
                "Answer using ONLY the excerpts above. Cite them like [1] after each claim. "
                "For code, copy the exact calls and arguments shown in the excerpts "
                "instead of writing new ones. If the excerpts describe both a GUI way "
                "and a Python way, give both briefly.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg}]


def generate(question: str, hits):
    """Ask Ollama for an answer and yield it piece by piece (streaming)."""
    payload = {
        "model": LLM_MODEL,
        "messages": build_messages(question, hits),
        "stream": True,
        "options": {
            "temperature": 0.1,  # low = factual, sticks to the sources
            "num_ctx": 8192,     # context window big enough for 5 chunks + answer
        },
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            yield data.get("message", {}).get("content", "")
            if data.get("done"):
                break


def print_sources(hits):
    for rank, hit in enumerate(hits, 1):
        print(f"[{rank}] ({hit['score']}) {hit['title']} > {hit['section']}")
        print(f"    {hit['url']}")


if __name__ == "__main__":
    # python rag.py "question"            -> sources + answer written by the LLM
    # python rag.py --search "question"   -> retrieval only (shows the passages found)
    args = sys.argv[1:]
    search_only = bool(args) and args[0] == "--search"
    if search_only:
        args = args[1:]
    question = " ".join(args) or "How do I create a group of faces?"
    print(f"Question: {question}\n")

    hits = retrieve(question)
    if search_only:
        for rank, hit in enumerate(hits, 1):
            preview = hit["text"].split("\n\n", 1)[-1][:200].replace("\n", " ")
            print(f"{rank}. [{hit['score']}] {hit['title']} > {hit['section']}")
            print(f"   {hit['url']}")
            print(f"   {preview}...\n")
        sys.exit()

    print("Sources:")
    print_sources(hits)
    print("\nAnswer:\n")
    start = time.time()
    for piece in generate(question, hits):
        print(piece, end="", flush=True)
    print(f"\n\n({time.time() - start:.0f}s)")
