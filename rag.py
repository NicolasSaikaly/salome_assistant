"""Core of the assistant: retrieval (step 2b) + generation (step 3).

    retrieve(question)        -> the doc chunks closest to the question
    generate(question, hits)  -> the LLM answer, written from those chunks only

From the terminal:
    python rag.py "How do I create a group of faces?"
    python rag.py --search "How do I create a group of faces?"   (retrieval only)
"""

import json
import os
import re
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

# Query rewriting before the search (needs Ollama, adds ~2-5 s per question on CPU).
#   auto (default): only for follow-up questions and non-English questions.
#                   On English questions it HURT retrieval (see eval/RESULTS.md).
#   1: always   0: never        e.g.  REWRITE=1 python eval_retrieval.py
REWRITE = os.environ.get("REWRITE", "auto")
REWRITE_MODEL = os.environ.get("REWRITE_MODEL", "qwen2.5:3b")
# NB: the examples below are deliberately NOT taken from eval/questions.jsonl,
# otherwise the evaluation would be biased.
REWRITE_PROMPT = """You turn a user's question about the SALOME Mesh module (SMESH)
into a search query for its English reference documentation.
Rules:
- Write in English, even if the question is in another language.
- Replace everyday words with the technical meshing terms a reference manual would use,
  and add 2-3 likely synonyms or feature names.
- If the question is a follow-up, use the previous conversation to make it self-contained.
- Never invent function or method names: only use words from the question or common
  meshing vocabulary.
- Output ONLY the query, on one line, at most 25 words. Do not answer the question.

Examples:
Question: comment supprimer un groupe ?
Query: delete remove group of mesh elements, deleting groups
Question: how can I make my mesh show up in red?
Query: change mesh display color, colors and size of elements, display properties
Previous conversation: User: How do I delete some nodes? Assistant: Use Modification > Remove > Nodes...
Question: and with a script?
Query: remove delete nodes with a Python script"""
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


def search(query: str, n: int):
    """Raw embedding search: the n chunks closest to the query, best first."""
    res = get_collection().query(query_embeddings=embed([query], is_query=True),
                                 n_results=n)
    return [{"text": doc, "title": meta["title"], "section": meta["section"],
             "url": meta["url"],
             "score": round(1 - dist, 3)}   # cosine distance -> similarity (1 = identical)
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0],
                                       res["distances"][0])]


def fuse(*ranked_lists, c: int = 60):
    """Reciprocal Rank Fusion: merge several rankings into one.

    Each chunk gets sum(1 / (c + rank)) over the lists it appears in, so a chunk
    ranked well by both the original and the rewritten query comes first.
    """
    scores, by_key = {}, {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, 1):
            key = hit["text"]
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            by_key.setdefault(key, hit)
    return [by_key[key] for key in sorted(scores, key=scores.get, reverse=True)]


def retrieve_with_query(question: str, k: int = TOP_K, rerank: bool = None,
                        rewrite: bool = None, history=None):
    """Return (the k most relevant chunks, the search query actually used).

    rerank / rewrite = None follow the RERANK / REWRITE settings.
    history = previous (question, answer) pairs, used to rewrite follow-up questions.
    """
    rerank = RERANK if rerank is None else rerank
    if rewrite is None:
        rewrite = (REWRITE == "1" or
                   (REWRITE == "auto" and (bool(history) or not looks_english(question))))
    n = max(CANDIDATES, k) if (rerank or rewrite) else k

    hits, query = search(question, n), question
    if rewrite:
        query = rewrite_query(question, history)
        if query != question:
            # Keep the original search too: if the rewrite goes wrong, we lose nothing
            hits = fuse(hits, search(query, n))

    if rerank:
        # The cross-encoder reads question and passage together -> a relevance score
        scores = get_reranker().predict([(question, h["text"]) for h in hits])
        for hit, s in zip(hits, scores):
            hit["score"] = round(float(s), 2)   # relevance (higher = better, can be < 0)
        hits.sort(key=lambda h: h["score"], reverse=True)
        return diversify(hits, k), query
    return hits[:k], query


def retrieve(question: str, k: int = TOP_K, rerank: bool = None, rewrite: bool = None,
             history=None):
    """Return the k chunks most relevant to the question, best first."""
    return retrieve_with_query(question, k, rerank, rewrite, history)[0]


# ---------------------------------------------------------------------------
# Query rewriting: the LLM turns the user's question into a good search query
# ---------------------------------------------------------------------------

FRENCH_WORDS = {"comment", "je", "mon", "ma", "mes", "le", "la", "les", "un", "une",
                "des", "du", "de", "pour", "avec", "dans", "sur", "est", "quel", "quelle",
                "faire", "peut", "puis", "maillage", "noeuds", "nœuds", "et", "ou", "en"}


def looks_english(question: str) -> bool:
    """Cheap language check: accents or several common French words -> not English."""
    if re.search(r"[éèêàâùûçôîœ]", question.lower()):
        return False
    words = re.findall(r"[a-zœ]+", question.lower())
    return sum(w in FRENCH_WORDS for w in words) < 2


def rewrite_query(question: str, history=None, model: str = None) -> str:
    """Rewrite a question into an English search query in reference-manual vocabulary.

    Fixes three things at once: users' everyday wording vs. the doc's technical terms,
    questions asked in French, and follow-ups ("and with Python?") that need the
    previous exchange to make sense. Falls back to the original question on error.
    """
    context = ""
    if history:
        last = history[-2:]  # the last two exchanges are enough for follow-ups
        context = "Previous conversation:\n" + "\n".join(
            f"User: {q}\nAssistant: {a[:300]}" for q, a in last) + "\n\n"
    payload = {
        "model": model or REWRITE_MODEL,
        "messages": [{"role": "system", "content": REWRITE_PROMPT},
                     {"role": "user", "content": f"{context}Question: {question}"}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 60},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        query = resp.json()["message"]["content"].strip().strip('"').splitlines()[0]
        return query or question
    except (requests.RequestException, KeyError, IndexError):
        return question


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


def generate(question: str, hits, model: str = None):
    """Ask Ollama for an answer and yield it piece by piece (streaming)."""
    payload = {
        "model": model or LLM_MODEL,
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


def list_llm_models():
    """Names of the models installed in Ollama (empty list if Ollama is not running)."""
    try:
        resp = requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=3)
        return sorted(m["name"] for m in resp.json().get("models", []))
    except requests.RequestException:
        return []


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
