"""Web interface for the SALOME Mesh Assistant.

Run:   streamlit run app.py
Then open http://localhost:8501
"""

import html
import re
import time

import requests
import streamlit as st

import rag

st.set_page_config(page_title="SALOME Mesh Assistant", page_icon="◆", layout="centered",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------- look & feel
# Palette: white page, steel-grey surfaces, one steel-blue accent for links/citations.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink: #1D232B;
  --muted: #5C6672;
  --line: #E3E6EA;
  --surface: #F4F6F8;
  --accent: #2F5D8A;
}

.stApp, .stApp [data-testid="stMarkdownContainer"], .stApp p, .stApp li, .stApp h1,
.stApp h2, .stApp h3, .stApp label, .stApp button, .stApp input, .stApp textarea,
.stApp summary p {
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}
code, pre, .stCode, kbd { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; }

/* Streamlit chrome we don't need */
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, #MainMenu { display: none !important; }
[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 7rem; }

/* Messages: no avatars, user on the right in a bubble, assistant as plain text */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] { display: none; }
[data-testid="stChatMessage"] { background: transparent; padding: 0.25rem 0; gap: 0; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
  margin-top: 1.4rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
  [data-testid="stChatMessageContent"] {
  flex: 0 1 auto; max-width: 85%; margin-left: auto !important; margin-right: 0 !important;
  background: var(--surface); border-radius: 18px; padding: 0.6rem 1.05rem;
}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p { margin: 0; line-height: 1.5; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdown"],
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
  margin: 0 !important;
}
[data-testid="stChatMessageContent"] { color: var(--ink); font-size: 0.98rem; line-height: 1.65; }

/* Citations [n] in answers */
a.cite {
  display: inline-block; min-width: 1.15rem; padding: 0 0.3rem; margin: 0 0.1rem;
  font-size: 0.72rem; line-height: 1.15rem; text-align: center; vertical-align: 0.15rem;
  color: var(--accent); background: #E8EEF5; border-radius: 4px; text-decoration: none;
}
a.cite:hover { background: var(--accent); color: #fff; }
[data-testid="stChatMessageContent"] a { color: var(--accent); }

/* Sources list */
.sources { margin: 0.4rem 0 0 0; padding: 0; list-style: none; font-size: 0.86rem; }
.sources li { padding: 0.3rem 0; border-top: 1px solid var(--line); display: flex; gap: 0.6rem; }
.sources li:first-child { border-top: none; }
.sources .n { color: var(--muted); min-width: 1rem; }
.sources a { color: var(--ink); text-decoration: none; }
.sources a:hover { color: var(--accent); text-decoration: underline; }
.sources .sec { color: var(--muted); }
.sources .excerpt { display: block; color: var(--muted); font-size: 0.8rem; margin-top: 0.15rem; }
[data-testid="stExpander"] details { border: none; background: transparent; }
[data-testid="stExpander"] summary { padding-left: 0; background: transparent !important; }
[data-testid="stExpander"] summary p { color: var(--muted); font-size: 0.86rem; }
[data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0 0 0.2rem 1.4rem; }
.stApp ul.sources li { font-size: 0.88rem; line-height: 1.45; margin: 0; }

.meta { color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }
.warn { font-size: 0.86rem; color: #7A4B00; background: #FFF6E0; border-left: 3px solid #E0A200;
        padding: 0.45rem 0.75rem; border-radius: 4px; margin: 0.4rem 0 0.2rem 0; }
.warn code { background: transparent; color: #7A4B00; }
.searched { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.4rem; }

/* Empty state */
.welcome { text-align: center; margin-top: 22vh; }
.welcome h1 { font-size: 1.9rem; font-weight: 600; color: var(--ink); letter-spacing: -0.01em;
              margin-bottom: 0.4rem; padding: 0; }
.welcome p { color: var(--muted); font-size: 1rem; max-width: 30rem; margin: 0 auto; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .brand { font-weight: 600; font-size: 1.05rem; color: var(--ink); margin: 0; }
[data-testid="stSidebar"] .note { color: var(--muted); font-size: 0.8rem; line-height: 1.5; }

/* Input */
[data-testid="stChatInput"] textarea { font-size: 0.98rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading the search index…")
def load_search():
    """Load the embedding model and the database once for the whole app."""
    rag.get_model()
    return rag.get_collection().count()


def linkify(answer: str, hits) -> str:
    """Turn citations like [2] into small links to the doc section.

    Code blocks are left untouched: [0] in Python code is an index, not a citation.
    """
    def repl(m):
        i = int(m.group(1))
        if 1 <= i <= len(hits):
            url = html.escape(hits[i - 1]["url"], quote=True)
            return f'<a class="cite" href="{url}" target="_blank" title="{html.escape(hits[i - 1]["title"])}">{i}</a>'
        return m.group(0)
    parts = answer.split("```")
    for j in range(0, len(parts), 2):          # even parts are outside code blocks
        parts[j] = re.sub(r"\[(\d+)\](?!\()", repl, parts[j])
    return "```".join(parts)


def show_sources(hits):
    with st.expander(f"{len(hits)} sources"):
        rows = []
        for i, h in enumerate(hits, 1):
            excerpt = ""
            if st.session_state.get("show_text"):
                body = h["text"].split("\n\n", 1)[-1][:400].replace("\n", " ")
                excerpt = f'<span class="excerpt">{html.escape(body)}…</span>'
            rows.append(
                f'<li><span class="n">{i}</span><span><a href="{html.escape(h["url"], quote=True)}" '
                f'target="_blank">{html.escape(h["title"])}</a> '
                f'<span class="sec">› {html.escape(h["section"])}</span>{excerpt}</span></li>')
        st.markdown(f'<ul class="sources">{"".join(rows)}</ul>', unsafe_allow_html=True)


def warn_invented(names):
    if names:
        listed = ", ".join(f"<code>{html.escape(n)}</code>" for n in names)
        verb = "does" if len(names) == 1 else "do"
        st.markdown(f'<div class="warn">Check before use: {listed} {verb} not appear anywhere in '
                    f'the SALOME Mesh documentation, so the model probably invented '
                    f'{"it" if len(names) == 1 else "them"}.</div>',
                    unsafe_allow_html=True)


def meta(text: str):
    st.markdown(f'<div class="meta">{html.escape(text)}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<p class="brand">SALOME Mesh Assistant</p>', unsafe_allow_html=True)
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.write("")

    models = rag.list_llm_models()
    if models:
        default = models.index(rag.LLM_MODEL) if rag.LLM_MODEL in models else 0
        model = st.selectbox("Model", models, index=default,
                             help="Smaller models answer faster; larger ones are more accurate.")
    else:
        model = None
        st.error("Ollama is not running. Start it with `ollama serve`, then reload the page.")

    with st.expander("Search settings"):
        k = st.slider("Passages given to the model", 3, 8, rag.TOP_K)
        mode = st.radio("Rephrase my question before searching", ["Auto", "Always", "Never"],
                        help="Auto rephrases follow-up questions and questions that are not "
                             "in English (such as French) into an English search query.")
        rewrite = {"Auto": None, "Always": True, "Never": False}[mode]
        st.toggle("Show excerpts in sources", key="show_text")

    try:
        n_chunks = load_search()
    except Exception as e:  # index not built yet
        st.error(f"The search index is missing. Run `python index.py`, then reload.\n\n{e}")
        st.stop()
    st.markdown(
        f'<p class="note">Answers are based only on the SALOME Mesh (SMESH) documentation '
        f'({n_chunks} indexed passages). Everything runs on this computer; '
        f'your questions are not sent anywhere.</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- conversation
if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Ask a question about meshing in SALOME")

if not st.session_state.messages and not question:
    st.markdown(
        '<div class="welcome"><h1>SALOME Mesh Assistant</h1>'
        '<p>Ask how to do something in SALOME Mesh, in the interface or with a Python '
        'script. Each answer links to the documentation it is based on.</p></div>',
        unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
            continue
        if msg.get("query"):
            st.markdown(f'<div class="searched">Searched for “{html.escape(msg["query"])}”</div>',
                        unsafe_allow_html=True)
        st.markdown(msg["content"], unsafe_allow_html=True)
        warn_invented(msg.get("invented", []))
        if msg.get("hits"):
            show_sources(msg["hits"])
        if msg.get("info"):
            meta(msg["info"])

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Previous (question, answer) pairs, so follow-ups like "and in Python?" work
        msgs = st.session_state.messages[:-1]
        history = [(q["content"], a["content"]) for q, a in zip(msgs[::2], msgs[1::2])]
        t0 = time.time()
        with st.spinner("Searching the documentation…"):
            hits, query = rag.retrieve_with_query(question, k=k, rewrite=rewrite,
                                                  history=history)
        shown_query = query if query != question else None
        if shown_query:
            st.markdown(f'<div class="searched">Searched for “{html.escape(query)}”</div>',
                        unsafe_allow_html=True)

        if model is None:
            text = "No language model is available, so here are the most relevant passages."
            st.markdown(text)
            show_sources(hits)
            st.session_state.messages.append(
                {"role": "assistant", "content": text, "hits": hits, "query": shown_query})
            st.stop()

        placeholder, answer = st.empty(), ""
        try:
            with st.spinner("Writing the answer…"):
                for piece in rag.generate(question, hits, model=model):
                    answer += piece
                    placeholder.markdown(answer + " ▍")
        except requests.RequestException as e:
            st.error(f"Ollama stopped responding ({e}). Check that it is running, then ask again.")
            st.stop()

        invented = rag.invented_calls(answer, hits)   # check before adding links
        answer = linkify(answer, hits)
        placeholder.markdown(answer, unsafe_allow_html=True)
        warn_invented(invented)
        show_sources(hits)
        info = f"{model}, {time.time() - t0:.0f} s"
        s = rag.LAST_STATS
        if s:
            info += (f" (reading the sources {s['read_s']:.0f} s, writing {s['write_s']:.0f} s"
                     + (f", loading the model {s['load_s']:.0f} s" if s["load_s"] > 1 else "")
                     + ")")
        meta(info)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits, "info": info,
         "query": shown_query, "invented": invented})
