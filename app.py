"""Step 4 - Web interface for the SALOME Mesh assistant.

Run:   streamlit run app.py
Then open the address shown in the terminal (usually http://localhost:8501).
"""

import re
import time

import requests
import streamlit as st

import rag

st.set_page_config(page_title="SALOME Mesh Assistant", page_icon="🔷", layout="wide")

EXAMPLES = [
    "How do I create a group of faces?",
    "How do I export my mesh to a MED file?",
    "How can I check the aspect ratio of my elements?",
    "How do I merge nodes that are at the same location with a Python script?",
]


# ---------------------------------------------------------------- loading
@st.cache_resource(show_spinner="Loading the search index…")
def load_search():
    """Load the embedding model and the database once for the whole app."""
    rag.get_model()
    return rag.get_collection().count()


def linkify(answer: str, hits) -> str:
    """Turn citations like [2] into clickable links to the matching doc page."""
    def repl(m):
        i = int(m.group(1))
        return f"[[{i}]]({hits[i - 1]['url']})" if 1 <= i <= len(hits) else m.group(0)
    return re.sub(r"\[(\d+)\](?!\()", repl, answer)


def show_sources(hits, expanded=False):
    with st.expander(f"📚 Sources ({len(hits)})", expanded=expanded):
        for i, h in enumerate(hits, 1):
            st.markdown(f"**[{i}]** [{h['title']} › {h['section']}]({h['url']}) "
                        f"<small>· score {h['score']}</small>", unsafe_allow_html=True)
            if st.session_state.get("show_text"):
                st.caption(h["text"].split("\n\n", 1)[-1][:600] + " …")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    models = rag.list_llm_models()
    if models:
        default = models.index(rag.LLM_MODEL) if rag.LLM_MODEL in models else 0
        model = st.selectbox("Language model", models, index=default,
                             help="Small models answer faster, bigger ones are more accurate.")
    else:
        model = None
        st.error("Ollama is not reachable. Start it with `ollama serve`.")
    k = st.slider("Passages given to the model", 3, 8, rag.TOP_K)
    st.toggle("Show passage text in sources", key="show_text")
    st.divider()
    try:
        n_chunks = load_search()
        st.caption(f"Index: {n_chunks} chunks · {rag.EMBED_MODEL.split('/')[-1]}")
    except Exception as e:  # index not built yet
        st.error(f"Search index not found. Run `python index.py` first.\n\n{e}")
        st.stop()
    st.caption("Answers come only from the public SALOME Mesh (SMESH) documentation. "
               "Everything runs locally: no data leaves this machine.")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------- main
st.title("🔷 SALOME Mesh Assistant")
st.caption("Ask a question about meshing in SALOME — in the GUI or with a Python script.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("hits"):
            show_sources(msg["hits"])
        if msg.get("info"):
            st.caption(msg["info"])

# Example questions on an empty conversation
question = None
examples_box = st.empty()
if not st.session_state.messages:
    with examples_box.container():
        st.markdown("**Try an example:**")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, use_container_width=True):
                question = ex

question = st.chat_input("Ask about SALOME meshing…") or question

if question:
    examples_box.empty()  # hide the examples once the conversation starts
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        t0 = time.time()
        with st.spinner("Searching the documentation…"):
            hits = rag.retrieve(question, k=k)
        t_search = time.time() - t0
        show_sources(hits)

        if model is None:
            st.warning("No language model available: showing the passages found only.")
            st.session_state.messages.append(
                {"role": "assistant", "content": "*(no model — sources only)*", "hits": hits})
            st.stop()

        placeholder, answer = st.empty(), ""
        t1 = time.time()
        try:
            with st.spinner(f"Writing the answer with {model}…"):
                for piece in rag.generate(question, hits, model=model):
                    answer += piece
                    placeholder.markdown(answer + "▌")
        except requests.RequestException as e:
            st.error(f"Could not reach Ollama: {e}")
            st.stop()

        answer = linkify(answer, hits)
        placeholder.markdown(answer)
        info = f"{model} · search {t_search:.1f}s · answer {time.time() - t1:.0f}s"
        st.caption(info)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hits": hits, "info": info})
