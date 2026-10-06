"""
AI College Information Assistant  --  Llama + RAG version
Streamlit + Sentence Transformers (retrieval) + Llama via Ollama (generation)

Pipeline:
  1. Load college documents (data/*.md, *.txt) -> chunks
  2. Embed chunks with a Sentence Transformer (retrieval)
  3. Retrieve the top-k chunks for the question
  4. Send chunks + chat history to Llama (running locally in Ollama) -> streamed answer
  5. Save to persistent memory (follow-ups, repeated questions, name, history)

Setup for Llama:
  1. Install Ollama from https://ollama.com
  2. Run:  ollama pull llama3.2        (or llama3.2:1b for weaker laptops)
  3. Run:  streamlit run app_llama.py
"""
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ----------------------------- Config ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory.json")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_CHAT = f"{OLLAMA_HOST}/api/chat"
OLLAMA_TAGS = f"{OLLAMA_HOST}/api/tags"
DEFAULT_LLAMA = "llama3.2"

MAX_WORDS_PER_CHUNK = 120
MIN_SCORE = 0.30
REPEAT_THRESHOLD = 0.90
MAX_HISTORY = 200
CHAT_TURNS_TO_SEND = 6

SYSTEM_PROMPT = (
    "You are the AI College Information Assistant for ABC Institute of Technology. "
    "Answer the user's question using ONLY the college information provided in the message. "
    "If the answer is not in the provided information, say you don't have that information "
    "and suggest contacting the college office (info@abcit.edu.in, 040-1234-5678). "
    "Never invent fees, dates, phone numbers or facilities. "
    "Be friendly and concise. Use short bullet points for lists such as routes, fees or facilities. "
    "Use the earlier conversation to understand follow-up questions such as 'what about its timings?'."
)


FOLLOWUP_STARTERS = (
    "and ", "also ", "what about", "how about", "then ", "ok ", "okay ",
    "so ", "tell me more", "more ", "give more", "explain", "elaborate",
)
PRONOUNS = {"it", "its", "they", "them", "their", "those", "these", "he", "she"}


# --------------------------- Persistent memory ------------------------
def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                mem = json.load(f)
            mem.setdefault("name", None)
            mem.setdefault("history", [])
            mem.setdefault("last_topic", None)
            return mem
        except Exception:
            pass
    return {"name": None, "history": [], "last_topic": None}


def save_memory(mem):
    try:
        mem["history"] = mem["history"][-MAX_HISTORY:]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ------------------------ Document processing -------------------------
def split_into_chunks(text: str, source: str):
    chunks = []
    for sec in re.split(r"(?m)^##\s+", text):
        sec = sec.strip()
        if not sec:
            continue
        lines = sec.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if not body:
            body, title = title, source
        words = body.split()
        for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
            piece = " ".join(words[i : i + MAX_WORDS_PER_CHUNK])
            chunks.append({"title": title, "text": f"{title}. {piece}", "source": source})
    return chunks


def load_documents(extra_files=None):
    chunks = []
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.md")) + glob.glob(os.path.join(DATA_DIR, "*.txt")))
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            chunks += split_into_chunks(f.read(), os.path.basename(p))
    for name, content in extra_files or []:
        chunks += split_into_chunks(content, name)
    return chunks


def split_sentences(text: str):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if len(p.strip()) > 15]


# --------------------------- Models / index ---------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Building knowledge base...")
def build_index(chunk_texts: tuple):
    emb = get_embedder().encode(list(chunk_texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


# ------------------------------ Retrieval -----------------------------
def retrieve(query, chunks, index, k=4):
    q = get_embedder().encode([query], normalize_embeddings=True)[0]
    scores = index @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def answer_extractive(query, retrieved, n_sentences=4):
    """Fallback answer without any LLM (used if Llama is unavailable)."""
    sentences = []
    for chunk, _ in retrieved:
        sentences += split_sentences(chunk["text"].split(". ", 1)[-1])
    if not sentences:
        return retrieved[0][0]["text"]
    embedder = get_embedder()
    s_emb = embedder.encode(sentences, normalize_embeddings=True)
    q_emb = embedder.encode([query], normalize_embeddings=True)[0]
    order = np.argsort(-(s_emb @ q_emb))[:n_sentences]
    return " ".join(sentences[i] for i in sorted(order))


# ------------------------------- Llama --------------------------------
def ollama_models():
    """Return list of installed Ollama models, or None if Ollama isn't running."""
    try:
        r = requests.get(OLLAMA_TAGS, timeout=2)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def build_messages(query, retrieved, history, name, topic_hint=None):
    context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, (c, _) in enumerate(retrieved))
    system = SYSTEM_PROMPT + (f" The user's name is {name}." if name else "")
    hint = f"(We were discussing: {topic_hint})\n" if topic_hint else ""
    user_msg = f"College information:\n{context}\n\n{hint}Question: {query}"
    return [{"role": "system", "content": system}] + history + [{"role": "user", "content": user_msg}]


def llama_stream(messages, model):
    """Generator that yields the answer text piece by piece from Ollama."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2, "num_ctx": 4096},
    }
    with requests.post(OLLAMA_CHAT, json=payload, stream=True, timeout=300) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            if "error" in data:
                raise RuntimeError(data["error"])
            piece = data.get("message", {}).get("content", "")
            if piece:
                yield piece
            if data.get("done"):
                break


# ------------------------------ Memory logic --------------------------
def needs_context(q: str) -> bool:
    ql = q.lower().strip()
    words = set(re.findall(r"[a-z']+", ql))
    return ql.startswith(FOLLOWUP_STARTERS) or bool(PRONOUNS & words)


def find_repeated(query, memory):
    past = [h["question"] for h in memory["history"]]
    if not past:
        return None
    embedder = get_embedder()
    p = embedder.encode(past, normalize_embeddings=True)
    q = embedder.encode([query], normalize_embeddings=True)[0]
    sims = p @ q
    i = int(np.argmax(sims))
    return past[i] if sims[i] >= REPEAT_THRESHOLD else None


def handle_special(query, memory):
    ql = query.lower().strip()

    m = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Za-z]{2,20})\b", ql)
    if m and not any(w in ql for w in ("student", "from", "looking", "interested")):
        memory["name"] = m.group(1).capitalize()
        save_memory(memory)
        return f"Nice to meet you, **{memory['name']}**! 😊 I'll remember your name. How can I help you today?"

    if re.search(r"(what('?s| is) my name|do you (know|remember) my name|who am i)", ql):
        if memory["name"]:
            return f"Your name is **{memory['name']}**. 😊"
        return "I don't know your name yet. You can tell me by saying *My name is ...*"

    if re.search(
        r"(what|which).*(did i|have i|i have).*(ask|asked|say|said)|previous questions?|past questions?|"
        r"my (last|previous) question|chat history|what (did )?we (discuss|talk)|do you remember",
        ql,
    ):
        hist = memory["history"]
        if not hist:
            return "We haven't talked about anything yet. Ask me something about the college! 🎓"
        if re.search(r"(last|previous) question", ql):
            return f"Your last question was: **“{hist[-1]['question']}”**"
        lines = [f"{i}. {h['question']}" for i, h in enumerate(hist[-10:], 1)]
        return "Here are your recent questions:\n\n" + "\n".join(lines)

    if re.search(r"^(hi|hello|hey|good (morning|afternoon|evening))\b", ql) and len(ql.split()) <= 4:
        name = f", {memory['name']}" if memory["name"] else ""
        return f"Hello{name}! 👋 Ask me anything about the college: admissions, fees, hostel, bus routes, placements and more."

    if re.search(r"^(thanks|thank you|thx|ok thanks)\b", ql):
        return "You're welcome! 😊 Feel free to ask anything else."

    return None


def prepare_retrieval(query, chunks, index, k, memory):
    """Find relevant chunks, handling follow-ups and repeated questions."""
    note = ""
    repeated = find_repeated(query, memory)
    if repeated:
        note += f"💭 *You asked something similar earlier (“{repeated}”). Here is the answer again:*\n\n"

    search_query = query
    retrieved = retrieve(search_query, chunks, index, k)
    words = re.findall(r"[A-Za-z']+", query)
    topic_hint = None

    if memory["last_topic"] and (needs_context(query) or (len(words) <= 3 and retrieved[0][1] < 0.45)):
        topic_hint = memory["last_topic"]
        search_query = f"{topic_hint}. {query}"
        retrieved = retrieve(search_query, chunks, index, k)
        note += f"🔗 *Follow-up detected, continuing about “{topic_hint}”.*\n\n"

    topic = retrieved[0][0]["title"]
    return retrieved, search_query, topic, topic_hint, note


# ------------------------------- UI -----------------------------------
st.set_page_config(page_title="College Assistant (Llama + RAG)", page_icon="🦙", layout="centered")
st.title("🎓 AI College Information Assistant")
st.caption("Llama + RAG · Ask about courses, fees, hostel, bus routes, placements and more. It remembers you!")

if "memory" not in st.session_state:
    st.session_state["memory"] = load_memory()
memory = st.session_state["memory"]

with st.sidebar:
    st.header("⚙️ Settings")
    installed = ollama_models()
    if installed is None:
        st.error("🔴 Ollama is not running. Install it from ollama.com and run `ollama pull llama3.2`.")
        llama_model = st.text_input("Llama model name", DEFAULT_LLAMA)
    elif not installed:
        st.warning("🟡 Ollama is running but no model is installed. Run `ollama pull llama3.2`.")
        llama_model = st.text_input("Llama model name", DEFAULT_LLAMA)
    else:
        st.success("🟢 Ollama is running")
        llamas = [m for m in installed if "llama" in m.lower()]
        options = llamas or installed
        llama_model = st.selectbox("Llama model", options)

    mode = st.radio("Answer mode", ["Llama (RAG)", "Extractive (no LLM)"])
    top_k = st.slider("Chunks to retrieve (k)", 1, 6, 4)
    show_sources = st.checkbox("Show retrieved sources", value=True)

    st.divider()
    st.subheader("🧠 Memory")
    st.write(f"Name: **{memory['name'] or 'unknown'}**")
    st.write(f"Questions remembered: **{len(memory['history'])}**")
    if memory["last_topic"]:
        st.write(f"Current topic: **{memory['last_topic']}**")
    with st.expander("Past questions"):
        if memory["history"]:
            for h in reversed(memory["history"][-15:]):
                st.write(f"• {h['question']}")
        else:
            st.write("Nothing yet.")
    if st.button("🗑️ Clear memory & chat", use_container_width=True):
        memory.update({"name": None, "history": [], "last_topic": None})
        save_memory(memory)
        st.session_state["messages"] = []
        st.rerun()



    st.divider()
    st.subheader("📄 Add your own documents")
    uploads = st.file_uploader("Upload .txt / .md", type=["txt", "md"], accept_multiple_files=True)

extra = [(u.name, u.getvalue().decode("utf-8", errors="ignore")) for u in (uploads or [])]
chunks = load_documents(extra)
if not chunks:
    st.error("No documents found. Add .md or .txt files to the 'data' folder.")
    st.stop()
index = build_index(tuple(c["text"] for c in chunks))
st.sidebar.caption(f"Knowledge base: {len(chunks)} chunks")

if not st.session_state.get("messages"):
    if memory["history"]:
        who = f" back, {memory['name']}" if memory["name"] else " back"
        greet = (f"Welcome{who}! 👋 Last time you asked: **“{memory['history'][-1]['question']}”**. "
                 "Want to continue, or ask something new?")
    else:
        greet = ("Hi! 👋 I'm your college assistant powered by Llama. Use the **Explore topics** "
                 "buttons on the left or ask me anything about the college.")
    st.session_state["messages"] = [{"role": "assistant", "content": greet}]


def render_sources(sources):
    with st.expander("Sources"):
        for title, src, score in sources:
            st.write(f"**{title}** ({src}), similarity {score:.2f}")


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources") and show_sources:
            render_sources(m["sources"])
query = st.chat_input("Type your question here...")

if query:
    # chat history to send to Llama (before adding the current question)
    prior = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"]]
    prior = prior[-CHAT_TURNS_TO_SEND:]
    while prior and prior[0]["role"] != "user":
        prior.pop(0)

    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        special = handle_special(query, memory)
        sources = []
        if special:
            full = special
            st.markdown(full)
        else:
            with st.spinner("Searching college records..."):
                retrieved, search_query, topic, topic_hint, note = prepare_retrieval(
                    query, chunks, index, top_k, memory
                )
            sources = [(c["title"], c["source"], s) for c, s in retrieved]
            answer = ""
            save_it = True

            if retrieved[0][1] < MIN_SCORE:
                answer = ("Sorry, I couldn't find that in my college records. "
                          "Please contact the office at info@abcit.edu.in or 040-1234-5678.")
                topic = memory["last_topic"]
                if note:
                    st.markdown(note)
                st.markdown(answer)
                save_it = False
            else:
                if note:
                    st.markdown(note)
                if mode == "Llama (RAG)":
                    msgs = build_messages(query, retrieved, prior, memory["name"], topic_hint)
                    try:
                        answer = st.write_stream(llama_stream(msgs, llama_model))
                    except Exception as e:
                        st.warning(f"Llama is unavailable ({type(e).__name__}). Using the simple answer mode instead.")
                        answer = answer_extractive(search_query, retrieved)
                        st.markdown(answer)
                else:
                    answer = answer_extractive(search_query, retrieved)
                    st.markdown(answer)

            full = note + answer
            if show_sources:
                render_sources(sources)
            if save_it:
                memory["history"].append(
                    {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "question": query,
                     "answer": answer, "topic": topic}
                )
                memory["last_topic"] = topic
                save_memory(memory)

    st.session_state["messages"].append({"role": "assistant", "content": full, "sources": sources})
