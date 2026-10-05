"""
AI College Information Assistant  (with conversation memory)
Streamlit + Sentence Transformers + RAG (Retrieval Augmented Generation)

Pipeline:
  1. Load college documents (data/*.md, *.txt) -> split into chunks
  2. Embed chunks with a Sentence Transformer
  3. For a question: embed it, retrieve top-k similar chunks
  4. Generate the answer (Extractive or Generative FLAN-T5)

Memory features:
  - Persistent memory saved in chat_memory.json (survives page refresh / restart)
  - Follow-up understanding ("what about its timings?" uses the previous topic)
  - Repeated question recognition ("you asked this earlier")
  - Recall commands ("what did I ask before?", "what is my name?")
  - Remembers the user's name
"""
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ----------------------------- Config ---------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MEMORY_FILE = os.path.join(BASE_DIR, "chat_memory.json")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
MAX_WORDS_PER_CHUNK = 120
MIN_SCORE = 0.30          # below this similarity -> "I don't know"
REPEAT_THRESHOLD = 0.90   # similarity to a past question -> "asked before"
MAX_HISTORY = 200

TOPICS = {
    "🏛️ About college": "Tell me about the college",
    "📚 Courses": "What courses does the college offer?",
    "📝 Admissions": "What is the admission process?",
    "💰 B.Tech fees": "What is the B.Tech fee structure?",
    "🏠 Hostel": "Tell me about hostel facilities",
    "🍽️ Hostel fees": "What are the hostel fees?",
    "🚌 Bus overview": "Tell me about the college bus facility",
    "🗺️ Bus routes": "What are the bus routes?",
    "📖 Library": "What are the library timings?",
    "🎯 Placements": "Tell me about placements",
    "🎓 Scholarships": "What scholarships are available?",
    "📅 Attendance": "What is the minimum attendance required?",
    "⚽ Sports": "What sports facilities are there?",
    "🏥 Medical": "Is there a medical centre?",
    "🎉 Events": "What events happen in the college?",
    "📞 Contact": "How can I contact the college?",
}

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
    """Split markdown by '## Title' sections, then by size."""
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


# --------------------------- Model loading ----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading generation model (first time takes a while)...")
def get_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Building knowledge base...")
def build_index(chunk_texts: tuple):
    emb = get_embedder().encode(list(chunk_texts), normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


# ------------------------------- RAG ----------------------------------
def retrieve(query, chunks, index, k=3):
    q = get_embedder().encode([query], normalize_embeddings=True)[0]
    scores = index @ q
    top = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in top]


def answer_extractive(query, retrieved, n_sentences=4):
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


def answer_generative(query, retrieved, history_text=""):
    tok, model = get_generator()
    context = "\n".join(c["text"] for c, _ in retrieved)
    prompt = (
        "Answer the question using only the context below. Give a complete, helpful answer.\n\n"
        f"Context:\n{context}\n\n{history_text}Question: {query}\nAnswer:"
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=160, num_beams=4, early_stopping=True)
    return tok.decode(out[0], skip_special_tokens=True)


# ------------------------------ Memory logic --------------------------
def needs_context(q: str) -> bool:
    ql = q.lower().strip()
    words = set(re.findall(r"[a-z']+", ql))
    return ql.startswith(FOLLOWUP_STARTERS) or bool(PRONOUNS & words)


def find_repeated(query, memory):
    """Return the earlier question if the user asked something very similar."""
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
    """Handle memory-related commands. Returns reply text or None."""
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


def get_answer(query, chunks, index, mode, k, memory):
    """Returns (answer, retrieved, topic, note)."""
    note = ""
    repeated = find_repeated(query, memory)
    if repeated:
        note = f"💭 *You asked something similar earlier (“{repeated}”). Here is the answer again:*\n\n"

    search_query = query
    retrieved = retrieve(search_query, chunks, index, k)
    words = re.findall(r"[A-Za-z']+", query)

    # Follow-up handling: add the previous topic as context
    if memory["last_topic"] and (needs_context(query) or (len(words) <= 3 and retrieved[0][1] < 0.45)):
        search_query = f"{memory['last_topic']}. {query}"
        retrieved = retrieve(search_query, chunks, index, k)
        note += f"🔗 *Follow-up detected, continuing about “{memory['last_topic']}”.*\n\n"

    topic = retrieved[0][0]["title"]
    if retrieved[0][1] < MIN_SCORE:
        return (
            "Sorry, I couldn't find that in my college records. "
            "Please contact the office at info@abcit.edu.in or 040-1234-5678.",
            retrieved, memory["last_topic"], "",
        )

    if mode == "Generative (FLAN-T5)":
        recent = memory["history"][-2:]
        htxt = "".join(f"Earlier question: {h['question']}\n" for h in recent)
        text = answer_generative(search_query, retrieved, htxt)
    else:
        text = answer_extractive(search_query, retrieved)
    return text, retrieved, topic, note


# ------------------------------- UI -----------------------------------
st.set_page_config(page_title="College Information Assistant", page_icon="🎓", layout="centered")
st.title("🎓 AI College Information Assistant")
st.caption("Ask about courses, fees, hostel, bus routes, placements and more. It remembers your conversation!")

if "memory" not in st.session_state:
    st.session_state["memory"] = load_memory()
memory = st.session_state["memory"]

with st.sidebar:
    st.header("⚙️ Settings")
    mode = st.radio("Answer mode", ["Extractive (fast)", "Generative (FLAN-T5)"])
    top_k = st.slider("Chunks to retrieve (k)", 1, 5, 3)
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
    st.subheader("🔎 Explore topics")
    for label, q in TOPICS.items():
        if st.button(label, use_container_width=True, key=f"topic_{label}"):
            st.session_state["pending"] = q

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
        greet = ("Hi! 👋 I'm your college assistant. Use the **Explore topics** buttons on the left "
                 "or ask me anything about the college.")
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

query = st.chat_input("Type your question here...") or st.session_state.pop("pending", None)

if query:
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
                answer, retrieved, topic, note = get_answer(query, chunks, index, mode, top_k, memory)
            full = note + answer
            st.markdown(full)
            sources = [(c["title"], c["source"], s) for c, s in retrieved]
            if show_sources:
                render_sources(sources)
            memory["history"].append(
                {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "question": query,
                 "answer": answer, "topic": topic}
            )
            memory["last_topic"] = topic
            save_memory(memory)

    st.session_state["messages"].append({"role": "assistant", "content": full, "sources": sources})
