# 🎓 AI College Information Assistant

A chatbot that answers questions about a college (admissions, fees, hostel, bus routes, placements, library and more) using **RAG (Retrieval Augmented Generation)**. It remembers past questions and understands follow-ups.

## ✨ Features
- Answers from the college's own data, so it doesn't make things up
- Semantic search with **Sentence Transformers** (understands meaning, not just keywords)
- Two answer modes: **Extractive** (fast) and **Generative** (FLAN-T5)
- **Conversation memory:** follow-up questions, repeated-question recognition, remembers the user's name and past questions
- Memory saved to a file, so it survives restarts
- Shows the sources used for each answer
- Upload extra `.txt` / `.md` documents from the sidebar

## 🛠️ Tech Stack
Python 3.12 · Streamlit · Sentence Transformers · Transformers (FLAN-T5) · PyTorch · NumPy · Pandas

## 📁 Project Structure
```
college-information-assistant/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── data/
    └── college_info.md   # College knowledge base
```

## 🚀 Installation and Run

1. **Clone or download** this repository
   ```
   git clone https://github.com/YOUR-USERNAME/college-information-assistant.git
   cd college-information-assistant
   ```
2. **Create a virtual environment**
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac / Linux
   ```
3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
4. **Run the app**
   ```
   streamlit run app.py
   ```
5. Open **http://localhost:8501** in your browser.

> The first run downloads the AI models and needs internet. After that it works offline.

## 💬 Example Questions
- What courses does the college offer?
- What are the hostel fees?
- What about its timings? *(follow-up)*
- What are the bus routes?
- What is the minimum attendance required?
- What did I ask before?

## ⚙️ How It Works
1. College data is split into small **chunks** by topic.
2. Each chunk is converted into a **vector embedding** (all-MiniLM-L6-v2).
3. The user's question is embedded and the **most similar chunks are retrieved**.
4. The answer is built from those chunks (extractive or FLAN-T5 generation).
5. The question, topic and answer are saved in **memory** for follow-ups and recall.

## 🔧 Customization
Edit `data/college_info.md` with your own college details. Use `## Heading` for each topic and write short paragraphs in full sentences.

## 🔮 Future Scope
Voice input and output · Multi-language support · PDF upload · Larger LLM integration · Admin panel to update data

## 👤 Author
**Your Name** · Your College · Project Expo 2026

## 🦙 Llama + RAG Version (`app_llama.py`)
This version uses **Llama running locally through Ollama** to write the final answer from the retrieved chunks.

1. Install Ollama from https://ollama.com
2. Download a Llama model (pick one):
   ```
   ollama pull llama3.2        # 3B, good quality (about 2 GB)
   ollama pull llama3.2:1b     # 1B, faster on weak laptops
   ```
3. Run the app:
   ```
   streamlit run app_llama.py
   ```
The sidebar shows whether Ollama is running. If Llama is unavailable, the app falls back to the simple extractive answer.
