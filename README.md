# AI College Information Assistant (with memory)

## Setup (Python 3.12)
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```
First run downloads models (internet needed once). Run once tonight so it works offline tomorrow.

## Customize
Edit `data/college_info.md` with your real college details. Keep the `## Heading` format
(one topic per heading, short paragraphs written as full sentences).

## Memory
Saved in `chat_memory.json` (delete the file or use "Clear memory" to reset before the demo).
Try: "What are the hostel fees?" -> "What about its timings?" -> "What did I ask before?"
