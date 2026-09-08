# 🧠 Interview Questions Generator
Generate a tailored set of technical interview questions — grouped by competency area, each paired with a grounded sample answer — from any job description PDF.
 
Upload a JD, and the app retrieves the parts of it that actually matter, asks an LLM to draft realistic interview questions per competency area, then writes a sample answer for each question grounded in the specific JD text it's about.
 
Built primarily as a hands-on project for practicing Retrieval-Augmented Generation (RAG) patterns with LangChain, FAISS, and Google's Gemini models — and secondarily because I use it to prep for my own interviews.

## Demo
<img width="1919" height="911" alt="image" src="https://github.com/user-attachments/assets/1320b165-f6a9-4f3d-a6f5-0d68ff454519" />
<img width="1915" height="916" alt="image" src="https://github.com/user-attachments/assets/33f8f4d0-39d5-45d9-bab4-eb6defe37d3c" />
<img width="1918" height="911" alt="image" src="https://github.com/user-attachments/assets/a0ada6e9-d483-4a28-ac4b-1e6757e486c8" />

## How It Works
 
```mermaid
flowchart LR
    A[Upload JD PDF] --> B[Chunk text]
    B --> C[Embed chunks<br/>gemini-embedding-001]
    C --> D[(FAISS vector store)]
    D --> E[Retrieve JD context]
    E --> F[Generate questions<br/>1 structured LLM call]
    F --> G[Retrieve context<br/>per question]
    G --> H[Generate all answers<br/>1 structured LLM call]
    H --> I[Q&A grouped by category]
```


## Tech Stack
 
- **LangChain** — orchestration (loaders, text splitters, prompt templates, structured output)
- **FAISS** — local vector similarity search
- **Google Gemini** (`gemini-3.6-flash` + `gemini-embedding-001`) — generation and embeddings
- **Pydantic** — structured output schemas for questions and answers
- **Streamlit** — web UI
- **pypdf** — PDF text extraction

## Getting Started
 
### Prerequisites
- Python 3.10+
- A [Google AI Studio](https://aistudio.google.com/) API key (free tier works)
### Installation
 
```bash
git clone https://github.com/kxvyx/Interview-Questions-Generator-using-langchain.git
cd Interview-Questions-Generator-using-langchain
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
 
> `requirements.txt` is currently a full environment freeze rather than a curated dependency list — see [Known Limitations](#known-limitations--assumptions).
 
### Configuration
 
Create a `.env` file in the project root:
 
```
GEMINI_API_KEY=your-api-key-here
```
 
Verify it's picked up correctly, at zero API cost:
 
```bash
python test_env.py
```
 
### Usage
 
**Web app (recommended)** — upload any JD PDF and get results in the browser:
```bash
streamlit run streamlit_app.py
```
 
**CLI** — no sample PDF is included in the repo (see `.gitignore`); place your own JD PDF in the project root named `Deloitte_JD.pdf`, then run:
```bash
python main.py
```

## Design Decisions
 
A few choices here were deliberate, not accidental:
 
- **Two bulk LLM calls, not one per question.** All questions are generated in a single structured-output call, and all answers in another — instead of 25+ individual calls. This trades a slightly longer wait per run for staying well within free-tier daily request limits.
- **Per-question retrieval, shared generation call.** Each question still gets its own FAISS lookup so its answer is grounded in the JD text it's actually about, but all (question, context) pairs are batched into one prompt rather than one call per question — precise retrieval without the API cost of a call per question.
- **The FAISS cache is invalidated by a content fingerprint, not just "does the folder exist."** A naive `if os.path.exists(index_path): load it` cache will happily serve embeddings from a completely different PDF. This cache is keyed on embedding model + chunk count, so a new or edited document rebuilds it automatically.

## Known Limitations & Assumptions
 
- **Text-extractable PDFs only.** No OCR — a scanned or image-only JD will produce empty or garbled chunks.
- **One document at a time.** Not built for batch-processing multiple JDs in a single run.
- **Shared cache path across uploads.** The Streamlit app and the CLI both write to the same `faiss_index/` path. The cache key (embedding model + chunk count) makes accidental reuse unlikely but not impossible — two different PDFs that happen to produce the same chunk count would collide. A per-file content hash would close this gap.
- **No persistence across hosting restarts.** The FAISS cache lives on local disk; deployed somewhere with an ephemeral filesystem (e.g. Streamlit Community Cloud's default), it rebuilds on every restart.
- **`requirements.txt` is a full `pip freeze`,** not a curated list — it includes packages left over from the broader environment it was generated in, not things this project actually imports.
- **No automated tests** beyond `test_env.py`, which only checks that the API key loads.

## Roadmap
 
- [ ] Scope the FAISS cache per uploaded file (hash the PDF content instead of a shared path)
- [ ] Stream questions/answers into the UI as they're generated instead of blocking on the full batch
- [ ] Trim `requirements.txt` to direct dependencies only
- [ ] Add pytest coverage for chunking, caching, and answer-matching logic
- [ ] Let users adjust category/question counts from the UI instead of hardcoded values
- [ ] Export the generated Q&A set to PDF/Markdown for offline studying
- [ ] Support additional LLM providers behind the same interface
