# CollegeGPT

A RAG (Retrieval-Augmented Generation) application that lets you upload your own documents and ask questions from them. Built with FastAPI, pgvector, and Groq.

---

## What It Does

- Upload any `.txt` document — lecture notes, textbooks, research papers
- CollegeGPT chunks and embeds the document into a vector database
- Ask questions in natural language — the app retrieves the most relevant passages and generates a grounded answer
- Answers are based strictly on your documents — no hallucination from the model's own knowledge
- Filter chat by specific document when you have multiple files uploaded

---

## Live Demo

🚀 Coming soon

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | pgvector on Neon.tech |
| Frontend | Vanilla HTML + CSS + JS |
| Deployment | Railway |

---

## Project Structure

```
collegegpt/
├── rag_api.py        # FastAPI backend
├── index.html        # Frontend UI
├── requirements.txt  # Python dependencies
├── railway.toml      # Railway deployment config
├── .env              # Environment variables (not committed)
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Frontend UI |
| `POST` | `/upload` | Upload and index a `.txt` file |
| `POST` | `/chat` | Ask a question (optional `?source=filename`) |
| `GET` | `/documents` | List all indexed documents |
| `DELETE` | `/documents/{filename}` | Delete a document and its chunks |

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/Pratham-K-890/collegegpt.git
cd collegegpt
```

**2. Create a virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the project root:
```
DATABASE_URL=your_neon_postgresql_connection_string
GROQ_API_KEY=your_groq_api_key
```

**5. Set up pgvector on Neon**

Run this in your Neon SQL editor:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE rag_documents (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    chunk TEXT,
    embedding vector(384),
    source VARCHAR
);
```

**6. Start the server**
```bash
uvicorn rag_api:app --reload
```

**7. Open in browser**
```
http://localhost:8000
```

---

## How It Works

1. **Chunking** — uploaded documents are split into overlapping 2-sentence chunks to preserve context across sentence boundaries
2. **Embedding** — each chunk is converted into a 384-dimensional vector using `all-MiniLM-L6-v2`
3. **Storage** — vectors are stored in PostgreSQL with the pgvector extension on Neon.tech
4. **Retrieval** — user queries are embedded and compared against stored vectors using cosine distance (`<=>` operator)
5. **Generation** — top 3 retrieved chunks are passed to Groq as context, with the instruction to answer only from the provided context

---

## Environment Variables

| Variable | Description |
|---|---|
| `DATABASE_URL` | Neon PostgreSQL connection string |
| `GROQ_API_KEY` | Groq API key from console.groq.com |

---

## Author

**Pratham K** — [@Pratham-K-890](https://github.com/Pratham-K-890)
