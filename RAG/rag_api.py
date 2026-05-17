import os
from typing import Optional

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()
messages = []
client = Groq()
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cursor():
    global conn, cursor
    try:
        cursor.execute("SELECT 1")
    except Exception:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        cursor = conn.cursor()
    return cursor


def chunk_text(sentences, chunk_size=2, overlap=1):
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = sentences[i : i + chunk_size]
        chunks.append(". ".join(chunk))
    return chunks

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    with open("index.html") as f:
        return HTMLResponse(f.read())


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    filename = file.filename

    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks = chunk_text(sentences)
    encoded_chunks = model.encode(chunks)

    get_cursor().execute("DELETE FROM rag_documents WHERE source = %s", (filename,))
    conn.commit()

    for chunk, embedding in zip(chunks, encoded_chunks):
        get_cursor().execute(
            "INSERT INTO rag_documents (chunk, embedding, source) VALUES (%s, %s, %s)",
            (chunk, embedding.tolist(), filename),
        )
    conn.commit()

    return {"message": f"Uploaded {len(chunks)} chunks from {filename}"}


@app.post("/chat")
async def chat(request: ChatRequest, source: Optional[str] = None):
    query_vec = model.encode(request.question).tolist()

    if source:
        get_cursor().execute(
            """
            SELECT chunk, embedding <=> %s::vector AS distance
            FROM rag_documents
            WHERE source = %s
            ORDER BY distance ASC
            LIMIT 3
            """,
            (query_vec, source),
        )
    else:
        get_cursor().execute(
            """
            SELECT chunk, embedding <=> %s::vector AS distance
            FROM rag_documents
            ORDER BY distance ASC
            LIMIT 3
            """,
            (query_vec,),
        )

    results = cursor.fetchall()

    context = "\n\n".join([chunk for chunk, _ in results])
    prompt = f"""Answer the question using only the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {request.question}
"""

    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [chunk for chunk, _ in results],
    }


@app.get("/documents")
async def list_documents():
    get_cursor().execute("SELECT DISTINCT source FROM rag_documents;")
    docs = cursor.fetchall()
    return {"documents": [doc[0] for doc in docs]}


@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    get_cursor().execute(
        "DELETE FROM rag_documents WHERE source = %s",
        (filename,),
    )
    conn.commit()
    return {"rows_deleted": cursor.rowcount}