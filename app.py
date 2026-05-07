"""
app.py
------
FastAPI application: serves the chat UI, handles file uploads, chat,
conversation management, and user authentication.
"""

import logging
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from jose import JWTError, jwt
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db, init_db, close_db
from memory_manager import (
    build_chat_history,
    create_conversation,
    create_user,
    delete_conversation,
    get_conversation,
    get_conversation_messages,
    get_user_by_username,
    list_user_conversations,
    rename_conversation,
    save_message,
)
from pdf_reader import read_pdfs
from data_preprocessor import preprocess_pages, merge_pages
from chunker import chunk_text
from vector_store import add_documents, get_collection_stats
from query_processor import process_query

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Chatbot API",
    description="Enterprise Multi-Document RAG Chatbot powered by Gemini + ChromaDB",
    version="1.0.0",
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
        ) from exc


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[int] = None
    user_id: int


class NewConversationRequest(BaseModel):
    user_id: int
    title: str = "New Conversation"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def on_startup() -> None:
    await init_db()
    logger.info("Application started.")


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_db()
    logger.info("Application stopped.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse("chatui.html", {"request": request})


@app.get("/health")
async def health():
    stats = get_collection_stats()
    return {"status": "ok", "vector_store": stats}


# --- Auth ---


@app.post("/register")
async def register(
    payload: RegisterRequest, db: AsyncSession = Depends(get_db)
):
    existing = await get_user_by_username(db, payload.username)
    if existing:
        raise HTTPException(
            status_code=400, detail="Username already registered."
        )
    user = await create_user(
        db,
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password),
    )
    token = create_access_token({"sub": str(user.id)})
    return {
        "user_id": user.id,
        "username": user.username,
        "access_token": token,
    }


@app.post("/login")
async def login(payload: LoginRequest, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_username(db, payload.username)
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=401, detail="Incorrect username or password."
        )
    token = create_access_token({"sub": str(user.id)})
    return {
        "user_id": user.id,
        "username": user.username,
        "access_token": token,
    }


# --- Document Upload ---


@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
):
    """
    Accept one or more PDF files, extract text, chunk, embed, and store
    in ChromaDB.  Returns a summary of processed documents.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    saved_paths: List[Path] = []
    for upload in files:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=422,
                detail=f"Only PDF files are accepted. Got: {upload.filename}",
            )

        dest = UPLOAD_DIR / f"{uuid.uuid4()}_{upload.filename}"
        async with aiofiles.open(dest, "wb") as f:
            content = await upload.read()
            if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=413,
                    detail=f"{upload.filename} exceeds {MAX_FILE_SIZE_MB} MB limit.",
                )
            await f.write(content)
        saved_paths.append(dest)
        logger.info("Saved upload: %s", dest.name)

    # Process PDFs
    docs = read_pdfs(saved_paths)
    results = []
    total_chunks = 0

    for doc in docs:
        if not doc.success:
            results.append(
                {"filename": doc.filename, "status": "error", "detail": doc.error}
            )
            continue

        cleaned_pages = preprocess_pages(doc.pages)
        full_text = merge_pages(cleaned_pages)
        chunks = chunk_text(full_text, source=doc.filename)
        n = add_documents(chunks)
        total_chunks += n
        results.append(
            {
                "filename": doc.filename,
                "status": "ok",
                "pages": doc.page_count,
                "chunks": n,
            }
        )
        logger.info("Indexed %s → %d chunks", doc.filename, n)

    return {
        "processed": len(docs),
        "total_chunks_indexed": total_chunks,
        "documents": results,
    }


# --- Chat ---


@app.post("/chat")
async def chat(
    payload: ChatRequest, db: AsyncSession = Depends(get_db)
):
    """
    Handle a user chat message.  Runs the RAG pipeline and persists the
    conversation to SQLite.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Resolve or create conversation
    if payload.conversation_id:
        conv = await get_conversation(db, payload.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")
        conversation_id = conv.id
    else:
        # Auto-title from first N words of the query
        title = " ".join(query.split()[:6]) + ("…" if len(query.split()) > 6 else "")
        conv = await create_conversation(db, user_id=payload.user_id, title=title)
        conversation_id = conv.id

    # Build chat history for multi-turn context
    history = await build_chat_history(db, conversation_id)

    # Run RAG pipeline
    try:
        result = process_query(query, chat_history=history)
    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist messages
    await save_message(db, conversation_id, role="user", content=query)
    await save_message(
        db, conversation_id, role="assistant", content=result["answer"]
    )

    return {
        "conversation_id": conversation_id,
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks_used": result["chunks_used"],
    }


# --- Conversation Management ---


@app.get("/history/{user_id}")
async def get_history(user_id: int, db: AsyncSession = Depends(get_db)):
    """Return all conversations for a user (ordered newest-first)."""
    convs = await list_user_conversations(db, user_id)
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]


@app.get("/conversation/{conversation_id}")
async def get_conv_messages(
    conversation_id: int, db: AsyncSession = Depends(get_db)
):
    """Return all messages in a conversation."""
    msgs = await get_conversation_messages(db, conversation_id)
    return [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "timestamp": m.timestamp.isoformat(),
        }
        for m in msgs
    ]


@app.delete("/conversation/{conversation_id}")
async def delete_conv(
    conversation_id: int, db: AsyncSession = Depends(get_db)
):
    """Delete a conversation and all its messages."""
    await delete_conversation(db, conversation_id)
    return {"deleted": conversation_id}


@app.post("/conversation/new")
async def new_conversation(
    payload: NewConversationRequest, db: AsyncSession = Depends(get_db)
):
    conv = await create_conversation(
        db, user_id=payload.user_id, title=payload.title
    )
    return {"id": conv.id, "title": conv.title}


@app.get("/stats")
async def stats():
    """Return vector store statistics."""
    return get_collection_stats()
