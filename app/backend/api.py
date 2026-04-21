"""
GraphReason FastAPI backend.

Endpoints:
  POST /api/auth/store-key              — Store user's encrypted OpenAI API key
  GET  /api/auth/has-key                — Check if user has a stored key
  POST /api/chats                       — Create new chat session
  GET  /api/chats                       — List user's chats
  GET  /api/chats/{chat_id}             — Get chat detail + messages
  DELETE /api/chats/{chat_id}           — Delete chat + graph
  POST /api/chats/{chat_id}/upload      — Upload PDFs (multipart, ≤10 files)
  POST /api/chats/{chat_id}/process     — Trigger full processing pipeline
  GET  /api/chats/{chat_id}/status      — Poll processing status
  GET  /api/chats/{chat_id}/graph       — Graph summary (entities/relationships count)
  POST /api/chats/{chat_id}/messages    — Send a chat message (returns assistant reply)
  GET  /api/chats/{chat_id}/messages    — Get full message history

All endpoints (except health) require Clerk JWT in Authorization header.
"""

import os
import asyncio
import tempfile
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import database as db
import auth as clerk_auth
import graph_builder
from models import (
    APIKeyRequest, APIKeyResponse,
    CreateChatResponse, ProcessingStatus, ChatStatus,
    ChatListItem, ChatDetail, ChatRequest, ChatResponse, GraphSummary,
)
from pdf_pipeline import extract_pdf_to_json
from rlm_analysis import analyze_document, analyze_cross_document
from graph_builder import build_graph, get_graph_summary
from graph_retrieval import graph_search
from chat_engine import answer_question

load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    print("GraphReason API ready")
    yield
    await graph_builder.close_driver()
    print("Shutdown complete")


app = FastAPI(title="GraphReason API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "GraphReason API"}


# ── Auth / API key management ─────────────────────────────────────────────────

@app.post("/api/auth/store-key", response_model=APIKeyResponse)
async def store_api_key(
    request: APIKeyRequest,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    await db.get_or_create_user(clerk_id)
    encrypted = clerk_auth.encrypt_key(request.openai_api_key)
    await db.store_openai_key(clerk_id, encrypted)
    return APIKeyResponse(has_key=True, message="API key stored successfully.")


@app.get("/api/auth/has-key", response_model=APIKeyResponse)
async def has_api_key(clerk_id: str = Depends(clerk_auth.get_current_user)):
    await db.get_or_create_user(clerk_id)
    encrypted = await db.get_encrypted_openai_key(clerk_id)
    has_key = encrypted is not None and len(encrypted) > 0
    return APIKeyResponse(has_key=has_key, message="Key found." if has_key else "No key stored.")


async def _get_openai_key(clerk_id: str) -> str:
    encrypted = await db.get_encrypted_openai_key(clerk_id)
    if not encrypted:
        raise HTTPException(
            status_code=400,
            detail="No OpenAI API key stored. Please add your key in Settings."
        )
    return clerk_auth.decrypt_key(encrypted)


# ── Chat management ───────────────────────────────────────────────────────────

@app.post("/api/chats", response_model=CreateChatResponse)
async def create_chat(clerk_id: str = Depends(clerk_auth.get_current_user)):
    await db.get_or_create_user(clerk_id)
    chat = await db.create_chat(clerk_id)
    return CreateChatResponse(chat_id=chat.chat_id, message="Chat created.")


@app.get("/api/chats", response_model=List[ChatListItem])
async def list_chats(clerk_id: str = Depends(clerk_auth.get_current_user)):
    chats = await db.get_user_chats(clerk_id)
    return [ChatListItem(**c) for c in chats]


@app.get("/api/chats/{chat_id}", response_model=ChatDetail)
async def get_chat(
    chat_id: str,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")
    chat = await db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")
    messages = await db.get_chat_messages(chat_id)
    docs = await db.get_chat_documents(chat_id)
    return ChatDetail(
        chat_id=chat.chat_id,
        title=chat.title,
        status=chat.status,
        created_at=chat.created_at,
        messages=messages,
        document_count=len(docs),
    )


@app.delete("/api/chats/{chat_id}")
async def delete_chat(
    chat_id: str,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")
    # Clean up Neo4j graph
    try:
        from graph_builder import get_driver
        driver = get_driver()
        async with driver.session() as session:
            await session.run(
                "MATCH (n {chat_id: $chat_id}) DETACH DELETE n",
                chat_id=chat_id,
            )
    except Exception:
        pass
    await db.delete_chat(chat_id)
    return {"message": "Chat deleted."}


# ── Document upload ───────────────────────────────────────────────────────────

@app.post("/api/chats/{chat_id}/upload")
async def upload_documents(
    chat_id: str,
    files: List[UploadFile] = File(...),
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 documents per chat.")

    # Validate file types
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail=f"Only PDF files are accepted. Got: {f.filename}"
            )

    # Save file names (actual processing happens in /process)
    saved = []
    upload_dir = Path(tempfile.gettempdir()) / "graphreason" / chat_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        dest = upload_dir / f.filename
        with open(dest, "wb") as out:
            content = await f.read()
            out.write(content)
        saved.append(f.filename)

    return {
        "chat_id": chat_id,
        "uploaded": saved,
        "message": f"Uploaded {len(saved)} file(s). Call /process to start analysis."
    }


# ── Processing pipeline ───────────────────────────────────────────────────────

async def _run_pipeline(chat_id: str, openai_api_key: str):
    """
    Full processing pipeline — runs in background.
    Stages: extraction → RLM analysis → cross-doc analysis → graph construction
    """
    upload_dir = Path(tempfile.gettempdir()) / "graphreason" / chat_id
    pdf_files = list(upload_dir.glob("*.pdf"))

    if not pdf_files:
        await db.update_chat_status(chat_id, "failed", "No PDF files found.", 0)
        return

    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=openai_api_key)

    total = len(pdf_files)
    per_doc_analyses = []

    # ── Stage 1 + 2: Extract PDFs + RLM per-document analysis ──────────────
    for i, pdf_path in enumerate(pdf_files):
        try:
            progress_pct = (i / total) * 50  # first 50% is extraction + per-doc
            await db.update_chat_status(
                chat_id, "processing",
                f"Extracting {pdf_path.name} ({i+1}/{total})",
                progress_pct,
            )

            # Stage 1-3: PDF → structured JSON
            doc_json = await extract_pdf_to_json(str(pdf_path), client)

            # Save to Neon
            await db.save_document(
                chat_id=chat_id,
                filename=pdf_path.name,
                json_content=doc_json,
                page_count=doc_json.get("pages", 0),
            )

            await db.update_chat_status(
                chat_id, "processing",
                f"Analyzing {pdf_path.name} with RLM ({i+1}/{total})",
                progress_pct + (25 / total),
            )

            # RLM analysis
            analysis = await analyze_document(doc_json, openai_api_key, pdf_path.name)
            per_doc_analyses.append(analysis)

        except Exception as e:
            print(f"[pipeline] Error processing {pdf_path.name}: {e}")
            # Continue with other docs; don't fail the whole pipeline

    if not per_doc_analyses:
        await db.update_chat_status(chat_id, "failed", "All document analyses failed.", 0)
        return

    # ── Stage 3: Cross-document analysis ────────────────────────────────────
    await db.update_chat_status(chat_id, "processing", "Cross-document analysis...", 75)
    try:
        cross_analysis = await analyze_cross_document(per_doc_analyses, openai_api_key)
    except Exception as e:
        print(f"[pipeline] Cross-doc analysis failed: {e}")
        cross_analysis = {}

    # ── Stage 4: Build Neo4j graph ───────────────────────────────────────────
    await db.update_chat_status(chat_id, "processing", "Building knowledge graph...", 85)
    try:
        await build_graph(chat_id, per_doc_analyses, cross_analysis)
    except Exception as e:
        print(f"[pipeline] Graph construction failed: {e}")
        # Graph failure is non-fatal; chat still works with RLM only

    # ── Done ─────────────────────────────────────────────────────────────────
    doc_names = [p.name for p in pdf_files]
    title = doc_names[0].replace(".pdf", "") if len(doc_names) == 1 else f"{len(doc_names)} documents"
    await db.update_chat_title(chat_id, title)
    await db.update_chat_status(chat_id, "ready", "Processing complete.", 100)

    # Clean up temp files
    try:
        shutil.rmtree(upload_dir)
    except Exception:
        pass


@app.post("/api/chats/{chat_id}/process")
async def process_documents(
    chat_id: str,
    background_tasks: BackgroundTasks,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")

    chat = await db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")
    if chat.status == "processing":
        raise HTTPException(status_code=409, detail="Already processing.")

    openai_api_key = await _get_openai_key(clerk_id)
    await db.update_chat_status(chat_id, "processing", "Starting pipeline...", 0)
    background_tasks.add_task(_run_pipeline, chat_id, openai_api_key)

    return {"message": "Processing started. Poll /status for updates."}


@app.get("/api/chats/{chat_id}/status", response_model=ProcessingStatus)
async def get_status(
    chat_id: str,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")
    chat = await db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return ProcessingStatus(
        chat_id=chat_id,
        status=chat.status,
        stage=chat.processing_stage or "",
        progress=chat.processing_progress or 0.0,
    )


# ── Graph summary ─────────────────────────────────────────────────────────────

@app.get("/api/chats/{chat_id}/graph", response_model=GraphSummary)
async def get_graph(
    chat_id: str,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")
    try:
        summary = await get_graph_summary(chat_id)
        return GraphSummary(**summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {e}")


# ── Chat messages ─────────────────────────────────────────────────────────────

@app.post("/api/chats/{chat_id}/messages", response_model=ChatResponse)
async def send_message(
    chat_id: str,
    request: ChatRequest,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")

    chat = await db.get_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found.")
    if chat.status != "ready":
        raise HTTPException(
            status_code=409,
            detail=f"Chat is not ready yet. Current status: {chat.status}"
        )

    openai_api_key = await _get_openai_key(clerk_id)

    # Save user message
    await db.add_message(chat_id, "user", request.message)

    # Get conversation history for context
    history = await db.get_chat_messages(chat_id)

    # Generate answer
    result = await answer_question(
        question=request.message,
        chat_id=chat_id,
        openai_api_key=openai_api_key,
        conversation_history=history[:-1],  # exclude the just-saved user message
    )

    # Save assistant message
    await db.add_message(
        chat_id=chat_id,
        role="assistant",
        content=result["message"],
        retrieval_method=result["retrieval_method"],
        sources=result["sources"],
    )

    return ChatResponse(**result)


@app.get("/api/chats/{chat_id}/messages")
async def get_messages(
    chat_id: str,
    clerk_id: str = Depends(clerk_auth.get_current_user),
):
    if not await db.verify_chat_owner(chat_id, clerk_id):
        raise HTTPException(status_code=403, detail="Not your chat.")
    messages = await db.get_chat_messages(chat_id)
    return {"chat_id": chat_id, "messages": messages}
