"""
Neon (Postgres) database layer using SQLAlchemy async.

Schema:
  users        — Clerk user ID + encrypted OpenAI key
  chats        — per-user chat sessions with status
  messages     — message history per chat
  documents    — uploaded PDF JSON representations
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import (
    Column, String, Text, DateTime, Float, Integer,
    JSON, ForeignKey, select, update, delete
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, relationship
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_async_engine(DATABASE_URL, echo=False, connect_args=_connect_args)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    clerk_id = Column(String, primary_key=True)
    encrypted_openai_key = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")


class Chat(Base):
    __tablename__ = "chats"

    chat_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.clerk_id"), nullable=False)
    title = Column(String, default="New Chat")
    status = Column(String, default="setup")   # setup | processing | ready | failed
    processing_stage = Column(String, default="")
    processing_progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="chats")
    messages = relationship("Message", back_populates="chat", cascade="all, delete-orphan",
                            order_by="Message.timestamp")
    documents = relationship("Document", back_populates="chat", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.chat_id"), nullable=False)
    role = Column(String, nullable=False)          # user | assistant
    content = Column(Text, nullable=False)
    retrieval_method = Column(String, nullable=True)  # graph | rlm | hybrid
    sources = Column(JSON, default=list)
    timestamp = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="messages")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chat_id = Column(String, ForeignKey("chats.chat_id"), nullable=False)
    original_filename = Column(String, nullable=False)
    json_content = Column(JSON, nullable=True)   # extracted document JSON
    page_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    chat = relationship("Chat", back_populates="documents")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ── CRUD helpers ──────────────────────────────────────────────────────────────

async def get_or_create_user(clerk_id: str) -> User:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.clerk_id == clerk_id))
        user = result.scalar_one_or_none()
        if not user:
            user = User(clerk_id=clerk_id)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user


async def store_openai_key(clerk_id: str, encrypted_key: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(User)
            .where(User.clerk_id == clerk_id)
            .values(encrypted_openai_key=encrypted_key)
        )
        await session.commit()


async def get_encrypted_openai_key(clerk_id: str) -> Optional[str]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(User.encrypted_openai_key).where(User.clerk_id == clerk_id)
        )
        row = result.first()
        return row[0] if row else None


async def create_chat(user_id: str, title: str = "New Chat") -> Chat:
    async with AsyncSessionLocal() as session:
        chat = Chat(chat_id=str(uuid.uuid4()), user_id=user_id, title=title)
        session.add(chat)
        await session.commit()
        await session.refresh(chat)
        return chat


async def get_user_chats(user_id: str) -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
        )
        chats = result.scalars().all()

        out = []
        for c in chats:
            doc_result = await session.execute(
                select(Document).where(Document.chat_id == c.chat_id)
            )
            doc_count = len(doc_result.scalars().all())
            out.append({
                "chat_id": c.chat_id,
                "title": c.title,
                "status": c.status,
                "created_at": c.created_at,
                "updated_at": c.updated_at,
                "document_count": doc_count,
            })
        return out


async def get_chat(chat_id: str) -> Optional[Chat]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Chat).where(Chat.chat_id == chat_id)
        )
        return result.scalar_one_or_none()


async def update_chat_status(chat_id: str, status: str, stage: str = "", progress: float = 0.0):
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Chat)
            .where(Chat.chat_id == chat_id)
            .values(status=status, processing_stage=stage,
                    processing_progress=progress, updated_at=datetime.utcnow())
        )
        await session.commit()


async def update_chat_title(chat_id: str, title: str):
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(Chat).where(Chat.chat_id == chat_id).values(title=title)
        )
        await session.commit()


async def save_document(chat_id: str, filename: str, json_content: dict, page_count: int) -> Document:
    async with AsyncSessionLocal() as session:
        doc = Document(
            doc_id=str(uuid.uuid4()),
            chat_id=chat_id,
            original_filename=filename,
            json_content=json_content,
            page_count=page_count,
        )
        session.add(doc)
        await session.commit()
        await session.refresh(doc)
        return doc


async def get_chat_documents(chat_id: str) -> List[Document]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Document).where(Document.chat_id == chat_id)
        )
        return result.scalars().all()


async def add_message(
    chat_id: str,
    role: str,
    content: str,
    retrieval_method: Optional[str] = None,
    sources: Optional[list] = None,
) -> Message:
    async with AsyncSessionLocal() as session:
        msg = Message(
            message_id=str(uuid.uuid4()),
            chat_id=chat_id,
            role=role,
            content=content,
            retrieval_method=retrieval_method,
            sources=sources or [],
        )
        session.add(msg)
        await session.execute(
            update(Chat).where(Chat.chat_id == chat_id).values(updated_at=datetime.utcnow())
        )
        await session.commit()
        await session.refresh(msg)
        return msg


async def get_chat_messages(chat_id: str) -> List[Dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.timestamp)
        )
        msgs = result.scalars().all()
        return [
            {
                "message_id": m.message_id,
                "role": m.role,
                "content": m.content,
                "retrieval_method": m.retrieval_method,
                "sources": m.sources,
                "timestamp": m.timestamp,
            }
            for m in msgs
        ]


async def delete_chat(chat_id: str):
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Chat).where(Chat.chat_id == chat_id))
        await session.commit()


async def verify_chat_owner(chat_id: str, user_id: str) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Chat.user_id).where(Chat.chat_id == chat_id)
        )
        row = result.first()
        return row is not None and row[0] == user_id
