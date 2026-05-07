"""
models.py
---------
SQLAlchemy ORM models for Users, Conversations, and Messages.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Boolean
)
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class User(Base):
    """Represents an authenticated user."""

    __tablename__ = "users"

    id: int = Column(Integer, primary_key=True, index=True)
    username: str = Column(String(64), unique=True, index=True, nullable=False)
    email: str = Column(String(128), unique=True, index=True, nullable=False)
    password_hash: str = Column(String(256), nullable=False)
    is_active: bool = Column(Boolean, default=True)
    created_at: datetime = Column(DateTime, default=datetime.utcnow)

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class Conversation(Base):
    """Represents a chat conversation session."""

    __tablename__ = "conversations"

    id: int = Column(Integer, primary_key=True, index=True)
    user_id: int = Column(Integer, ForeignKey("users.id"), nullable=False)
    title: str = Column(String(256), default="New Conversation")
    created_at: datetime = Column(DateTime, default=datetime.utcnow)
    updated_at: datetime = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title!r}>"


class Message(Base):
    """Represents a single message within a conversation."""

    __tablename__ = "messages"

    id: int = Column(Integer, primary_key=True, index=True)
    conversation_id: int = Column(
        Integer, ForeignKey("conversations.id"), nullable=False
    )
    role: str = Column(String(16), nullable=False)   # "user" | "assistant"
    content: str = Column(Text, nullable=False)
    timestamp: datetime = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r}>"
