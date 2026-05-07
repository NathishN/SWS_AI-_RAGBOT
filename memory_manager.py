"""
memory_manager.py
-----------------
Manages persistent chat memory: saving, retrieving and restoring
conversations and messages from the SQLite database.
"""

import logging
from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from models import Conversation, Message, User

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------


async def create_conversation(
    db: AsyncSession,
    user_id: int,
    title: str = "New Conversation",
) -> Conversation:
    """Create and persist a new conversation for a user."""
    conv = Conversation(user_id=user_id, title=title)
    db.add(conv)
    await db.flush()
    await db.refresh(conv)
    logger.debug("Created conversation id=%d for user_id=%d", conv.id, user_id)
    return conv


async def get_conversation(
    db: AsyncSession, conversation_id: int
) -> Optional[Conversation]:
    """Fetch a single conversation with its messages pre-loaded."""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.messages))
        .where(Conversation.id == conversation_id)
    )
    return result.scalars().first()


async def list_user_conversations(
    db: AsyncSession, user_id: int
) -> List[Conversation]:
    """Return all conversations for a user, ordered newest-first."""
    result = await db.execute(
        select(Conversation)
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
    )
    return list(result.scalars().all())


async def rename_conversation(
    db: AsyncSession, conversation_id: int, title: str
) -> None:
    """Update the title of a conversation."""
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(title=title, updated_at=datetime.utcnow())
    )


async def delete_conversation(
    db: AsyncSession, conversation_id: int
) -> None:
    """Delete a conversation and all its messages (cascade)."""
    await db.execute(
        delete(Conversation).where(Conversation.id == conversation_id)
    )
    logger.info("Deleted conversation id=%d", conversation_id)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------


async def save_message(
    db: AsyncSession,
    conversation_id: int,
    role: str,
    content: str,
) -> Message:
    """Persist a single message and touch the conversation's updated_at."""
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(msg)

    # Touch parent conversation so ordering stays correct
    await db.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .values(updated_at=datetime.utcnow())
    )

    await db.flush()
    await db.refresh(msg)
    return msg


async def get_conversation_messages(
    db: AsyncSession, conversation_id: int
) -> List[Message]:
    """Return all messages for a conversation, ordered chronologically."""
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.timestamp.asc())
    )
    return list(result.scalars().all())


async def build_chat_history(
    db: AsyncSession,
    conversation_id: int,
    max_messages: int = 20,
) -> List[dict]:
    """
    Build a list of ``{role, content}`` dicts suitable for Gemini's
    multi-turn history.  Returns at most *max_messages* recent messages.
    """
    messages = await get_conversation_messages(db, conversation_id)
    recent = messages[-max_messages:]
    return [{"role": m.role, "content": m.content} for m in recent]


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------


async def get_user_by_username(
    db: AsyncSession, username: str
) -> Optional[User]:
    result = await db.execute(
        select(User).where(User.username == username)
    )
    return result.scalars().first()


async def get_user_by_id(db: AsyncSession, user_id: int) -> Optional[User]:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalars().first()


async def create_user(
    db: AsyncSession, username: str, email: str, password_hash: str
) -> User:
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    await db.flush()
    await db.refresh(user)
    logger.info("Created user id=%d username=%r", user.id, username)
    return user
