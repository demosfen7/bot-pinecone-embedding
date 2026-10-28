"""Долговременная память пользователя в Pinecone: записать, прочитать, удалить векторы.

Воспоминания лежат в отдельном namespace. ID записи начинается с id пользователя
(`<user_id>-<случайный хвост>`), поэтому по префиксу можно перечислить все записи
пользователя, а по префиксу при удалении — убедиться, что чужое не тронуто.
"""
import uuid
from datetime import datetime, timezone

import config
from embeddings import get_embedding
from pinecone_store import get_index
from search import semantic_search

# Если новое воспоминание почти совпадает со старым, второй копии не создаём.
DUPLICATE_SCORE = 0.95


def _prefix(user_id: int) -> str:
    return f"{user_id}-"


def _user_filter(user_id: int) -> dict:
    return {"user_id": {"$eq": str(user_id)}}


def remember(user_id: int, text: str) -> dict:
    """Сохраняет воспоминание. Возвращает {id, text, duplicate}."""
    text = text.strip()
    vector = get_embedding(text)

    # Защита от дублей: ищем самое близкое существующее воспоминание.
    nearest = get_index().query(
        vector=vector,
        top_k=1,
        include_metadata=True,
        namespace=config.MEMORY_NAMESPACE,
        filter=_user_filter(user_id),
    ).matches
    if nearest and nearest[0].score >= DUPLICATE_SCORE:
        return {"id": nearest[0].id, "text": nearest[0].metadata.get("text", ""), "duplicate": True}

    memory_id = f"{_prefix(user_id)}{uuid.uuid4().hex[:12]}"
    get_index().upsert(
        vectors=[
            {
                "id": memory_id,
                "values": vector,
                "metadata": {
                    "text": text,
                    "user_id": str(user_id),
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
            }
        ],
        namespace=config.MEMORY_NAMESPACE,
    )
    return {"id": memory_id, "text": text, "duplicate": False}


def recall(user_id: int, query: str, top_k: int = 3) -> list[dict]:
    """Смысловой поиск по воспоминаниям пользователя."""
    return semantic_search(
        query,
        top_k=top_k,
        namespace=config.MEMORY_NAMESPACE,
        metadata_filter=_user_filter(user_id),
    )


def list_memories(user_id: int) -> list[dict]:
    """Все воспоминания пользователя: [{id, text, created_at}], новые первыми."""
    index = get_index()
    ids: list[str] = []
    for page in index.list(prefix=_prefix(user_id), namespace=config.MEMORY_NAMESPACE):
        ids.extend(page)
    if not ids:
        return []

    items = []
    for start in range(0, len(ids), 100):  # fetch принимает ограниченное число ID за раз
        fetched = index.fetch(ids=ids[start:start + 100], namespace=config.MEMORY_NAMESPACE)
        for memory_id, vector in fetched.vectors.items():
            metadata = vector.metadata or {}
            items.append(
                {
                    "id": memory_id,
                    "text": metadata.get("text", ""),
                    "created_at": metadata.get("created_at", ""),
                }
            )
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


def forget(user_id: int, memory_ids: list[str]) -> dict:
    """Удаляет воспоминания по ID. Чужие и неподходящие ID пропускаются."""
    own = [m for m in memory_ids if m.startswith(_prefix(user_id))]
    skipped = [m for m in memory_ids if m not in own]
    if own:
        get_index().delete(ids=own, namespace=config.MEMORY_NAMESPACE)
    return {"deleted": own, "skipped": skipped}
