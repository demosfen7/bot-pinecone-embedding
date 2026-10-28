"""Загрузка фактов в Pinecone пакетами. Повторный запуск безопасен: upsert перезаписывает по ID."""
import sys

import config
from data import FACTS
from embeddings import EmbeddingError, get_embeddings
from pinecone_store import ensure_index, get_client

BATCH_SIZE = 10


def main() -> int:
    ids = [f["id"] for f in FACTS]
    assert len(ids) == len(set(ids)), "ID фактов должны быть уникальными"
    assert len({f["text"] for f in FACTS}) == len(FACTS), "Тексты фактов должны быть уникальными"

    try:
        index = ensure_index(get_client())
        uploaded = 0
        for start in range(0, len(FACTS), BATCH_SIZE):
            batch = FACTS[start:start + BATCH_SIZE]
            vectors = get_embeddings([f["text"] for f in batch])
            index.upsert(
                vectors=[
                    {
                        "id": fact["id"],
                        "values": vector,
                        # Исходный текст храним в metadata, чтобы показывать его в поиске.
                        "metadata": {"text": fact["text"], "topic": fact["topic"]},
                    }
                    for fact, vector in zip(batch, vectors)
                ],
                namespace=config.NAMESPACE,
            )
            uploaded += len(batch)
            print(f"Загружено {uploaded}/{len(FACTS)}")
    except (EmbeddingError, config.ConfigError) as exc:
        print(f"ОШИБКА: {exc}")
        return 1

    stats = index.describe_index_stats()
    count = stats.namespaces.get(config.NAMESPACE, {}).get("vector_count", 0)
    print(f"Готово. Отправлено записей: {uploaded}. В namespace '{config.NAMESPACE}' сейчас: {count}")
    print("Если счётчик меньше — подождите несколько секунд: статистика Pinecone обновляется с задержкой.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
