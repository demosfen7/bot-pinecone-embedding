"""Семантический поиск. Запуск без аргументов прогоняет демо-запросы по базе фактов."""
import sys

import config
from embeddings import ApiError, get_embedding
from pinecone_store import get_index


def semantic_search(
    query: str,
    top_k: int | None = None,
    namespace: str | None = None,
    metadata_filter: dict | None = None,
) -> list[dict]:
    """Возвращает список {id, score, text, topic, metadata} по убыванию релевантности.

    namespace по умолчанию — база фактов; metadata_filter нужен, например,
    чтобы искать только в воспоминаниях конкретного пользователя.
    """
    vector = get_embedding(query)
    kwargs = {"filter": metadata_filter} if metadata_filter else {}
    response = get_index().query(
        vector=vector,
        top_k=top_k or config.SEARCH_TOP_K,
        include_metadata=True,
        namespace=namespace or config.NAMESPACE,
        **kwargs,
    )
    results = []
    for match in response.matches:
        metadata = dict(match.metadata or {})
        results.append(
            {
                "id": match.id,
                "score": match.score,
                "text": metadata.get("text", ""),
                "topic": metadata.get("topic", ""),
                "metadata": metadata,
            }
        )
    return results


# Запросы проверяют смысловые связи, а не совпадение слов; ожидаемые ID — в README.
DEMO_QUERIES = [
    "Что можно хранить тысячелетиями и не бояться порчи?",
    "Необычные явления на других планетах и звёздах",
    "Как металлические конструкции реагируют на погоду?",
    "Выдающиеся учёные и их уникальные достижения",
    "Истоки компьютерной эпохи",
]


def main() -> int:
    queries = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUERIES
    try:
        for query in queries:
            print(f"\nЗапрос: {query}")
            for rank, item in enumerate(semantic_search(query), start=1):
                print(f"  {rank}. [{item['id']}] score={item['score']:.4f} — {item['text']}")
    except (ApiError, config.ConfigError) as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
