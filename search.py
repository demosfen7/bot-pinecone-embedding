"""Семантический поиск по фактам. Запуск без аргументов прогоняет демо-запросы."""
import sys
from functools import lru_cache

import config
from embeddings import EmbeddingError, get_embedding
from pinecone_store import get_client


@lru_cache(maxsize=1)
def _get_index():
    """Находим host индекса один раз, дальше переиспользуем подключение."""
    pc = get_client()
    return pc.Index(host=pc.describe_index(config.INDEX_NAME).host)


def semantic_search(query: str, top_k: int | None = None) -> list[dict]:
    """Возвращает список {id, score, text, topic}, отсортированный по убыванию релевантности."""
    vector = get_embedding(query)
    response = _get_index().query(
        vector=vector,
        top_k=top_k or config.SEARCH_TOP_K,
        include_metadata=True,
        namespace=config.NAMESPACE,
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": (match.metadata or {}).get("text", ""),
            "topic": (match.metadata or {}).get("topic", ""),
        }
        for match in response.matches
    ]


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
    except (EmbeddingError, config.ConfigError) as exc:
        print(f"ОШИБКА: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
