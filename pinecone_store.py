"""Подключение к Pinecone, создание индекса и проверка его готовности."""
import time
from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

import config


def get_client() -> Pinecone:
    return Pinecone(api_key=config.pinecone_api_key())


@lru_cache(maxsize=1)
def get_index():
    """Находим host индекса один раз, дальше переиспользуем подключение."""
    pc = get_client()
    return pc.Index(host=pc.describe_index(config.INDEX_NAME).host)


def ensure_index(pc: Pinecone, wait_seconds: int = 120):
    """Создаёт индекс, если его нет, ждёт готовности и возвращает объект Index."""
    existing = pc.list_indexes().names()

    if config.INDEX_NAME not in existing:
        print(f"Индекс '{config.INDEX_NAME}' не найден — создаю...")
        pc.create_index(
            name=config.INDEX_NAME,
            dimension=config.EMBEDDING_DIMENSION,
            metric="cosine",  # косинусная близость — стандарт для текстовых эмбеддингов
            spec=ServerlessSpec(cloud=config.CLOUD, region=config.REGION),
        )
    else:
        info = pc.describe_index(config.INDEX_NAME)
        if info.dimension != config.EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Индекс '{config.INDEX_NAME}' имеет размерность {info.dimension}, "
                f"а модель выдаёт {config.EMBEDDING_DIMENSION}. "
                f"Удалите индекс или поменяйте PINECONE_INDEX_NAME."
            )

    # Ждём, пока индекс станет готовым принимать данные.
    deadline = time.time() + wait_seconds
    while not pc.describe_index(config.INDEX_NAME).status["ready"]:
        if time.time() > deadline:
            raise TimeoutError(f"Индекс не стал готов за {wait_seconds} секунд.")
        time.sleep(2)

    info = pc.describe_index(config.INDEX_NAME)
    print(
        f"Индекс '{info.name}' готов: dimension={info.dimension}, "
        f"metric={info.metric}, host={info.host}"
    )
    return pc.Index(host=info.host)
