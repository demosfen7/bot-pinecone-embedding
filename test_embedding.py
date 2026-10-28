"""Проверка: эмбеддинг одной тестовой фразы создаётся."""
import sys

import config
from embeddings import EmbeddingError, get_embedding

TEST_PHRASE = "Осьминог имеет три сердца."


def main() -> int:
    try:
        vector = get_embedding(TEST_PHRASE)
    except (EmbeddingError, config.ConfigError) as exc:
        print(f"ОШИБКА: {exc}")
        return 1

    print(f"Фраза: {TEST_PHRASE}")
    print(f"Модель: {config.EMBEDDING_MODEL}")
    print(f"Размерность вектора: {len(vector)}")
    print(f"Первые 5 значений: {[round(v, 5) for v in vector[:5]]}")

    if len(vector) != config.EMBEDDING_DIMENSION:
        print(f"ВНИМАНИЕ: ожидалась размерность {config.EMBEDDING_DIMENSION}, проверьте EMBEDDING_DIMENSION.")
        return 1
    print("OK: вектор успешно создан.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
