"""Получение эмбеддингов через OpenAI-совместимый Embedding API (Proxy API)."""
import openai
from openai import OpenAI

import config

# Лимит модели — 8191 токен. Для русского текста это грубо 2–3 символа на токен,
# поэтому берём консервативную границу по символам, чтобы не тянуть tiktoken.
MAX_CHARS = 12_000


class EmbeddingError(RuntimeError):
    """Понятная пользователю ошибка получения эмбеддинга."""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Создаём клиент один раз и переиспользуем."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.embedding_api_key(),
            base_url=config.EMBEDDING_BASE_URL,
            timeout=30,
        )
    return _client


def _validate(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise EmbeddingError("Пустой текст: эмбеддинг построить нельзя.")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise EmbeddingError(
            f"Текст слишком длинный ({len(text)} символов, максимум {MAX_CHARS})."
        )
    return text


def _call_api(inputs: list[str]) -> list[list[float]]:
    try:
        response = _get_client().embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=inputs,
            dimensions=config.EMBEDDING_DIMENSION,
        )
    except openai.AuthenticationError as exc:
        raise EmbeddingError("Неверный API-ключ для Embedding API (EMBEDDING_API_KEY).") from exc
    except openai.NotFoundError as exc:
        raise EmbeddingError(
            f"Модель или URL не найдены. Проверьте EMBEDDING_MODEL и EMBEDDING_BASE_URL: {exc}"
        ) from exc
    except openai.RateLimitError as exc:
        raise EmbeddingError("Превышен лимит запросов или закончился баланс API.") from exc
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise EmbeddingError("Embedding API недоступен: проверьте сеть и EMBEDDING_BASE_URL.") from exc
    except openai.APIStatusError as exc:
        raise EmbeddingError(f"Embedding API вернул ошибку {exc.status_code}: {exc.message}") from exc

    # Сортируем по index на случай, если прокси вернёт данные не по порядку.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def get_embedding(text: str) -> list[float]:
    """Принимает текст, возвращает вектор."""
    return _call_api([_validate(text)])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Пакетная версия: один запрос на список текстов."""
    return _call_api([_validate(t) for t in texts])
