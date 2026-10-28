"""Клиент OpenAI-совместимого API: эмбеддинги и общая обработка ошибок."""
from contextlib import contextmanager

import openai
from openai import OpenAI

import config

# Лимит модели — 8191 токен. Для русского текста это грубо 2–3 символа на токен,
# поэтому берём консервативную границу по символам, чтобы не тянуть tiktoken.
MAX_CHARS = 12_000


class ApiError(RuntimeError):
    """Понятная пользователю ошибка обращения к OpenAI API."""


# Старое имя оставлено, чтобы не переписывать скрипты проверки.
EmbeddingError = ApiError

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Создаём клиент один раз и переиспользуем (и для эмбеддингов, и для чата)."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=config.embedding_api_key(),
            base_url=config.EMBEDDING_BASE_URL,
            timeout=60,
        )
    return _client


@contextmanager
def api_errors():
    """Превращает исключения openai в ApiError с понятным текстом."""
    try:
        yield
    except openai.AuthenticationError as exc:
        raise ApiError("Неверный API-ключ OpenAI (EMBEDDING_API_KEY).") from exc
    except openai.NotFoundError as exc:
        raise ApiError(
            f"Модель или URL не найдены. Проверьте EMBEDDING_MODEL, CHAT_MODEL и EMBEDDING_BASE_URL: {exc}"
        ) from exc
    except openai.RateLimitError as exc:
        raise ApiError("Превышен лимит запросов или закончился баланс API.") from exc
    except (openai.APIConnectionError, openai.APITimeoutError) as exc:
        raise ApiError("API недоступен: проверьте сеть и EMBEDDING_BASE_URL.") from exc
    except openai.APIStatusError as exc:
        raise ApiError(f"API вернул ошибку {exc.status_code}: {exc.message}") from exc


def _validate(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ApiError("Пустой текст: эмбеддинг построить нельзя.")
    text = text.strip()
    if len(text) > MAX_CHARS:
        raise ApiError(f"Текст слишком длинный ({len(text)} символов, максимум {MAX_CHARS}).")
    return text


def _call_api(inputs: list[str]) -> list[list[float]]:
    with api_errors():
        response = get_client().embeddings.create(
            model=config.EMBEDDING_MODEL,
            input=inputs,
            dimensions=config.EMBEDDING_DIMENSION,
        )
    # Сортируем по index на случай, если прокси вернёт данные не по порядку.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def get_embedding(text: str) -> list[float]:
    """Принимает текст, возвращает вектор."""
    return _call_api([_validate(text)])[0]


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Пакетная версия: один запрос на список текстов."""
    return _call_api([_validate(t) for t in texts])
