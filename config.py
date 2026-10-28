"""Единая точка чтения настроек из переменных окружения (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Не хватает обязательной настройки."""


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your-"):
        raise ConfigError(
            f"Переменная окружения {name} не задана. "
            f"Скопируйте .env.example в .env и заполните её."
        )
    return value


def pinecone_api_key() -> str:
    return _require("PINECONE_API_KEY")


def embedding_api_key() -> str:
    return _require("EMBEDDING_API_KEY")


def telegram_bot_token() -> str:
    return _require("TELEGRAM_BOT_TOKEN")


# Необязательные параметры имеют значения по умолчанию.
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "interesting-facts")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
NAMESPACE = os.getenv("PINECONE_NAMESPACE", "facts")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "").strip() or None
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

SEARCH_TOP_K = int(os.getenv("SEARCH_TOP_K", "3"))
# Ниже этого порога результат считаем нерелевантным и не показываем модели.
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.25"))

# Модель для диалога; OPENAI_MODEL оставлен как запасное имя переменной.
CHAT_MODEL = os.getenv("CHAT_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
# Для рассуждающих моделей вызов инструментов работает только с "none"; для остальных оставьте пустым.
CHAT_REASONING_EFFORT = os.getenv("CHAT_REASONING_EFFORT", "").strip()
# Отдельный namespace для того, что бот запоминает о пользователях.
MEMORY_NAMESPACE = os.getenv("PINECONE_MEMORY_NAMESPACE", "memory")
# Сколько последних реплик диалога держим в контексте.
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "10"))
