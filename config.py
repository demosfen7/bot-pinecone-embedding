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
