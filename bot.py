"""Telegram-бот: принимает вопрос, ищет по смыслу в Pinecone, отвечает ближайшими фактами."""
import asyncio
import html
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import config
from embeddings import EmbeddingError
from search import semantic_search

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
# httpx пишет в INFO полный URL запроса вместе с токеном бота — скрываем.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

HELP_TEXT = (
    "Я ищу интересные факты <b>по смыслу</b>, а не по словам.\n"
    "Напишите тему или вопрос, например:\n"
    "• <i>что не портится тысячелетиями?</i>\n"
    "• <i>необычное про планеты</i>"
)


def format_results(query: str, results: list[dict]) -> str:
    if not results:
        return "Ничего не нашлось. Попробуйте переформулировать запрос."
    lines = [f"🔎 Результаты по запросу «{html.escape(query)}»:\n"]
    for rank, item in enumerate(results, start=1):
        percent = max(0.0, item["score"]) * 100
        lines.append(
            f"<b>{rank}.</b> {html.escape(item['text'])}\n"
            f"   📊 релевантность: {percent:.0f}% · тема: {html.escape(item['topic'])} · "
            f"<code>{html.escape(item['id'])}</code>\n"
        )
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(f"Привет! 👋\n\n{HELP_TEXT}", parse_mode=ParseMode.HTML)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.HTML)


async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = (update.message.text or "").strip()
    if not query:
        await update.message.reply_text("Напишите текстовый запрос.")
        return
    await update.message.chat.send_action("typing")
    try:
        # Поиск синхронный — уводим в поток, чтобы не блокировать бота.
        results = await asyncio.to_thread(semantic_search, query)
    except EmbeddingError as exc:
        await update.message.reply_text(f"⚠️ Не удалось обработать запрос: {exc}")
        return
    except Exception:
        log.exception("Ошибка поиска")
        await update.message.reply_text("⚠️ Поиск временно недоступен (проверьте Pinecone). Попробуйте позже.")
        return
    await update.message.reply_text(format_results(query, results), parse_mode=ParseMode.HTML)


def main() -> None:
    app = Application.builder().token(config.telegram_bot_token()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    log.info("Бот запущен. Остановка — Ctrl+C.")
    app.run_polling()


if __name__ == "__main__":
    main()
