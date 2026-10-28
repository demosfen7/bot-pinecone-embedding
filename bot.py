"""Telegram-бот: ведёт беседу, опираясь на векторную базу, и помнит пользователя."""
import asyncio
import html
import logging

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import chat
import config
import memory
from embeddings import ApiError
from search import semantic_search

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
# httpx пишет в INFO полный URL запроса вместе с токеном бота — скрываем.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

TELEGRAM_LIMIT = 4096

HELP_TEXT = (
    "Я собеседник с памятью: болтаю, знаю интересные факты и помню, что вы мне рассказали.\n\n"
    "Попробуйте:\n"
    "• <i>Что можно хранить тысячелетиями?</i>\n"
    "• <i>Запомни, что мне нравится жёлтый Beetle</i>\n"
    "• <i>Что ты обо мне помнишь?</i>\n"
    "• <i>Забудь про Beetle</i>\n\n"
    "Команды:\n"
    "/memory — всё, что я о вас помню\n"
    "/search запрос — сырой поиск по базе фактов\n"
    "/reset — начать разговор заново (память остаётся)"
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


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.chat_data["history"] = []
    await update.message.reply_text("Начали разговор заново. Память о вас осталась.")


async def show_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        items = await asyncio.to_thread(memory.list_memories, update.effective_user.id)
    except Exception:
        log.exception("Ошибка чтения памяти")
        await update.message.reply_text("⚠️ Не удалось прочитать память (проверьте Pinecone).")
        return
    if not items:
        await update.message.reply_text("Пока я о вас ничего не помню. Скажите «запомни, что ...».")
        return
    lines = ["🧠 Что я помню:\n"] + [f"• {html.escape(i['text'])}\n  <code>{i['id']}</code>" for i in items]
    await update.message.reply_text("\n".join(lines)[:TELEGRAM_LIMIT], parse_mode=ParseMode.HTML)


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text("Напишите запрос после команды: /search что не портится")
        return
    try:
        results = await asyncio.to_thread(semantic_search, query)
    except ApiError as exc:
        await update.message.reply_text(f"⚠️ {exc}")
        return
    except Exception:
        log.exception("Ошибка поиска")
        await update.message.reply_text("⚠️ Поиск временно недоступен (проверьте Pinecone).")
        return
    await update.message.reply_text(format_results(query, results), parse_mode=ParseMode.HTML)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = (update.message.text or "").strip()
    if not user_text:
        return
    history = context.chat_data.setdefault("history", [])
    await update.message.chat.send_action("typing")
    try:
        # Работа с API синхронная — уводим в поток, чтобы не блокировать бота.
        answer = await asyncio.to_thread(chat.reply, update.effective_user.id, history, user_text)
    except ApiError as exc:
        await update.message.reply_text(f"⚠️ {exc}", do_quote=True)
        return
    except Exception:
        log.exception("Ошибка обработки сообщения")
        await update.message.reply_text(
            "⚠️ Что-то пошло не так (проверьте Pinecone и ключи). Попробуйте позже.", do_quote=True
        )
        return

    history.extend([{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}])
    del history[:-config.HISTORY_LIMIT]
    # do_quote=True — ответ приходит цитатой на сообщение пользователя.
    await update.message.reply_text(answer[:TELEGRAM_LIMIT], do_quote=True)


def main() -> None:
    app = Application.builder().token(config.telegram_bot_token()).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("memory", show_memory))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    log.info("Бот запущен. Остановка — Ctrl+C.")
    app.run_polling()


if __name__ == "__main__":
    main()
