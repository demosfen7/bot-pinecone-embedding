"""Диалог с LLM: ответ строится на найденных фактах и воспоминаниях,
а запись/чтение/удаление памяти модель выполняет сама через вызов инструментов."""
import json
import logging

import config
import memory
from embeddings import api_errors, get_client
from search import semantic_search

log = logging.getLogger("chat")

MAX_TOOL_STEPS = 5  # защита от бесконечной цепочки вызовов инструментов

SYSTEM_PROMPT = """Ты — дружелюбный собеседник в Telegram-боте «Интересные факты» с долговременной памятью.
Отвечай по-русски, живо и по делу, 2–5 предложений, без длинных списков.

Тебе дают два источника контекста:
1. «Факты из базы» — проверенные интересные факты. Если вопрос связан с ними, опирайся на них и не искажай.
2. «Что ты помнишь о пользователе» — личные записи. Учитывай их естественно, не перечисляя без нужды.
Если в базе нет ответа, можешь ответить из общих знаний, но честно скажи, что в базе этого нет.

Память пользователя меняется только через инструменты:
- remember — когда просят «запомни…» или пользователь сообщает устойчивое предпочтение или факт о себе. Сохраняй короткую формулировку от третьего лица («Любит жёлтый Volkswagen Beetle»).
- recall / list_memories — когда спрашивают, что ты помнишь или знаешь о пользователе.
- forget — когда просят «забудь…» или «удали…». Сначала найди нужную запись через recall или list_memories, затем удали её по id. Если записей подходит несколько и неясно, какую удалить, уточни у пользователя.
После действия с памятью коротко подтверди, что именно сделано. Не утверждай, что запомнил или удалил, пока инструмент не вернул успех."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Сохранить воспоминание о пользователе в долговременную память.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Короткая самодостаточная формулировка."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Найти по смыслу воспоминания о пользователе, подходящие под запрос.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memories",
            "description": "Показать все сохранённые воспоминания пользователя с их id.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Удалить воспоминания по их id (id берутся из recall или list_memories).",
            "parameters": {
                "type": "object",
                "properties": {"memory_ids": {"type": "array", "items": {"type": "string"}}},
                "required": ["memory_ids"],
            },
        },
    },
]


def _run_tool(name: str, args: dict, user_id: int) -> str:
    """Выполняет инструмент и возвращает результат в виде JSON-строки для модели."""
    try:
        if name == "remember":
            result = memory.remember(user_id, args["text"])
        elif name == "recall":
            found = memory.recall(user_id, args["query"])
            result = [{"id": r["id"], "text": r["text"], "score": round(r["score"], 3)} for r in found]
        elif name == "list_memories":
            result = memory.list_memories(user_id)
        elif name == "forget":
            result = memory.forget(user_id, list(args["memory_ids"]))
        else:
            result = {"error": f"неизвестный инструмент {name}"}
    except Exception as exc:  # модель сообщит пользователю, что действие не удалось
        log.exception("Ошибка инструмента %s", name)
        result = {"error": str(exc)}
    return json.dumps(result, ensure_ascii=False)


def build_context(query: str, user_id: int) -> str:
    """Подмешиваем к запросу релевантные факты и воспоминания (RAG)."""
    facts = [r for r in semantic_search(query) if r["score"] >= config.MIN_SCORE]
    memories = [r for r in memory.recall(user_id, query) if r["score"] >= config.MIN_SCORE]

    parts = ["Факты из базы:"]
    parts += [f"- {r['text']} (релевантность {r['score']:.2f})" for r in facts] or ["- ничего подходящего"]
    parts.append("\nЧто ты помнишь о пользователе:")
    parts += [f"- {r['text']}" for r in memories] or ["- ничего подходящего"]
    return "\n".join(parts)


def reply(user_id: int, history: list[dict], user_text: str) -> str:
    """Возвращает ответ бота. history — прошлые реплики вида {role, content}."""
    system = f"{SYSTEM_PROMPT}\n\n{build_context(user_text, user_id)}"
    messages = [{"role": "system", "content": system}, *history, {"role": "user", "content": user_text}]

    extra = {"reasoning_effort": config.CHAT_REASONING_EFFORT} if config.CHAT_REASONING_EFFORT else {}
    with api_errors():
        for _ in range(MAX_TOOL_STEPS):
            response = get_client().chat.completions.create(
                model=config.CHAT_MODEL, messages=messages, tools=TOOLS, **extra
            )
            message = response.choices[0].message
            if not message.tool_calls:
                return (message.content or "").strip() or "Не нашёл, что ответить. Спросите иначе?"

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments},
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            for call in message.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = _run_tool(call.function.name, args, user_id)
                log.info("tool %s(%s) -> %s", call.function.name, args, result[:200])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    return "Не удалось завершить действие за разумное число шагов. Попробуйте сформулировать проще."
