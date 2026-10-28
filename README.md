# Векторная память на Pinecone + Telegram-бот

Семантический поиск по 24 интересным фактам. Эмбеддинги создаются через OpenAI-совместимый Embedding API (Proxy API), векторы хранятся в Pinecone, а Telegram-бот отвечает на вопросы ближайшими по смыслу фактами.

## Технологии
- Python 3.13
- Pinecone (serverless-индекс, метрика cosine)
- `text-embedding-3-small` (1536 измерений) через Proxy API
- python-telegram-bot

## Структура
| Файл | Назначение |
|---|---|
| `config.py` | чтение настроек из `.env` |
| `embeddings.py` | `get_embedding(text)` с обработкой ошибок |
| `pinecone_store.py` | подключение, создание индекса, проверка готовности |
| `data.py` | 24 уникальных факта |
| `test_embedding.py` | проверка одного эмбеддинга |
| `upload_data.py` | пакетная загрузка в Pinecone |
| `search.py` | `semantic_search()` и демо-запросы |
| `bot.py` | Telegram-бот |

## Установка
```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Заполните `.env` (секреты вводите только в этот файл, он в `.gitignore`).

## Pinecone: индекс
1. Зарегистрируйтесь на pinecone.io, создайте проект и API-ключ (API Keys → Create).
2. Индекс создаётся автоматически при `upload_data.py`. Параметры: **serverless**, **dimension = 1536** (как у модели), **metric = cosine**, облако `aws`, регион `us-east-1`.
3. Для `text-embedding-3-large` поставьте `EMBEDDING_DIMENSION=3072` и другое `PINECONE_INDEX_NAME`: у существующего индекса размерность менять нельзя.

## Переменные окружения
| Переменная | Смысл |
|---|---|
| `PINECONE_API_KEY` | ключ Pinecone |
| `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`, `PINECONE_NAMESPACE` | параметры индекса |
| `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL` | ключ и URL Proxy API (OpenAI-совместимый, обычно оканчивается на `/v1`) |
| `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION` | модель и размерность вектора |
| `TELEGRAM_BOT_TOKEN` | токен от @BotFather |
| `SEARCH_TOP_K` | сколько результатов возвращать (по умолчанию 3) |

## Запуск (по порядку)
```bash
python test_embedding.py   # 1. эмбеддинг создаётся?
python upload_data.py      # 2. создать индекс и загрузить 24 факта
python search.py           # 3. демо-запросы (или: python search.py "ваш запрос")
python bot.py              # 4. запустить бота
```

## Примеры запросов
Ожидаемые результаты — прогноз по смыслу; реальные score зависят от модели, сверяйте при запуске.

| Запрос | Ожидаемые ID | Почему |
|---|---|---|
| Что можно хранить тысячелетиями и не бояться порчи? | `fact-001` | мёд из гробниц; слов «мёд» в запросе нет |
| Необычные явления на других планетах и звёздах | `fact-003`, `fact-010` | Венера и нейтронная звезда |
| Как металлические конструкции реагируют на погоду? | `fact-004`, возможно `fact-014` | башня расширяется от жары; молния — тоже погода |
| Выдающиеся учёные и их уникальные достижения | `fact-020` | двойной Нобель Кюри |
| Истоки компьютерной эпохи | `fact-015`, `fact-023` | первый «баг» и первое сообщение ARPANET |

## Чек-лист готовности
- [ ] `upload_data.py` вывел «Индекс ... готов»
- [ ] `test_embedding.py` печатает размерность 1536 и «OK»
- [ ] в Pinecone Console индекс `interesting-facts`, ≥ 20 записей
- [ ] `search.py` возвращает ожидаемые ID из таблицы выше
- [ ] бот отвечает на `/start` и на смысловой запрос
- [ ] `.env` не попал в репозиторий (`git status`)

## Скриншоты для демонстрации
1. Pinecone Console: индекс (dimension, metric) и число записей.
2. Вывод `test_embedding.py`.
3. Вывод `upload_data.py`.
4. Вывод `search.py` с демо-запросами.
5. Диалог с ботом: 2–3 запроса и ответы.

## Типичные ошибки
| Симптом | Причина / решение |
|---|---|
| `Неверный API-ключ` | проверьте `EMBEDDING_API_KEY` и что ключ подходит к `EMBEDDING_BASE_URL` |
| `Модель или URL не найдены` | в `EMBEDDING_BASE_URL` нужен путь до `/v1`; проверьте имя модели у прокси |
| `unexpected keyword dimensions` / 400 | прокси или модель не поддерживает `dimensions` (например, ada-002): уберите параметр в `embeddings.py` |
| Размерность индекса не совпадает | другая модель, чем при создании индекса: новый `PINECONE_INDEX_NAME` |
| Поиск пуст сразу после загрузки | Pinecone обновляется с задержкой, подождите несколько секунд |
| Бот молчит | неверный токен или запущен второй экземпляр бота |

## Сдача zip-архивом
Включить: `*.py`, `requirements.txt`, `.env.example`, `.gitignore`, `README.md`, скриншоты.
Исключить: `.env` (секреты!), `.venv/`, `__pycache__/`.
