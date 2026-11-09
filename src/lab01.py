from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


BASE_URL = "http://10.32.2.11:35000/v1"
MODEL_NAME = "qwen3.8-27b"
DEFAULT_DB = Path("data/products.db")


class ProductQuery(BaseModel):
    """Проверенные параметры поиска; единственный вход для SQL-поиска."""

    category: Literal["laptop", "smartphone", "tablet", "other"] = Field(
        description="Категория товара; other для неподдерживаемой категории."
    )
    max_price: int | None = Field(
        default=None,
        ge=0,
        description="Максимальная цена в рублях или null без ограничения.",
    )
    sort_by: Literal["price_asc", "price_desc"] = Field(
        default="price_asc", description="Направление сортировки по цене."
    )


@dataclass
class Product:
    id: int
    name: str
    category: str
    price: int


@dataclass
class SearchResult:
    status: Literal["ok", "unsupported_category"]
    products: list[Product]
    message: str


@dataclass
class LogEvent:
    stage: str
    message: str
    attempt: int | None = None


@dataclass
class PipelineLog:
    events: list[LogEvent] = field(default_factory=list)

    def add(self, stage: str, message: str, attempt: int | None = None) -> None:
        self.events.append(LogEvent(stage, message, attempt))

    def as_json(self) -> str:
        return json.dumps([asdict(item) for item in self.events], ensure_ascii=False, indent=2)


@dataclass
class ToolCallOutcome:
    status: Literal["ok", "no_tool_call", "invalid_arguments"]
    results: list[SearchResult]
    message: str


CATALOG: list[tuple[str, str, int]] = [
    ("Acer Aspire 5", "laptop", 54_990),
    ("Lenovo IdeaPad Slim 3", "laptop", 64_990),
    ("ASUS VivoBook 16", "laptop", 79_990),
    ("Apple MacBook Air M2", "laptop", 109_990),
    ("Xiaomi Redmi Note 14", "smartphone", 24_990),
    ("Samsung Galaxy A56", "smartphone", 39_990),
    ("Google Pixel 9", "smartphone", 74_990),
    ("Apple iPhone 16", "smartphone", 89_990),
    ("Samsung Galaxy Tab A9+", "tablet", 22_990),
    ("Xiaomi Pad 7", "tablet", 34_990),
    ("Huawei MatePad 11.5", "tablet", 42_990),
    ("Apple iPad Air", "tablet", 79_990),
    ("Logitech MX Keys", "accessory", 12_990),
    ("Sony WH-1000XM5", "accessory", 31_990),
    ("Kingston XS1000 SSD", "accessory", 10_990),
]


def init_database(db_path: Path | str = DEFAULT_DB) -> None:
    """Создаёт воспроизводимый каталог из 15 товаров."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("DROP TABLE IF EXISTS products")
        connection.execute(
            """CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price INTEGER NOT NULL CHECK(price >= 0)
            )"""
        )
        connection.executemany(
            "INSERT INTO products(name, category, price) VALUES (?, ?, ?)", CATALOG
        )
        connection.commit()
    finally:
        connection.close()


def search_products(query: ProductQuery, db_path: Path | str = DEFAULT_DB) -> SearchResult:
    """Выполняет безопасный параметризованный поиск без участия LLM в SQL."""
    if query.category == "other":
        return SearchResult(
            status="unsupported_category",
            products=[],
            message="Категория не поддерживается каталогом.",
        )

    order = {"price_asc": "ASC", "price_desc": "DESC"}[query.sort_by]
    sql = "SELECT id, name, category, price FROM products WHERE category = ?"
    values: list[Any] = [query.category]
    if query.max_price is not None:
        sql += " AND price <= ?"
        values.append(query.max_price)
    sql += f" ORDER BY price {order}, id ASC"

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(sql, values).fetchall()
    finally:
        connection.close()
    products = [Product(*row) for row in rows]
    return SearchResult("ok", products, f"Найдено товаров: {len(products)}.")


def create_llm(enable_thinking: bool = False) -> ChatOpenAI:
    """Создаёт OpenAI-совместимый клиент. Сетевой вызов здесь не выполняется."""
    load_dotenv()
    token = os.getenv("LLM_API_TOKEN")
    if not token:
        raise RuntimeError("Не найден LLM_API_TOKEN в .env или переменных окружения.")
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=token,
        base_url=BASE_URL,
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
    )


def basic_chain(llm: Any):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Ты определяешь категорию товара. Отвечай на {language} кратко."),
            ("human", "Запрос пользователя: {query}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def structured_chain(llm: Any):
    parser = PydanticOutputParser(pydantic_object=ProductQuery)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Преобразуй запрос в параметры каталога. "
                "Неизвестную категорию обозначай other.\n{format_instructions}",
            ),
            ("human", "Запрос пользователя: {query}"),
        ]
    )
    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    return chain, parser


def retrying_structured_chain(llm: Any):
    """Повторяет исключительно LLM+парсер, максимум три попытки."""
    chain, _ = structured_chain(llm)
    return chain.with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )


def invoke_with_logged_retry(chain: Any, payload: dict[str, Any], log: PipelineLog) -> ProductQuery:
    """Повторяет цепочку разбора и записывает каждую попытку в журнал."""
    attempt = 0

    def parse_once(values: dict[str, Any]) -> ProductQuery:
        nonlocal attempt
        attempt += 1
        log.add("llm", "Начало вызова LLM", attempt)
        try:
            result = chain.invoke(values)
        except OutputParserException as error:
            log.add("parse_error", f"{type(error).__name__}: {error}", attempt)
            raise
        log.add("validation", result.model_dump_json(), attempt)
        return result

    return RunnableLambda(parse_once).with_retry(
        retry_if_exception_type=(OutputParserException,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    ).invoke(payload)


def run_logged_search(
    llm: Any,
    payload: dict[str, Any],
    db_path: Path | str = DEFAULT_DB,
    log: PipelineLog | None = None,
) -> tuple[SearchResult, PipelineLog]:
    """Выполняет поиск только после успешного разбора всех retry-попыток."""
    current_log = log or PipelineLog()
    chain, _ = structured_chain(llm)
    query = invoke_with_logged_retry(chain, payload, current_log)
    result = search_products(query, db_path)
    current_log.add("search", result.message)
    return result, current_log


def search_pipeline(llm: Any, db_path: Path | str = DEFAULT_DB):
    """LCEL: структурирование -> обычная Python-функция поиска."""
    parser_chain = retrying_structured_chain(llm)
    return parser_chain | RunnableLambda(lambda query: search_products(query, db_path))


def build_search_tool(db_path: Path | str = DEFAULT_DB):
    @tool("search_products", args_schema=ProductQuery)
    def search_products_tool(
        category: Literal["laptop", "smartphone", "tablet", "other"],
        max_price: int | None = None,
        sort_by: Literal["price_asc", "price_desc"] = "price_asc",
    ) -> str:
        """Ищет товары каталога по проверенным параметрам категории, цены и сортировки."""
        query = ProductQuery(category=category, max_price=max_price, sort_by=sort_by)
        result = search_products(query, db_path)
        return json.dumps(
            {"status": result.status, "message": result.message, "products": [asdict(p) for p in result.products]},
            ensure_ascii=False,
        )

    return search_products_tool


def extract_valid_tool_calls(message: Any, db_path: Path | str = DEFAULT_DB) -> list[SearchResult]:
    """Возвращает результаты корректных вызовов search_products."""
    return handle_tool_calls(message, db_path).results


def handle_tool_calls(message: Any, db_path: Path | str = DEFAULT_DB) -> ToolCallOutcome:
    """Проверяет tool_calls и сообщает, если инструмент не выбран или аргументы неверны."""
    calls = [call for call in getattr(message, "tool_calls", []) if call.get("name") == "search_products"]
    if not calls:
        return ToolCallOutcome("no_tool_call", [], "Модель не вызвала инструмент поиска.")

    results: list[SearchResult] = []
    for call in calls:
        try:
            query = ProductQuery.model_validate(call.get("args", {}))
        except Exception as error:
            logging.warning("Некорректные аргументы tool call: %s", error)
            return ToolCallOutcome("invalid_arguments", [], f"Аргументы инструмента не прошли проверку: {error}")
        results.append(search_products(query, db_path))
    return ToolCallOutcome("ok", results, f"Выполнено вызовов поиска: {len(results)}.")


class Conversation:
    """Состояние диалога вне модели: последние параметры и история реплик."""

    def __init__(self) -> None:
        self.last_query: ProductQuery | None = None
        self.history: list[dict[str, str]] = []

    def state_input(self, user_text: str) -> dict[str, Any]:
        return {
            "previous_query": self.last_query.model_dump_json() if self.last_query else "null",
            "query": user_text,
        }

    def history_input(self, user_text: str) -> dict[str, Any]:
        return {"history": json.dumps(self.history, ensure_ascii=False), "query": user_text}

    def remember(self, user_text: str, query: ProductQuery) -> None:
        self.last_query = query
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": query.model_dump_json()})


def conversation_chain(llm: Any, use_history: bool = False):
    parser = PydanticOutputParser(pydantic_object=ProductQuery)
    context = "История: {history}" if use_history else "Предыдущие параметры: {previous_query}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Обнови ProductQuery по новой реплике. " + context + "\n{format_instructions}"),
            ("human", "Новая реплика: {query}"),
        ]
    )
    return prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser


def run_repl(db_path: Path | str = DEFAULT_DB) -> None:
    """Интерактивный интерфейс каталога."""
    llm = create_llm()
    chain = conversation_chain(llm)
    state = Conversation()
    print("Каталог готов. Введите запрос или exit.")
    while True:
        text = input("> ").strip()
        if text.lower() in {"exit", "quit", "выход"}:
            break
        if not text:
            continue
        log = PipelineLog()
        try:
            parsed = invoke_with_logged_retry(chain, state.state_input(text), log)
            if parsed.max_price is None:
                answer = input("Искать без ограничения цены? [y/N] ").strip().lower()
                if answer not in {"y", "yes", "д", "да"}:
                    print("Поиск отменён.")
                    continue
            result = search_products(parsed, db_path)
            log.add("search", result.message)
            state.remember(text, parsed)
            print(parsed.model_dump_json(indent=2))
            print(result.message)
            for product in result.products:
                print(f"- {product.name}: {product.price} руб.")
        except Exception as error:
            log.add("error", str(error))
            print(f"Ошибка обработки запроса: {error}")
        finally:
            print(log.as_json())


def save_demo(output_path: Path | str = "results/demo.json") -> None:
    """Сохраняет результат одного демонстрационного запроса."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    llm = create_llm()
    chain, _ = structured_chain(llm)
    started = time.perf_counter()
    result = chain.invoke({"query": "Найди ноутбуки дешевле 70 000 рублей"})
    output.write_text(
        json.dumps(
            {"query": "Найди ноутбуки дешевле 70 000 рублей", "result": result.model_dump(), "seconds": round(time.perf_counter() - started, 3)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Результат сохранён в {output}")


if __name__ == "__main__":
    init_database()
    run_repl()
