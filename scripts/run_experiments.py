from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from src.lab01 import (
    DEFAULT_DB,
    Conversation,
    ProductQuery,
    basic_chain,
    build_search_tool,
    create_llm,
    handle_tool_calls,
    init_database,
    search_pipeline,
    structured_chain,
)


OUT = Path("results/experiment_results.json")
MANDATORY = [
    "Найди ноутбуки дешевле 70 000 рублей",
    "Покажи самые дорогие смартфоны",
    "Хочу посмотреть планшеты",
    "Покажи ноутбуки до 50 тысяч",
    "Найди холодильники",
]
AMBIGUOUS = ["Покажи недорогие ноутбуки", "А сначала дорогие?"]
TEN_QUERIES = MANDATORY + AMBIGUOUS + [
    "Найди планшеты до 30 тысяч",
    "Покажи смартфоны до 20 тысяч",
    "Найди ноутбуки до 10 тысяч",
]


def serialise(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return serialise(asdict(value))
    if isinstance(value, ProductQuery):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "content"):
        return {
            "content": value.content,
            "tool_calls": getattr(value, "tool_calls", []),
            "usage_metadata": getattr(value, "usage_metadata", None),
        }
    if isinstance(value, list):
        return [serialise(item) for item in value]
    if isinstance(value, dict):
        return {key: serialise(item) for key, item in value.items()}
    return value


def timed(call):
    started = time.perf_counter()
    value = call()
    return value, round(time.perf_counter() - started, 3)


def capture(label: str, call):
    try:
        value, seconds = timed(call)
        return {"label": label, "seconds": seconds, "result": serialise(value)}
    except Exception as error:  # сохраняем ошибку для отчёта, не прерывая остальные проверки
        return {"label": label, "error": f"{type(error).__name__}: {error}"}


def raw_structured_response(llm: Any, query: str) -> dict[str, Any]:
    """Сохраняет content и usage до парсинга для сравнения thinking-режимов."""
    parser = PydanticOutputParser(pydantic_object=ProductQuery)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Преобразуй запрос в параметры каталога.\n{format_instructions}"),
            ("human", "Запрос пользователя: {query}"),
        ]
    )
    messages = prompt.invoke(
        {"format_instructions": parser.get_format_instructions(), "query": query}
    ).messages
    response, seconds = timed(lambda: llm.invoke(messages))
    record: dict[str, Any] = {
        "seconds": seconds,
        "content": response.content,
        "usage_metadata": response.usage_metadata,
    }
    try:
        record["parsed"] = parser.parse(response.content).model_dump()
        record["parser_ok"] = True
    except Exception as error:
        record["parser_ok"] = False
        record["parser_error"] = f"{type(error).__name__}: {error}"
    return record


def raw_conversation_response(
    llm: Any, values: dict[str, Any], use_history: bool
) -> tuple[ProductQuery, int | None]:
    """Возвращает параметры и фактическое число входных токенов ответа модели."""
    parser = PydanticOutputParser(pydantic_object=ProductQuery)
    context = "История: {history}" if use_history else "Предыдущие параметры: {previous_query}"
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Обнови ProductQuery по новой реплике. " + context + "\n{format_instructions}"),
            ("human", "Новая реплика: {query}"),
        ]
    )
    messages = prompt.partial(format_instructions=parser.get_format_instructions()).invoke(values).messages
    response = llm.invoke(messages)
    return parser.parse(response.content), (response.usage_metadata or {}).get("input_tokens")


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    init_database(DEFAULT_DB)
    llm_off = create_llm(enable_thinking=False)
    llm_on = create_llm(enable_thinking=True)
    structured_off, _ = structured_chain(llm_off)
    data: dict[str, Any] = {"model": "qwen3.8-27b"}

    direct = llm_off.invoke("Определи категорию: хочу планшет")
    data["part1"] = {
        "direct_type": type(direct).__name__,
        "direct": serialise(direct),
        "chain_type": "str",
        "batch": basic_chain(llm_off).batch(
            [
                {"query": "Нужен ноутбук", "language": "русском"},
                {"query": "Ищу смартфон", "language": "русском"},
                {"query": "Хочу планшет", "language": "русском"},
            ]
        ),
    }
    data["part2_tests"] = [
        capture(query, lambda query=query: structured_off.invoke({"query": query}))
        for query in MANDATORY + AMBIGUOUS
    ]
    data["thinking_comparison"] = []
    for query in AMBIGUOUS:
        data["thinking_comparison"].append(
            {
                "query": query,
                "thinking_off": capture(query, lambda query=query: raw_structured_response(llm_off, query)),
                "thinking_on": capture(query, lambda query=query: raw_structured_response(llm_on, query)),
            }
        )

    pipeline = search_pipeline(llm_off, DEFAULT_DB)
    data["part3_pipeline"] = [
        capture(query, lambda query=query: pipeline.invoke({"query": query})) for query in TEN_QUERIES
    ]
    data["without_database"] = [
        capture(
            query,
            lambda query=query: llm_off.invoke(
                f"Какие товары есть в каталоге? Ответь на вопрос: {query}"
            ),
        )
        for query in MANDATORY[:3]
    ]

    search_tool = build_search_tool(DEFAULT_DB)
    tool_llm = llm_off.bind_tools([search_tool])
    tool_results = []
    for query in MANDATORY:
        response = tool_llm.invoke(query)
        tool_results.append(
            {
                "query": query,
                "model_answer": serialise(response),
                "tool_outcome": serialise(handle_tool_calls(response, DEFAULT_DB)),
            }
        )
    data["part3_tools"] = tool_results
    data["part3_no_tool_call"] = serialise(
        handle_tool_calls(AIMessage(content="Инструмент не требуется"), DEFAULT_DB)
    )

    dialogues = [
        ["Покажи ноутбуки до 80 тысяч", "А сначала дорогие?", "А что есть среди смартфонов?"],
        ["Хочу планшеты", "Только до 35 тысяч", "Теперь самые дорогие"],
    ]
    comparison = []
    for dialogue in dialogues:
        state, history = Conversation(), Conversation()
        steps = []
        for phrase in dialogue:
            state_result, state_tokens = raw_conversation_response(
                llm_off, state.state_input(phrase), use_history=False
            )
            history_result, history_tokens = raw_conversation_response(
                llm_off, history.history_input(phrase), use_history=True
            )
            state.remember(phrase, state_result)
            history.remember(phrase, history_result)
            steps.append(
                {
                    "phrase": phrase,
                    "state_parameters": state_result.model_dump(),
                    "history_parameters": history_result.model_dump(),
                    "state_input_tokens": state_tokens,
                    "history_input_tokens": history_tokens,
                }
            )
        comparison.append(steps)
    data["part5_comparison"] = comparison

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Результаты сохранены: {OUT}")


if __name__ == "__main__":
    main()
