import tempfile
import unittest
from pathlib import Path

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.lab01 import (
    PipelineLog,
    ProductQuery,
    build_search_tool,
    handle_tool_calls,
    init_database,
    run_logged_search,
    retrying_structured_chain,
    search_products,
    structured_chain,
)


VALID_JSON = '{"category":"laptop","max_price":70000,"sort_by":"price_asc"}'


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tempdir.name) / "products.db"
        init_database(self.db)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_schema_rejects_negative_price(self) -> None:
        with self.assertRaises(ValidationError):
            ProductQuery(category="laptop", max_price=-1)

    def test_search_uses_price_filter_and_sort(self) -> None:
        result = search_products(
            ProductQuery(category="laptop", max_price=70_000, sort_by="price_desc"), self.db
        )
        self.assertEqual([p.name for p in result.products], ["Lenovo IdeaPad Slim 3", "Acer Aspire 5"])

    def test_other_category_does_not_search_everything(self) -> None:
        result = search_products(ProductQuery(category="other"), self.db)
        self.assertEqual(result.status, "unsupported_category")
        self.assertEqual(result.products, [])

    def test_empty_result(self) -> None:
        result = search_products(ProductQuery(category="tablet", max_price=10_000), self.db)
        self.assertEqual(result.products, [])

    def test_tool_validates_arguments(self) -> None:
        tool = build_search_tool(self.db)
        output = tool.invoke({"category": "smartphone", "max_price": 40_000, "sort_by": "price_asc"})
        self.assertIn("Xiaomi Redmi Note 14", output)

    def test_response_without_tool_call_is_reported(self) -> None:
        outcome = handle_tool_calls(AIMessage(content="Обычный текст"), self.db)
        self.assertEqual(outcome.status, "no_tool_call")
        self.assertEqual(outcome.results, [])


class RetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.db = Path(self.tempdir.name) / "products.db"
        init_database(self.db)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_without_retry_fails_on_first_invalid_response(self) -> None:
        llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="not JSON"), AIMessage(content=VALID_JSON)]
        )
        chain, _ = structured_chain(llm)
        with self.assertRaises(OutputParserException):
            chain.invoke({"query": "ноутбуки"})

    def test_second_attempt_succeeds(self) -> None:
        llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="not JSON"), AIMessage(content=VALID_JSON)]
        )
        result = retrying_structured_chain(llm).invoke({"query": "ноутбуки"})
        self.assertEqual(result.category, "laptop")

    def test_logged_retry_searches_once_after_successful_parse(self) -> None:
        llm = FakeMessagesListChatModel(
            responses=[AIMessage(content="not JSON"), AIMessage(content=VALID_JSON)]
        )
        result, log = run_logged_search(llm, {"query": "ноутбуки"}, self.db)
        self.assertEqual(len(result.products), 2)
        self.assertEqual(sum(event.stage == "llm" for event in log.events), 2)
        self.assertEqual(sum(event.stage == "parse_error" for event in log.events), 1)
        self.assertEqual(sum(event.stage == "search" for event in log.events), 1)

    def test_three_invalid_attempts_raise(self) -> None:
        llm = FakeMessagesListChatModel(responses=[AIMessage(content="bad")] * 3)
        log = PipelineLog()
        with self.assertRaises(OutputParserException):
            run_logged_search(llm, {"query": "ноутбуки"}, self.db, log)
        self.assertEqual(sum(event.stage == "llm" for event in log.events), 3)
        self.assertEqual(sum(event.stage == "search" for event in log.events), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
