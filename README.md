# Пайплайн поиска товаров на LangChain

Приложение преобразует запрос на естественном языке в параметры поиска, проверяет их через Pydantic и ищет товары в SQLite. Языковая модель не составляет SQL-запросы и не обращается к базе напрямую.

## Структура

| Путь | Содержание |
|---|---|
| `src/lab01.py` | Схемы, LCEL-цепочки, поиск, инструменты, retry и консольный интерфейс. |
| `data/products.db` | Каталог из 15 товаров. |
| `scripts/` | Инициализация базы и запуск экспериментов. |
| `tests/` | Проверки схемы, SQLite-поиска, tool и retry. |
| `notebooks/lab01.ipynb` | Последовательная демонстрация частей работы. |
| `results/experiment_results.json` | Результаты экспериментов с моделью. |
| `report/lab01_report.md` | Отчёт. |

## Установка

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m scripts.init_catalog
```

Настройки модели находятся в `.env`:

```dotenv
LLM_API_TOKEN=ваш_токен
```

## Запуск

```powershell
python -m scripts.test_model
python -m scripts.run_experiments
python -m src.lab01
```

Последняя команда открывает консольный диалог. Для завершения введите `exit`.

## Тесты

```powershell
python -m unittest discover -s tests -v
```
