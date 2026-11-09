from src.lab01 import DEFAULT_DB, init_database


if __name__ == "__main__":
    init_database(DEFAULT_DB)
    print(f"Каталог создан: {DEFAULT_DB.resolve()}")
