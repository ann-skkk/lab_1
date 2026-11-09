import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


BASE_URL = "http://10.32.2.11:35000/v1"
MODEL = "qwen3.8-27b"


def main() -> None:
    load_dotenv()
    token = os.getenv("LLM_API_TOKEN") or os.getenv("OPENAI_API_KEY")
    if not token:
        sys.exit(
            "Токен не найден. В PowerShell задайте: "
            '$env:LLM_API_TOKEN = "ваш_токен"'
        )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Привет! Ответь ровно одной фразой: "
                    "тестовый запрос успешно получен."
                ),
            }
        ],
        "max_tokens": 64,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    request = Request(
        url=f"{BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)

        print("HTTP status:", response.status)
        print("Ответ модели:", result["choices"][0]["message"]["content"])
        print("Использование токенов:", result.get("usage"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        print(f"HTTP error {error.code}: {body}")
    except URLError as error:
        print(f"Network error: {error.reason}")
    except TimeoutError:
        print("Превышено время ожидания ответа модели.")
    except (KeyError, IndexError, json.JSONDecodeError) as error:
        print(f"Неожиданный формат ответа: {error}")


if __name__ == "__main__":
    main()
