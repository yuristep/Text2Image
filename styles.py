import os
import json
import argparse
import requests
from dotenv import load_dotenv


def fetch_styles(api_key: str, api_secret: str, timeout: int = 30) -> list:
    """Запрашивает список стилей из FusionBrain API и возвращает список объектов.

    Args:
        api_key: Значение переменной api_key.
        api_secret: Значение переменной secret_key.
        timeout: Таймаут HTTP-запроса в секундах.

    Returns:
        Список словарей со стилями.

    Raises:
        RuntimeError: Если ответ не 200 или не JSON.
    """
    url = "https://api-key.fusionbrain.ai/key/api/v1/styles"
    headers = {
        "X-Key": f"Key {api_key}",
        "X-Secret": f"Secret {api_secret}",
        "Accept": "application/json",
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"API /styles вернул {resp.status_code}: {resp.text}")
    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError("API /styles вернул не-JSON")
    return resp.json()


def fetch_styles_from_cdn(timeout: int = 30) -> list:
    """Запрашивает список стилей из публичного CDN без авторизации.

    Используется как запасной источник, если закрытый API недоступен/недосягаем.
    Документация упоминает публичный список: cdn.fusionbrain.ai/static/styles/api
    """
    url = "https://cdn.fusionbrain.ai/static/styles/api"
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"CDN /static/styles/api вернул {resp.status_code}: {resp.text}")
    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError("CDN /static/styles/api вернул не-JSON")
    return resp.json()

def fetch_style_presets(timeout: int = 30) -> list:
    """Получает пресеты стилей (name/title/titleEn/image) из CDN.

    Если CDN недоступен или изменился путь, возвращает известный список по умолчанию.
    """
    candidates = [
        "https://cdn.fusionbrain.ai/static/styles/presets",
        "https://cdn.fusionbrain.ai/static/styles",  # запасной вариант
    ]
    for url in candidates:
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200 and "application/json" in resp.headers.get("Content-Type", ""):
                data = resp.json()
                # ожидаемый формат — массив объектов с полями name/title/titleEn/image
                if isinstance(data, list) and data and isinstance(data[0], dict) and "name" in data[0]:
                    return data
        except Exception:
            pass
    # Фолбэк — список по состоянию на август 2024
    return [
        {
            "name": "KANDINSKY",
            "title": "Кандинский",
            "titleEn": "Kandinsky",
            "image": "https://cdn.fusionbrain.ai/static/download/img-style-kandinsky.png",
        },
        {
            "name": "UHD",
            "title": "Детальное фото",
            "titleEn": "Detailed photo",
            "image": "https://cdn.fusionbrain.ai/static/download/img-style-detail-photo.png",
        },
        {
            "name": "ANIME",
            "title": "Аниме",
            "titleEn": "Anime",
            "image": "https://cdn.fusionbrain.ai/static/download/img-style-anime.png",
        },
        {
            "name": "DEFAULT",
            "title": "Свой стиль",
            "titleEn": "No style",
            "image": "https://cdn.fusionbrain.ai/static/download/img-style-personal.png",
        },
    ]

def main():
    parser = argparse.ArgumentParser(description="Получение списка стилей FusionBrain")
    parser.add_argument("-n", "--top", type=int, default=5, help="Сколько первых стилей показать")
    parser.add_argument("-o", "--output", type=str, default=None, help="Сохранить JSON в файл")
    parser.add_argument("--source", choices=["auto", "api", "cdn"], default="auto", help="Источник стилей: закрытый API или публичный CDN")
    parser.add_argument("--show-raw-error", action="store_true", help="Показывать сырое тело ошибки")
    parser.add_argument("--presets", action="store_true", help="Получить пресеты стилей (name/title/titleEn/image)")
    args = parser.parse_args()

    # Загружаем ключи из .env или examples/.env
    load_dotenv()
    api_key = os.getenv("api_key")
    api_secret = os.getenv("secret_key")
    if not api_key or not api_secret:
        load_dotenv("examples/.env")
        api_key = os.getenv("api_key")
        api_secret = os.getenv("secret_key")
    if not api_key or not api_secret:
        print("Не найдены api_key/secret_key в .env или examples/.env")
        raise SystemExit(1)

    # Режим пресетов — отдельный путь
    if args.presets:
        presets = fetch_style_presets()
        print(f"Количество пресетов: {len(presets)}")
        for item in presets[: max(0, args.top)]:
            print(f"name: {item.get('name')}, title: {item.get('title')}, titleEn: {item.get('titleEn')}")
        if args.output:
            try:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(presets, f, ensure_ascii=False, indent=2)
                print(f"Пресеты сохранены в {args.output}")
            except Exception as e:
                print(f"Не удалось сохранить файл: {e}")
        return

    styles = []
    if args.source in ("auto", "api"):
        try:
            styles = fetch_styles(api_key, api_secret)
        except Exception as e:
            msg = str(e)
            print(f"Предупреждение: не удалось получить стили по закрытому API. {msg}")
            if args.show_raw_error:
                print("(raw)", msg)
            if args.source == "api":
                print("Подсказка: проверьте доступность эндпоинта /key/api/v1/styles для вашего аккаунта и корректность ключей.")
                return
    if not styles and args.source in ("auto", "cdn"):
        try:
            styles = fetch_styles_from_cdn()
        except Exception as e2:
            msg = str(e2)
            print(f"Предупреждение: не удалось получить стили из публичного CDN. {msg}")
            if args.show_raw_error:
                print("(raw)", msg)
            print("Подсказка: endpoint CDN мог измениться. Сверьтесь с документацией.")
            return

    print(f"Количество стилей: {len(styles)}")
    for style in styles[: max(0, args.top)]:
        style_id = style.get("id")
        name = style.get("name")
        print(f"ID: {style_id}, Name: {name}")

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(styles, f, ensure_ascii=False, indent=2)
            print(f"Стили сохранены в {args.output}")
        except Exception as e:
            print(f"Не удалось сохранить файл: {e}")


if __name__ == "__main__":
    main()


