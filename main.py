import os
import logging
from dotenv import load_dotenv
import json
import requests
import time
import base64
from PIL import Image
from io import BytesIO
import argparse


class Text2ImageAPI:
    def __init__(self, prompt: str | None, width: int, height: int, style: str, neg_prompt: str,
                 show: bool, save: bool, debug: bool, outfile: str | None = None, prompt_file: str | None = None):
        self.DEBUG = debug
        self.init_loger()
        self.init_env()
        self.init_settings()

        self.pipeline_id = self.get_pipeline()  # получение id пайплайна
        if not self.pipeline_id:
            self.logger.error("Не удалось получить pipeline id")
            return
        if not (self.availability_service(self.pipeline_id)):
            return

        # выбрать промпт: -p приоритетнее, иначе --prompt-file, иначе examples/promt.txt
        effective_prompt = None
        if isinstance(prompt, str) and prompt.strip():
            effective_prompt = self._normalize_and_validate_prompt(prompt.strip(), source='-p')
        else:
            path = prompt_file or "examples/promt.txt"
            effective_prompt, file_negative = self.load_prompt_from_file(path)
            if not neg_prompt and file_negative:
                neg_prompt = file_negative

        uuid = self.generate(effective_prompt, style, neg_prompt, pipeline_id=self.pipeline_id, width=width, height=height)
        file_or_base64 = self.check_generation(uuid)
        # choose output file name
        output_path = outfile
        if not output_path and save:
            output_path = self._generate_result_filename(style, uuid)
        if not output_path:
            output_path = f"image_{int(time.time())}.jpg"
        self.base64_to_image(file_or_base64, show=show, save=save, filename=output_path)

    def init_loger(self):
        self.logger = logging.getLogger(__name__)
        if self.DEBUG:
            self.logger.setLevel(logging.DEBUG)
            os.makedirs('log', exist_ok=True)
            handler = logging.FileHandler(f"log/DEBUG {__name__}.log", mode='w', encoding="UTF-8")
        else:
            self.logger.setLevel(logging.INFO)
            os.makedirs('log', exist_ok=True)
            handler = logging.FileHandler(f"log/{__name__}.log", mode='w', encoding="UTF-8")
        handler.setFormatter(logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(handler)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.debug(f"Инициализация логера")

    def init_env(self):
        self.logger.debug(f"Инициализация виртуального окружения")
        load_dotenv()
        api_key = os.getenv("api_key")
        secret_key = os.getenv("secret_key")
        if not api_key or not secret_key:
            # Fallback to examples/.env if root .env not present
            load_dotenv("examples/.env")
            api_key = os.getenv("api_key")
            secret_key = os.getenv("secret_key")
        if not api_key or not secret_key:
            self.logger.error("Не найдены api_key/secret_key в .env или examples/.env")
            raise SystemExit(1)
        self.api_key = api_key
        self.secret_key = secret_key

    def init_settings(self):
        with open('settings.json') as f:
            settings = json.load(f)
        self.URL = settings['URL']
        self.KEYS = settings['KEYS']
        # HTTP session with default headers and timeouts
        self.TIMEOUT = 30
        self.AUTH_HEADERS = {
            'X-Key': f'Key {self.api_key}',
            'X-Secret': f'Secret {self.secret_key}',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.AUTH_HEADERS)
        self.logger.debug(f"Инициализация настроек скрипта")

    def load_prompt_from_file(self, path: str = "examples/promt.txt") -> tuple[str, str]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content:
                self.logger.error(f"Файл промпта пуст: {path}")
                raise SystemExit(1)
            query_text, negative_text = self._parse_prompt_tags(content)
            query_text = self._normalize_and_validate_prompt(query_text, source=f"file:{path}")
            if negative_text:
                negative_text = " ".join(negative_text.split())
            self.logger.info(f"Промпт загружен из файла: {path}")
            return query_text, (negative_text or "")
        except FileNotFoundError:
            self.logger.error(f"Файл промпта не найден: {path}")
            raise SystemExit(1)
        except Exception as e:
            self.logger.error(f"Ошибка чтения промпта из файла: {e}")
            raise SystemExit(1)

    def _parse_prompt_tags(self, text: str) -> tuple[str, str]:
        """Извлекает теги #query: и #negativePrompt: из файла промта.

        Поддерживаются два формата:
        1) Однострочный: `#query: ...`, `#negativePrompt: ...`
        2) Блочный: строки после тега относятся к нему, пока не встретится другой тег
        Остальные строки (без тегов) считаются частью основного промта.
        """
        query_parts: list[str] = []
        negative_parts: list[str] = []
        other_lines: list[str] = []
        mode: str | None = None  # 'query' | 'negative' | None

        for raw in text.splitlines():
            line = raw.rstrip()
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith('#query:'):
                mode = 'query'
                after = stripped.split(':', 1)[1].strip()
                if after:
                    query_parts.append(after)
                continue
            if lower.startswith('#negativeprompt:'):
                mode = 'negative'
                after = stripped.split(':', 1)[1].strip()
                if after:
                    negative_parts.append(after)
                continue

            # не теговая строка
            if not stripped:
                continue
            if stripped.startswith('#'):
                # коммент, игнорируем
                continue
            if mode == 'query':
                query_parts.append(stripped)
            elif mode == 'negative':
                negative_parts.append(stripped)
            else:
                other_lines.append(stripped)

        query_text = " ".join([p for p in query_parts if p]) if query_parts else (
            " ".join(other_lines) if other_lines else text
        )
        negative_text = " ".join([p for p in negative_parts if p]) if negative_parts else ""
        return query_text, negative_text

    def _normalize_and_validate_prompt(self, text: str, source: str = "-p") -> str:
        # Схлопываем многострочные/избыточные пробелы
        normalized = " ".join(text.split())
        max_len = 1000
        if len(normalized) > max_len:
            self.logger.warning(f"Промпт из {source} длиннее {max_len} символов и будет обрезан")
            normalized = normalized[:max_len]
        if not normalized:
            self.logger.error("Пустой промпт после нормализации")
            raise SystemExit(1)
        return normalized

    def _generate_result_filename(self, style: str, job_uuid: str) -> str:
        """Генерация имени файла: result_{STYLE}_{id}-{nn}.jpg

        id — последние 12 символов uuid задания.
        """
        style_token = (style or "DEFAULT").strip() or "DEFAULT"
        style_token = style_token.upper().replace(" ", "_")
        uid_tail = (job_uuid or "")[-12:] if job_uuid else f"{int(time.time())}"
        candidate = f"result_{style_token}_{uid_tail}.jpg"
        return candidate

    def get_pipeline(self):
        self.logger.debug(f"Получение списка пайплайнов")
        response = self.session.get(self.URL + self.KEYS["PIPELINES"], timeout=self.TIMEOUT)
        if "application/json" not in response.headers.get("Content-Type", ""):
            self.logger.error(f"Bad response: {response.status_code} {response.text}")
            return None
        data = response.json()
        if not data:
            self.logger.error("Пустой список пайплайнов")
            return None
        # Документация указывает поле id у элемента
        pipeline_id = data[0].get("id") or data[0].get("uuid")
        self.logger.info(f"Выбран пайплайн id: {pipeline_id}")
        return pipeline_id

    def availability_service(self, pipeline_id: str):
        try:
            params = {'pipeline_id': pipeline_id}
            response = self.session.get(self.URL + self.KEYS["availability"], params=params, timeout=self.TIMEOUT)
            if "application/json" not in response.headers.get("Content-Type", ""):
                self.logger.warning(f"Availability non-JSON: {response.status_code} {response.text}. Продолжаем.")
                return True
            data = response.json()
            status = data.get('pipeline_status') or data.get('status')
            self.logger.info(f"Статус пайплайна <{pipeline_id}>: {status}")
            if status and status in ("DISABLED_BY_QUEUE", "OK", "ACTIVE", "ENABLED"):
                return True
            self.logger.warning('Пайплайн может быть недоступен, продолжаем попытку запуска')
            return True
        except Exception as e:
            self.logger.warning(f"Проверка доступности не удалась: {e}. Продолжаем.")
            return True

    def is_ready(self, pipeline_id: str, detailed: bool = False) -> bool:
        """Проверка готовности пайплайна.

        Возвращает True, если статус ACTIVE/OK/ENABLED, иначе:
        - при detailed=False возвращает False
        - при detailed=True выбрасывает RuntimeError с полем body (ответ сервера)
        """
        params = {'pipeline_id': pipeline_id}
        response = self.session.get(self.URL + self.KEYS["availability"], params=params, timeout=self.TIMEOUT)
        content_type = response.headers.get("Content-Type", "")
        body: dict | str
        if "application/json" in content_type:
            try:
                body = response.json()
            except Exception:
                body = response.text
        else:
            body = response.text

        status = None
        if isinstance(body, dict):
            status = body.get('pipeline_status') or body.get('status')

        ready_values = {"ACTIVE", "OK", "ENABLED"}
        is_ready = (status in ready_values) if status is not None else (response.ok)
        if is_ready:
            return True
        if detailed:
            err = RuntimeError("MODEL_NOT_READY")
            setattr(err, 'body', body)
            raise err
        return False

    def generate(self, prompt: str, style: str, neg_prompt: str, pipeline_id: str, images=1, width=1024, height=1024): #TODO переписать под асинхронность
        params = {
            "type": "GENERATE",
            "numImages": images,
            "width": width,
            "height": height,
            # поля в корне согласно документации
            **({"style": style} if style else {}),
            **({"negativePromptDecoder": neg_prompt} if neg_prompt else {}),
            "generateParams": {
                "query": f"{prompt}"
            }
        }

        data = {
            'pipeline_id': (None, pipeline_id),
            'params': (None, json.dumps(params), 'application/json')
        }
        if self.DEBUG:
            try:
                debug_request = {
                    'url': self.URL + self.KEYS['RUN'],
                    'pipeline_id': pipeline_id,
                    'params': params,
                }
                self.logger.debug(json.dumps(debug_request, ensure_ascii=False, indent=2))
            except Exception:
                # на случай проблем сериализации
                self.logger.debug(f"REQUEST url={self.URL + self.KEYS['RUN']} pipeline_id={pipeline_id} params={params}")
        response = self.session.post(self.URL + self.KEYS['RUN'], files=data, timeout=self.TIMEOUT)
        if "application/json" not in response.headers.get("Content-Type", ""):
            self.logger.error(f"Bad response: {response.status_code} {response.text}")
            raise RuntimeError("API returned non-JSON for RUN")
        data = response.json()
        self.logger.debug(data)
        self.logger.info(f"Генерация запущена под id: {data['uuid']}")
        return data['uuid']

    def check_generation(self, request_id, attempts=10, delay=10):
        while attempts > 0:
            self.logger.debug(f'Проверка генерации <{request_id}> осталось попыток {attempts}')
            response = self.session.get(self.URL + self.KEYS['status'] + request_id, timeout=self.TIMEOUT)
            if "application/json" not in response.headers.get("Content-Type", ""):
                self.logger.error(f"Bad response: {response.status_code} {response.text}")
                time.sleep(delay)
                attempts -= 1
                continue
            data = response.json()
            if data.get('status') == 'DONE':
                result = data.get('result', {})
                files = result.get('files') or []
                censored = result.get('censored')
                self.logger.info(f"Генерация <{request_id}> успешна\n"
                                 f"Цензура: {censored}")
                if not files:
                    self.logger.error('Файлы результата пусты')
                    return None
                return files[0]
            attempts -= 1
            time.sleep(delay)

    def base64_to_image(self, base64_string: str, show: bool = True, save: bool = True, filename: str = "image.jpg"):
        # поддержка URL файлов, которые возвращает API в result.files
        if base64_string.startswith("http://") or base64_string.startswith("https://"):
            try:
                resp = requests.get(base64_string, timeout=60)
                resp.raise_for_status()
                image_bytes = resp.content
                image_stream = BytesIO(image_bytes)
                img = Image.open(image_stream)
                if show:
                    img.show()
                if save:
                    img.save(filename)
                self.logger.info(f'Загрузка изображения по URL прошла успешно')
                return
            except Exception as e:
                self.logger.error(f"Не удалось скачать изображение по URL: {e}")
                return
        if "data:image" in base64_string:
            base64_string = base64_string.split(",")[1]
        image_bytes = base64.b64decode(base64_string)
        image_stream = BytesIO(image_bytes)
        img = Image.open(image_stream)
        if show:
            img.show()
        if save:
            img.save(filename)
        self.logger.info(f'Конвертация изображения прошла успешно')


def _parse_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    return val in {"1", "true", "t", "yes", "y"}


def _resolve_style_argument(style_arg: str | None) -> str:
    """Парсит аргумент стиля и, если возможно, нормализует по пресетам с "приближённым" совпадением.

    Ожидаемый ввод: только имя стиля (например, "ANIME", "Кандинский", "фото").
    Если найден `style_presets.json`, попытка найти лучшее совпадение среди name/title/titleEn
    с учётом регистра, подстрок и нестрогого сравнения.
    Возвращает canonical `name` при успехе, иначе исходное значение.
    """
    if not style_arg:
        return "DEFAULT"
    candidate = str(style_arg).strip()
    if not candidate:
        return "DEFAULT"

    def norm(s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    presets_path = 'style_presets.json'
    try:
        if os.path.exists(presets_path):
            with open(presets_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            target = norm(candidate)
            best_name = None
            best_score = 0.0
            from difflib import SequenceMatcher
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    fields = [item.get('name', ''), item.get('title', ''), item.get('titleEn', '')]
                    for val in fields:
                        v = norm(val)
                        if not v:
                            continue
                        # точное совпадение
                        if target == v:
                            return item.get('name', candidate)
                        # начинается с / содержит
                        score = 0.0
                        if v.startswith(target) or target.startswith(v):
                            score = 0.9
                        elif target in v or v in target:
                            score = 0.8
                        else:
                            score = SequenceMatcher(None, target, v).ratio()
                        if score > best_score:
                            best_score = score
                            best_name = item.get('name', candidate)
            if best_name and best_score >= 0.6:
                return best_name
    except Exception:
        pass
    return "DEFAULT"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text2Image')
    parser.add_argument('-p', '--prompt', type=str, default=None, help='краткий промт текстом')
    parser.add_argument('--prompt-file', type=str, default=None, help='файл с промтом (UTF-8)')
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('-st', '--style', type=str, default='')
    parser.add_argument('-np', '--ngprompt', type=str, default='')
    # keep backward compatibility for -sh/-s accepting 1/0, true/false
    parser.add_argument('-sh', '--show', type=str, default='0', help='1/0, true/false')
    parser.add_argument('-s', '--save', type=str, default='1', help='1/0, true/false')
    # debug: поддержка как "-db" без значения (включает), так и "-db=0/1"
    parser.add_argument('-db', '--debug', type=str, nargs='?', const='1', default='0', help='1/0, true/false; просто -db включает отладку')
    parser.add_argument('-o', '--outfile', type=str, default=None, help='output image filename')
    args = parser.parse_args()

    show = _parse_bool(args.show)
    save = _parse_bool(args.save)
    debug = _parse_bool(args.debug)

    resolved_style = _resolve_style_argument(args.style)
    Text2ImageAPI(args.prompt, args.width, args.height, resolved_style, args.ngprompt, show, save, debug, args.outfile, args.prompt_file)
