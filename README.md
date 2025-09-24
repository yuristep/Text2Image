# Text2Image - CLI генератор изображений

CLI-инструмент для генерации изображений по текстовому описанию через FusionBrain API (Kandinsky).

## 🚀 Возможности

- **Генерация изображений** по текстовому описанию через FusionBrain API
- **Умная обработка стилей** - автоматическое сопоставление пользовательского ввода с доступными стилями
- **Гибкие источники промптов** - из командной строки, файла или автоматически из `examples/promt.txt`
- **Автоматическое именование файлов** - `result_{STYLE}_{id}.jpg` при сохранении
- **Поддержка негативных промптов** - парсинг из файла с тегами `#negativePrompt:`
- **Режим отладки** - подробное логирование запросов и ответов API
- **Нормализация промптов** - автоматическое сокращение до 1000 символов с предупреждениями

## 📦 Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/yuristep/Text2Image.git
cd Text2Image
```

2. **Создайте виртуальное окружение:**
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# или
source .venv/bin/activate  # Linux/Mac
```

3. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

4. **Настройте API ключи:**
Создайте файл `.env` в корне проекта:
```env
# FusionBrain API ключи
# Получите ключи на https://fusionbrain.ai/
api_key=YOUR_API_KEY_HERE
secret_key=YOUR_SECRET_KEY_HERE
```

## 🎯 Использование

### Быстрый старт
```bash
python main.py -p="Пушистый кот в очках" -st="реализм" -sh=1 -s=1
```

### Промт из файла
```bash
python main.py --prompt-file="examples/promt.txt" -st="реализм" -sh=1 -o output.jpg
```

### Автоматический промт из examples/promt.txt
```bash
python main.py -st="реализм" -sh=1
```

### Кастомные размеры
```bash
python main.py -p="Пейзаж" --width=1920 --height=1080 -st="UHD" -sh=1
```

### Режим отладки
```bash
python main.py -p="Тест" -st="DEFAULT" -db -sh=1
```

## 📋 Параметры командной строки

| Флаг | Описание | Тип | По умолчанию |
|------|----------|-----|--------------|
| `-p, --prompt` | Текст промта | str | 'Кот в очках' |
| `--prompt-file` | Путь к файлу с промтом | str | `examples/promt.txt` |
| `--width` | Ширина изображения | int | 1024 |
| `--height` | Высота изображения | int | 1024 |
| `-st, --style` | Стиль изображения | str | '' |
| `-np, --ngprompt` | Негативный промт | str | '' |
| `-sh, --show` | Показать результат | bool | False |
| `-s, --save` | Сохранить результат | bool | True |
| `-db, --debug` | Режим отладки | bool | False |
| `-o, --outfile` | Имя файла для сохранения | str | auto |

## 🎨 Доступные стили

Скрипт поддерживает "умное" сопоставление стилей. Вы можете использовать:
- **Канонические имена**: `KANDINSKY`, `UHD`, `ANIME`, `DEFAULT`
- **Русские названия**: `Кандинский`, `Детальное фото`, `Аниме`, `Свой стиль`
- **Английские названия**: `Kandinsky`, `Detailed photo`, `Anime`, `No style`

Для получения актуального списка стилей:
```bash
python styles.py
```

## 📁 Структура проекта

```
Text2Image/
├── main.py              # Основной скрипт
├── styles.py            # Скрипт получения стилей
├── settings.json        # Конфигурация API
├── style_presets.json   # Предустановленные стили
├── requirements.txt     # Зависимости Python
├── .env                 # API ключи (создать самостоятельно)
├── examples/
│   └── promt.txt       # Пример промта с тегами
├── log/                # Логи работы скрипта
└── README.md           # Документация
```

## 📝 Формат файла промпта

Файл `examples/promt.txt` поддерживает специальные теги:

```
#query: Основной промт для генерации изображения
#negativePrompt: Что не должно быть на изображении
```

## 🔧 Дополнительные инструменты

### Получение списка стилей
```bash
python styles.py -n 10                    # Показать первые 10 стилей
python styles.py --presets                 # Показать предустановленные стили
python styles.py -o styles.json           # Сохранить в файл
```

## ⚠️ Важные замечания

- **Ограничение промтов**: API ограничивает длину промта 1000 символами. Скрипт автоматически нормализует и обрезает длинные тексты с предупреждением в логе.
- **API ключи**: Получите ключи на [fusionbrain.ai](https://fusionbrain.ai/)
- **Документация API**: [fusionbrain.ai/docs](https://fusionbrain.ai/docs/doc/api-dokumentaciya/)
- **Логи**: Все операции логируются в папку `log/`

## 🐛 Решение проблем

### Ошибка "Не найдены api_key/secret_key"
- Убедитесь, что файл `.env` создан в корне проекта
- Проверьте правильность ключей API

### Ошибка "Could not find platform independent libraries"
- Пересоздайте виртуальное окружение:
```bash
rm -rf .venv
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Ошибка "KeyError: 0" при получении пайплайна
- Проверьте правильность API ключей
- Убедитесь в доступности сервиса FusionBrain

## 📄 Лицензия

MIT License