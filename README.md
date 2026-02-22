# mtslinkdownloader
`mtslinkdownloader` - это инструмент для загрузки и обработки записей вебинаров, предоставляемых сервисом MTS Link. Он автоматически загружает видео и аудиофайлы вебинаров, синхронизирует их и создает единый видеоролик.

## Установка
### poetry:
```bash
poetry add git+https://github.com/motattack/mtslinkdownloader.git
```

### pip:
```bash
pip install git+https://github.com/motattack/mtslinkdownloader.git
```

## Использование
Для использования `mtslinkdownloader`, просто вызовите в терминале `mtslinkdownloader` с URL записи вебинара в качестве аргумента.

Вот несколько примеров использования:
### Пример 1. Загрузка обычной записи:
```bash
mtslinkdownloader https://my.mts-link.ru/12345678/987654321/record-new/123456789/record-file/1234567890
```

### Пример 2. Загрузка быстрой встречи:
```bash
mtslinkdownloader https://my.mts-link.ru/12345678/987654321/record-new/123456789
```

### Пример 3. Загрузка приватной записи:
```bash
mtslinkdownloader https://my.mts-link.ru/12345678/987654321/record-new/123456789/record-file/1234567890 --session-id a1b2c3d4
```

> **Примечание**: Узнать свой `sessionId` можно в кукисах сайта (нужно быть авторизованным). [Пример](https://raw.githubusercontent.com/motattack/mtslinkdownloader/refs/heads/master/get_sessionId.mp4) как это можно сделать.

## Использование в проекте

Если вы хотите интегрировать `mtslinkdownloader` в свой проект, вы можете использовать функцию `fetch_webinar_data` для загрузки данных вебинара.

Например, для ссылки: https://my.mts-link.ru/12345678/987654321/record-new/123456789/record-file/1234567890, формат будет следующим:
```
https://my.mts-link.ru/{organization_id}/{room_id}/record-new/{event_sessions}/record-file/{record_id}
```

```python
from mtslinkdownloader.webinar import fetch_webinar_data

fetch_webinar_data(
  event_sessions='123456789',
  record_id='1234567890',   # Нужен для обычной встречи, не нужен для быстрой.
  session_id='a1b2c3d4'    # Optional
)
```

## Docker
Для запуска через Docker:

1. Соберите образ:
   ```bash
   docker-compose build
   ```
2. Запустите контейнер:
   ```bash
   docker-compose run --rm mtslinkdownloader [URL] [--session-id SESSION_ID]
   ```
#### Пример с аргументами   
   ```bash
   docker-compose run --rm mtslinkdownloader https://my.mts-link.ru/12345678/987654321/record-new/123456789/record-file/1234567890 --session-id a1b2c3d4
   ```
