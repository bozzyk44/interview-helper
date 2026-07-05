# interview-helper

Realtime-суфлёр для собеседований: захватывает звук встречи (WASAPI loopback) и микрофона, транскрибирует через faster-whisper и отвечает на вопросы интервьюера через Claude Haiku (`claude -p`) — всё в реальном времени, локально, на Windows.

## Как это работает

```
loopback (звук встречи) ─┐
                          ├─> faster-whisper (small, int8, VAD) ─> детекция вопроса ─> claude -p --model haiku ─> стрим ответа в терминал
микрофон (мой голос) ────┘
```

Реплики помечаются источником: вопросы приходят из loopback, ваши ответы с микрофона попадают в контекст разговора.

## Требования

- Windows 11 (WASAPI loopback)
- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Claude Code CLI (`claude`) с активной подпиской
- ~2 ГБ RAM и 4+ ядра для модели `small` — подробнее в [docs/resources.md](docs/resources.md)

## Запуск

```powershell
uv sync

# живая сессия (микрофон + звук системы)
uv run python -m interview_helper.main

# отладка на записанном WAV
uv run python -m interview_helper.main --input-file tests/fixtures/question.wav

# другая модель whisper
uv run python -m interview_helper.main --model base
```

## Разработка

```powershell
uv run pytest -q                                  # тесты
uv run ruff format . ; uv run ruff check --fix .  # формат + линт
uv run python -m interview_helper.bench file.wav  # бенчмарк латентности
```

Дорожная карта и архитектура — в [CLAUDE.md](CLAUDE.md), оценка ресурсов — в [docs/resources.md](docs/resources.md).

## Этика

Инструмент предназначен для личного использования владельцем: тренировка ответов, mock-интервью, разбор собственных сессий. Использование на реальных собеседованиях — под вашу ответственность с учётом правил конкретного работодателя.
