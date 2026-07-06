"""Пост-сессионный отчёт: саммари, разбор Q&A и оценка по транскрипту из sessions/."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .answer import load_context, stream_claude
from .session_log import SESSIONS_DIR

REPORT_PROMPT = (
    "Ты — карьерный коуч, разбирающий прошедшее собеседование по его транскрипту "
    "([интервьюер] и [я] — кандидат; блоки «Ответ» — подсказки суфлёра, а не слова кандидата). "
    "Составь отчёт на русском в markdown:\n"
    "1. **Саммари встречи** — 5-10 предложений: о чём говорили, ключевые темы.\n"
    "2. **Вопросы и ответы** — по каждому вопросу интервьюера: как ответил кандидат "
    "и как стоило ответить лучше (если стоило).\n"
    "3. **Оценка** — сильные и слабые стороны, вердикт 1-10 с обоснованием, "
    "3 конкретных рекомендации к следующей сессии.\n"
    "Если дан контекст вакансии/резюме — оценивай относительно требований роли."
)


def latest_transcript() -> Path | None:
    files = sorted(
        (p for p in SESSIONS_DIR.glob("*.md") if not p.name.endswith(".report.md")),
        key=lambda p: p.name,
    )
    return files[-1] if files else None


def stream_report(transcript_path: Path) -> Iterator[str]:
    """Стримит текст отчёта; по завершении вызывающий сохраняет его рядом с транскриптом."""
    transcript = transcript_path.read_text(encoding="utf-8")
    context, _ = load_context()
    context_block = f"\n\n{context}" if context else ""
    prompt = f"{REPORT_PROMPT}{context_block}\n\nТранскрипт сессии:\n{transcript}"
    # латентность не критична: sonnet + high даёт вдумчивый разбор
    yield from stream_claude(prompt, model="sonnet", effort="high")


def report_path_for(transcript_path: Path) -> Path:
    return transcript_path.with_suffix(".report.md")
