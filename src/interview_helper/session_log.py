"""Журнал сессии: транскрипт и ответы пишутся в sessions/<время>.md по ходу встречи."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .transcribe import Utterance

SESSIONS_DIR = Path("sessions")


class SessionLog:
    def __init__(self, answer_model: str, effort: str | None) -> None:
        SESSIONS_DIR.mkdir(exist_ok=True)
        self.path = SESSIONS_DIR / time.strftime("%Y%m%d-%H%M%S.md")
        self._lock = threading.Lock()
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"# Сессия {started}\n\n"
            f"Модель ответов: {answer_model}"
            f"{f', effort {effort}' if effort else ''}\n\n## Транскрипт\n\n"
        )

    def _write(self, text: str) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(text)

    def utterance(self, utt: Utterance) -> None:
        who = "интервьюер" if utt.source == "loopback" else "я"
        ts = time.strftime("%H:%M:%S", time.localtime(utt.timestamp))
        self._write(f"- `{ts}` **{who}:** {utt.text}\n")

    def answer(self, question: str, text: str) -> None:
        body = text.strip().replace("\n", "\n> ")
        self._write(f"\n> **Вопрос:** {question}\n> **Ответ:**\n> {body}\n\n")

    def close(self) -> None:
        self._write(f"\n---\nЗавершено {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
