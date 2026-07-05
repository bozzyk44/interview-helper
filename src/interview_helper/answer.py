"""Ответы на вопросы интервьюера через `claude -p` (Haiku), стримингом."""

from __future__ import annotations

import json
import subprocess
from collections import deque
from collections.abc import Iterator

from .transcribe import Utterance

SYSTEM_PROMPT = (
    "Ты — суфлёр кандидата на собеседовании. Тебе дают транскрипт разговора "
    "([interviewer] — интервьюер, [me] — кандидат) и последний вопрос. "
    "Ответь от первого лица кандидата: сжато, 2-5 предложений или короткий список, "
    "без преамбул. Отвечай на языке вопроса."
)

CONTEXT_CHARS = 6000  # ~2000 токенов скользящего окна


class Answerer:
    def __init__(self, model: str = "haiku") -> None:
        self.model = model
        self.history: deque[Utterance] = deque()

    def add(self, utt: Utterance) -> None:
        self.history.append(utt)
        while sum(len(u.text) for u in self.history) > CONTEXT_CHARS:
            self.history.popleft()

    def is_question(self, utt: Utterance) -> bool:
        """MVP-эвристика: реплика интервьюера с вопросительным знаком или вопросительным словом."""
        if utt.source != "loopback":
            return False
        text = utt.text.lower()
        starters = (
            "расскаж",
            "как ",
            "почему",
            "что ",
            "какой",
            "can you",
            "what",
            "how",
            "why",
            "tell me",
        )
        return "?" in text or text.startswith(starters)

    def stream_answer(self, question: Utterance) -> Iterator[str]:
        """Стримит текст ответа по мере генерации."""
        transcript = "\n".join(
            f"[{'interviewer' if u.source == 'loopback' else 'me'}] {u.text}" for u in self.history
        )
        prompt = f"Транскрипт:\n{transcript}\n\nВопрос интервьюера: {question.text}"
        proc = subprocess.Popen(
            [
                "claude",
                "-p",
                prompt,
                "--model",
                self.model,
                "--system-prompt",
                SYSTEM_PROMPT,
                "--output-format",
                "stream-json",
                "--verbose",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            shell=True,  # claude — это .cmd-шим на Windows
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            text = extract_text(line)
            if text:
                yield text
        proc.wait()


def extract_text(line: str) -> str:
    """Достаёт текстовые дельты из stream-json строки; пустая строка, если их нет."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if event.get("type") == "assistant":
        return "".join(
            block.get("text", "")
            for block in event.get("message", {}).get("content", [])
            if block.get("type") == "text"
        )
    return ""
