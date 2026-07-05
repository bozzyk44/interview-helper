"""Ответы на вопросы интервьюера через `claude -p` (Haiku), стримингом."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from .transcribe import Utterance

SYSTEM_PROMPT = (
    "Ты — суфлёр кандидата на собеседовании. Тебе дают транскрипт разговора "
    "([interviewer] — интервьюер, [me] — кандидат) и последний вопрос. "
    "Ответь от первого лица кандидата: сжато, 2-5 предложений или короткий список, "
    "без преамбул. Отвечай на языке вопроса."
)

CONTEXT_CHARS = 6000  # ~2000 токенов скользящего окна

# вопрос распознаём, если одно из этих слов встретилось в начале реплики
QUESTION_WORDS = (
    "расскаж",
    "подскаж",
    "объясн",
    "опиши",
    "почему",
    "зачем",
    "кто",
    "что",
    "как",
    "какой",
    "какая",
    "какие",
    "каких",
    "сколько",
    "где",
    "когда",
    "чем",
    "можете",
    "можешь",
    "есть ли",
    "в чём",
    "в чем",
    "what",
    "how",
    "why",
    "who",
    "when",
    "where",
    "which",
    "can",
    "could",
    "tell",
    "explain",
    "describe",
    "difference",
)


class Answerer:
    def __init__(self, model: str = "haiku", answer_mic: bool = False) -> None:
        self.model = model
        self.answer_mic = answer_mic  # отладка: реагировать и на вопросы с микрофона
        self.history: deque[Utterance] = deque()
        self._proc: subprocess.Popen | None = None

    def add(self, utt: Utterance) -> None:
        self.history.append(utt)
        while sum(len(u.text) for u in self.history) > CONTEXT_CHARS:
            self.history.popleft()

    def is_question(self, utt: Utterance) -> bool:
        """Эвристика: «?» в тексте или вопросительное слово в первых словах реплики."""
        if utt.source != "loopback" and not self.answer_mic:
            return False
        text = utt.text.lower()
        if "?" in text:
            return True
        head = text.replace(",", " ").split()[:5]
        return any(word.startswith(q) for word in head for q in QUESTION_WORDS)

    def cancel(self) -> None:
        """Прерывает текущий ответ (новый вопрос вытесняет старый)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def stream_answer(self, question: Utterance) -> Iterator[str]:
        """Стримит текст ответа по мере генерации."""
        exe = shutil.which("claude")
        if exe is None:
            yield "[claude CLI не найден в PATH]"
            return
        transcript = "\n".join(
            f"[{'interviewer' if u.source == 'loopback' else 'me'}] {u.text}" for u in self.history
        )
        # системная инструкция и промпт уходят через stdin: никакого квотинга аргументов
        prompt = (
            f"{SYSTEM_PROMPT}\n\nТранскрипт:\n{transcript}\n\nВопрос интервьюера: {question.text}"
        )
        cmd = [
            "claude",
            "-p",
            "--model",
            self.model,
            "--output-format",
            "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--strict-mcp-config",  # не ждать внешних MCP-серверов
            "--max-turns",
            "1",  # только текстовый ответ, без инструментов
        ]
        if exe.lower().endswith((".cmd", ".bat")):  # npm-шим нельзя запустить без cmd.exe
            cmd = ["cmd", "/c", *cmd]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            # нейтральный cwd: иначе headless подхватит CLAUDE.md и хуки этого репо
            cwd=Path.home(),
        )
        self._proc = proc
        assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
        proc.stdin.write(prompt)
        proc.stdin.close()
        got_delta = False
        for line in proc.stdout:
            delta = extract_delta(line)
            if delta:
                got_delta = True
                yield delta
        code = proc.wait()
        if code != 0 and not got_delta and self._proc is proc:  # kill() при отмене — не ошибка
            err = proc.stderr.read().strip()
            yield f"[claude завершился с кодом {code}: {err[-500:] or 'нет stderr'}]"


def extract_delta(line: str) -> str:
    """Текстовая дельта из события stream-json; пустая строка, если её нет."""
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return ""
    if event.get("type") != "stream_event":
        return ""
    inner = event.get("event", {})
    if inner.get("type") != "content_block_delta":
        return ""
    delta = inner.get("delta", {})
    return delta.get("text", "") if delta.get("type") == "text_delta" else ""
