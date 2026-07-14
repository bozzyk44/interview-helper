"""Ответы на вопросы интервьюера через `claude -p` (Haiku), стримингом."""

from __future__ import annotations

import difflib
import json
import shutil
import subprocess
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path

from .transcribe import Utterance

SYSTEM_PROMPT = (
    "Ты — суфлёр кандидата на собеседовании. Тебе дают транскрипт разговора "
    "([interviewer] — интервьюер, [me] — кандидат) и последний вопрос. "
    "Ответь от первого лица кандидата, без преамбул. Структура: 1-2 предложения сути, "
    "затем маркированный список ключевых пунктов с конкретикой (термины, цифры, названия), "
    "и короткий пример из практики, если уместен. Не растекайся, но не жертвуй "
    "содержанием ради краткости. Отвечай на языке вопроса. Если ниже даны разделы "
    "«Вакансия» и «Резюме кандидата» — отвечай в их контексте: делай акценты на "
    "требованиях вакансии и опирайся на реальный опыт из резюме, не выдумывая новый."
)

CONTEXT_CHARS = 6000  # ~2000 токенов скользящего окна

# Параллельный триаж: пока основной ответ уже стримится, дешёвый Haiku решает,
# осмысленный ли это вопрос. NO — реплика без запроса информации (приветствие,
# поддакивание, обрывок, шум распознавания). Работает в параллель, чтобы не
# добавлять латентности; вердикт NO рубит основной ответ и стирает баббл.
TRIAGE_PROMPT = (
    "Ты фильтр суфлёра на собеседовании. Дан хвост транскрипта и последняя реплика "
    "интервьюера. Нужен ли кандидату содержательный ответ-подсказка на эту реплику? "
    "Ответь строго одним словом: YES или NO. NO — если это приветствие, поддакивание, "
    "реплика ни о чём, обрывок фразы, шум распознавания или короткая социальная фраза "
    "без запроса информации. Если сомневаешься — YES."
)
TRIAGE_CHARS = 1500  # хвоста транскрипта хватает для контекста, а Haiku отвечает быстрее

CONTEXT_DIR = Path("context")
CONTEXT_FILES = (("vacancy.md", "Вакансия"), ("resume.md", "Резюме кандидата"))
CONTEXT_FILE_CHARS = 8000  # ~2500 токенов на файл, чтобы не раздувать промпт


def load_context() -> tuple[str, list[str]]:
    """Контекст сессии из context/: текст для промпта и имена найденных файлов."""
    parts, names = [], []
    for filename, label in CONTEXT_FILES:
        path = CONTEXT_DIR / filename
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                parts.append(f"## {label}\n{text[:CONTEXT_FILE_CHARS]}")
                names.append(filename)
    return "\n\n".join(parts), names


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
    "разниц",
    "отлича",
)

# полные формы для fuzzy-сопоставления: whisper часто коверкает первое слово
# ("Весните" вместо "Объясните"), точный префикс такое не ловит
FUZZY_QUESTION_WORDS = (
    "расскажите",
    "подскажите",
    "объясните",
    "опишите",
    "почему",
    "зачем",
    "сколько",
    "можете",
    "explain",
    "describe",
)


def _fuzzy_question_word(word: str) -> bool:
    # императив на -ите/-йте («покажите», исковерканное «весните») — просьба
    if len(word) >= 6 and word.endswith(("ите", "йте")):
        return True
    return any(difflib.SequenceMatcher(None, word, q).ratio() >= 0.72 for q in FUZZY_QUESTION_WORDS)


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def resolve_effort(model: str, effort: str | None) -> str | None:
    """Effort с ограничениями: haiku не поддерживает effort вовсе, opus капнут на medium."""
    if effort is None or effort not in EFFORT_LEVELS or model == "haiku":
        return None
    if model == "opus" and EFFORT_LEVELS.index(effort) > EFFORT_LEVELS.index("medium"):
        return "medium"
    return effort


class Answerer:
    def __init__(
        self, model: str = "haiku", answer_mic: bool = False, effort: str | None = None
    ) -> None:
        self.model = model
        self.effort = resolve_effort(model, effort)
        self.answer_mic = answer_mic  # отладка: реагировать и на вопросы с микрофона
        self.context, self.context_names = load_context()
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
        head = text.replace(",", " ").replace(".", " ").split()[:6]
        if "ли" in head:  # косвенный вопрос без интонации: «интересно, сработает ли...»
            return True
        if any(word.startswith(q) for word in head for q in QUESTION_WORDS):
            return True
        return any(_fuzzy_question_word(word) for word in head)

    def _transcript(self) -> str:
        return "\n".join(
            f"[{'interviewer' if u.source == 'loopback' else 'me'}] {u.text}" for u in self.history
        )

    def cancel(self) -> None:
        """Прерывает текущий ответ (новый вопрос вытесняет старый)."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def triage(self, question: Utterance) -> bool:
        """Осмысленный ли это вопрос? Отдельный дешёвый Haiku-вызов, параллельно
        основному ответу. Fail-open: при ошибке/пустом ответе считаем, что вопрос
        стоящий — молча съесть настоящий вопрос хуже, чем показать лишний ответ."""
        prompt = (
            f"{TRIAGE_PROMPT}\n\nТранскрипт:\n{self._transcript()[-TRIAGE_CHARS:]}"
            f"\n\nРеплика интервьюера: {question.text}"
        )
        verdict = "".join(stream_claude(prompt, "haiku")).strip().upper()
        return not verdict.startswith("NO")

    def stream_answer(
        self,
        question: Utterance,
        on_proc: Callable[[subprocess.Popen], None] | None = None,
    ) -> Iterator[str]:
        """Стримит текст ответа по мере генерации."""
        context_block = f"\n\n{self.context}" if self.context else ""
        prompt = (
            f"{SYSTEM_PROMPT}{context_block}\n\n"
            f"Транскрипт:\n{self._transcript()}\n\nВопрос интервьюера: {question.text}"
        )

        def hold(proc: subprocess.Popen) -> None:
            self._proc = proc
            if on_proc is not None:
                on_proc(proc)

        yield from stream_claude(
            prompt,
            self.model,
            self.effort,
            on_proc=hold,
            was_cancelled=lambda p: self._proc is not p,
        )


def stream_claude(
    prompt: str,
    model: str,
    effort: str | None = None,
    on_proc: Callable[[subprocess.Popen], None] | None = None,
    was_cancelled: Callable[[subprocess.Popen], bool] | None = None,
) -> Iterator[str]:
    """Одноразовый headless-вызов claude -p со стримингом текстовых дельт."""
    exe = shutil.which("claude")
    if exe is None:
        yield "[claude CLI не найден в PATH]"
        return
    cmd = [
        "claude",
        "-p",
        "--model",
        model,
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--strict-mcp-config",  # не ждать внешних MCP-серверов
        "--max-turns",
        "1",  # только текстовый ответ, без инструментов
    ]
    if effort is not None:
        cmd += ["--effort", effort]
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
    if on_proc is not None:
        on_proc(proc)
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    # промпт через stdin: никакого квотинга аргументов Windows
    proc.stdin.write(prompt)
    proc.stdin.close()
    got_delta = False
    for line in proc.stdout:
        delta = extract_delta(line)
        if delta:
            got_delta = True
            yield delta
    code = proc.wait()
    cancelled = was_cancelled(proc) if was_cancelled is not None else False
    if code != 0 and not got_delta and not cancelled:  # kill() при отмене — не ошибка
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
