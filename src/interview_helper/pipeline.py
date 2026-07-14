"""Общий цикл пайплайна: захват -> транскрипция -> ответы, с колбэком событий.

События emit(): {"type": "status"|"utterance"|"answer_start"|"answer_delta"|"answer_end", ...}
Используется и терминальным main.py, и веб-обёрткой web.py.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

from .answer import Answerer
from .capture import AudioChunk, start_file_capture, start_live_capture
from .transcribe import Transcriber, Utterance

Emit = Callable[[dict], None]


def run_pipeline(
    emit: Emit,
    stop: threading.Event,
    input_file: str | None = None,
    model_size: str = "small",
    mic_device: int | None = None,
    loopback_device: int | None = None,
    language: str | None = None,
    answer_mic: bool = False,
    register_ask: Callable[[Callable[[str], None]], None] | None = None,
    answer_model: str = "haiku",
    effort: str | None = None,
) -> None:
    chunks: queue.Queue[AudioChunk] = queue.Queue()
    try:
        _run(
            emit,
            stop,
            chunks,
            input_file,
            model_size,
            mic_device,
            loopback_device,
            language,
            answer_mic,
            register_ask,
            answer_model,
            effort,
        )
    except Exception as e:  # noqa: BLE001 — поток фоновый, ошибку показываем пользователю
        emit({"type": "status", "text": f"Ошибка: {e!r}"})
    finally:
        emit({"type": "status", "text": "Остановлено"})


def _run(
    emit: Emit,
    stop: threading.Event,
    chunks: queue.Queue[AudioChunk],
    input_file: str | None,
    model_size: str,
    mic_device: int | None,
    loopback_device: int | None,
    language: str | None,
    answer_mic: bool,
    register_ask: Callable[[Callable[[str], None]], None] | None = None,
    answer_model: str = "haiku",
    effort: str | None = None,
) -> None:
    from .session_log import SessionLog

    emit({"type": "status", "text": f"Загружаю whisper ({model_size})..."})
    transcriber = Transcriber(model_size=model_size, language=language)
    answerer = Answerer(model=answer_model, answer_mic=answer_mic, effort=effort)
    log = SessionLog(answer_model, answerer.effort)
    effort_note = f", effort {answerer.effort}" if answerer.effort else ""
    ctx_note = (
        f", контекст: {', '.join(answerer.context_names)}"
        if answerer.context_names
        else ", контекст: нет (context/vacancy.md, context/resume.md)"
    )
    emit(
        {
            "type": "status",
            "text": f"whisper {model_size} на {transcriber.device}, "
            f"ответы: {answer_model}{effort_note}{ctx_note}, лог: {log.path}",
        }
    )

    if input_file:
        capture_stop = start_file_capture(chunks, input_file)
        emit({"type": "status", "text": f"Читаю файл {input_file}"})
    else:
        capture_stop = start_live_capture(
            chunks, mic_device=mic_device, loopback_device=loopback_device
        )
        emit({"type": "status", "text": "Слушаю встречу и микрофон"})

    answer_seq = 0

    def answer_worker(utt, aid: int, do_triage: bool) -> None:
        emit({"type": "answer_start", "id": aid, "question": utt.text})
        # триаж крутится параллельно основному ответу и убивает ИМЕННО его процесс
        # (proc_ref), не общий answerer._proc — иначе гонка со следующим вопросом
        proc_ref: dict[str, object] = {}
        discarded = threading.Event()

        def gate() -> None:
            if not answerer.triage(utt):
                discarded.set()
                proc = proc_ref.get("p")
                if proc is not None and proc.poll() is None:  # type: ignore[union-attr]
                    proc.kill()  # type: ignore[union-attr]

        gate_thread: threading.Thread | None = None
        if do_triage:
            gate_thread = threading.Thread(target=gate, daemon=True)
            gate_thread.start()

        parts = []
        for delta in answerer.stream_answer(utt, on_proc=lambda p: proc_ref.__setitem__("p", p)):
            if discarded.is_set():
                break
            if stop.is_set():
                answerer.cancel()
                break
            parts.append(delta)
            emit({"type": "answer_delta", "id": aid, "text": delta})

        if gate_thread is not None:
            gate_thread.join(timeout=3.0)  # короткий ответ мог опередить вердикт триажа
        if discarded.is_set():
            emit({"type": "answer_discard", "id": aid})  # стираем баббл-мусор в UI
            return
        emit({"type": "answer_end", "id": aid})
        if parts:
            log.answer(utt.text, "".join(parts))

    def ask(utt, do_triage: bool = True) -> None:
        # ответ стримится в фоне, чтобы не блокировать транскрипцию;
        # новый вопрос вытесняет недописанный ответ на предыдущий
        answerer.cancel()
        nonlocal answer_seq
        answer_seq += 1
        threading.Thread(
            target=answer_worker, args=(utt, answer_seq, do_triage), daemon=True
        ).start()

    def handle(utt) -> None:
        emit({"type": "utterance", "source": utt.source, "text": utt.text})
        log.utterance(utt)
        answerer.add(utt)
        if answerer.is_question(utt):
            ask(utt)

    if register_ask is not None:
        # force-ask из UI: реплика уже в history, только запускаем ответ
        import time as _time

        # ручной запрос из UI — отвечаем всегда, триаж не применяем
        register_ask(lambda text: ask(Utterance("loopback", text, _time.time()), do_triage=False))

    try:
        idle = 0
        while not stop.is_set():
            try:
                chunk = chunks.get(timeout=1)
            except queue.Empty:
                # тишина: дожимаем недосказанные реплики (loopback молчит без звука)
                for utt in transcriber.flush_stale():
                    handle(utt)
                idle += 1
                if input_file and idle >= 5:
                    break  # файл закончился
                continue
            idle = 0
            utt = transcriber.feed(chunk)
            if utt is not None:
                handle(utt)
    finally:
        capture_stop.set()
        log.close()
