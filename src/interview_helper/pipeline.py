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
    emit({"type": "status", "text": f"Загружаю whisper ({model_size})..."})
    transcriber = Transcriber(model_size=model_size, language=language)
    answerer = Answerer(model=answer_model, answer_mic=answer_mic, effort=effort)
    effort_note = f", effort {answerer.effort}" if answerer.effort else ""
    emit(
        {
            "type": "status",
            "text": f"whisper {model_size} на {transcriber.device}, "
            f"ответы: {answer_model}{effort_note}",
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

    def answer_worker(utt, aid: int) -> None:
        emit({"type": "answer_start", "id": aid, "question": utt.text})
        for delta in answerer.stream_answer(utt):
            if stop.is_set():
                answerer.cancel()
                break
            emit({"type": "answer_delta", "id": aid, "text": delta})
        emit({"type": "answer_end", "id": aid})

    def ask(utt) -> None:
        # ответ стримится в фоне, чтобы не блокировать транскрипцию;
        # новый вопрос вытесняет недописанный ответ на предыдущий
        answerer.cancel()
        nonlocal answer_seq
        answer_seq += 1
        threading.Thread(target=answer_worker, args=(utt, answer_seq), daemon=True).start()

    def handle(utt) -> None:
        emit({"type": "utterance", "source": utt.source, "text": utt.text})
        answerer.add(utt)
        if answerer.is_question(utt):
            ask(utt)

    if register_ask is not None:
        # force-ask из UI: реплика уже в history, только запускаем ответ
        import time as _time

        register_ask(lambda text: ask(Utterance("loopback", text, _time.time())))

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
