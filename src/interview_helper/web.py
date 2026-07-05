"""Веб-обёртка: одна страница, старт/стоп сессии, живой транскрипт и ответы через SSE.

Запуск: uv run python -m interview_helper.web  (http://localhost:8765)
"""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .pipeline import run_pipeline

app = FastAPI(title="interview-helper")

_lock = threading.Lock()
_stop: threading.Event | None = None
_thread: threading.Thread | None = None
_subscribers: list[queue.Queue[dict]] = []
_ask = None  # force-ask текущей сессии, выставляется пайплайном


def _emit(event: dict) -> None:
    for q in list(_subscribers):
        q.put(event)


class StartRequest(BaseModel):
    mode: str = "live"  # live | file
    path: str | None = None
    model: str = "small"
    mic_device: int | None = None
    loopback_device: int | None = None
    language: str | None = None  # None = автоопределение, иначе "ru" / "en"
    answer_mic: bool = False
    answer_model: str = "haiku"  # haiku | sonnet | opus
    effort: str | None = (
        None  # low | medium | high | xhigh | max (haiku: игнорируется, opus: <= medium)
    )


@app.get("/api/devices")
def devices() -> dict:
    from .capture import list_devices

    try:
        return list_devices()
    except Exception as e:  # noqa: BLE001 — нет аудиоустройств и т.п.
        return {"mic": [], "loopback": [], "error": str(e)}


@app.post("/api/start")
def start(req: StartRequest) -> dict:
    global _stop, _thread
    with _lock:
        if _thread is not None and _thread.is_alive():
            return {"ok": False, "error": "Сессия уже идёт"}
        input_file = None
        if req.mode == "file":
            if not req.path or not Path(req.path).exists():
                return {"ok": False, "error": f"Файл не найден: {req.path!r}"}
            input_file = req.path
        _stop = threading.Event()
        _thread = threading.Thread(
            target=run_pipeline,
            args=(_emit, _stop),
            kwargs={
                "input_file": input_file,
                "model_size": req.model,
                "mic_device": req.mic_device,
                "loopback_device": req.loopback_device,
                "language": req.language,
                "answer_mic": req.answer_mic,
                "register_ask": _register_ask,
                "answer_model": req.answer_model,
                "effort": req.effort,
            },
            daemon=True,
        )
        _thread.start()
    return {"ok": True}


def _register_ask(fn) -> None:
    global _ask
    _ask = fn


class AskRequest(BaseModel):
    text: str


@app.post("/api/ask")
def ask(req: AskRequest) -> dict:
    """Force-LLM: отправить реплику на ответ вручную, минуя детекцию вопроса."""
    if _ask is None or _thread is None or not _thread.is_alive():
        return {"ok": False, "error": "Сессия не запущена"}
    _ask(req.text)
    return {"ok": True}


@app.post("/api/stop")
def stop() -> dict:
    if _stop is not None:
        _stop.set()
    return {"ok": True}


@app.get("/api/events")
def events() -> StreamingResponse:
    q: queue.Queue[dict] = queue.Queue()
    _subscribers.append(q)

    def gen():
        try:
            while True:
                try:
                    event = q.get(timeout=15)
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            _subscribers.remove(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)


if __name__ == "__main__":
    main()
