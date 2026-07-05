"""Точка входа: связывает захват, транскрипцию и ответы; выводит всё в терминал."""

from __future__ import annotations

import argparse
import queue

from rich.console import Console

from .answer import Answerer
from .capture import AudioChunk, start_file_capture, start_live_capture
from .transcribe import Transcriber


def run(input_file: str | None, model_size: str) -> None:
    console = Console()
    chunks: queue.Queue[AudioChunk] = queue.Queue()

    console.print(f"[dim]Загружаю whisper ({model_size}, int8)...[/dim]")
    transcriber = Transcriber(model_size=model_size)
    answerer = Answerer()

    if input_file:
        stop = start_file_capture(chunks, input_file)
        console.print(f"[dim]Читаю {input_file}[/dim]")
    else:
        stop = start_live_capture(chunks)
        console.print("[dim]Слушаю встречу и микрофон. Ctrl+C для выхода.[/dim]")

    try:
        while True:
            try:
                chunk = chunks.get(timeout=5)
            except queue.Empty:
                if input_file:
                    break  # файл закончился
                continue
            utt = transcriber.feed(chunk)
            if utt is None:
                continue
            who = (
                "[bold red]интервьюер[/bold red]"
                if utt.source == "loopback"
                else "[green]я[/green]"
            )
            console.print(f"{who}: {utt.text}")
            answerer.add(utt)
            if answerer.is_question(utt):
                console.print("[bold cyan]ответ →[/bold cyan] ", end="")
                for delta in answerer.stream_answer(utt):
                    console.print(delta, end="")
                console.print()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interview helper — realtime суфлёр")
    parser.add_argument("--input-file", help="WAV-файл вместо живого захвата (отладка)")
    parser.add_argument("--model", default="small", help="размер модели whisper (base/small/...)")
    args = parser.parse_args()
    run(args.input_file, args.model)


if __name__ == "__main__":
    main()
