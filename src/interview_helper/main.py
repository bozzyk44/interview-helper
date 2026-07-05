"""Терминальная точка входа поверх общего пайплайна."""

from __future__ import annotations

import argparse
import threading

from rich.console import Console

from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Interview helper — realtime суфлёр")
    parser.add_argument("--input-file", help="WAV-файл вместо живого захвата (отладка)")
    parser.add_argument("--model", default="small", help="размер модели whisper (base/small/...)")
    parser.add_argument("--mic-device", type=int, help="индекс микрофона (см. --list-devices)")
    parser.add_argument("--loopback-device", type=int, help="индекс loopback-устройства")
    parser.add_argument("--list-devices", action="store_true", help="показать устройства и выйти")
    args = parser.parse_args()

    if args.list_devices:
        from .capture import list_devices

        for kind, devices in list_devices().items():
            print(f"{kind}:")
            for d in devices:
                mark = " (default)" if d["default"] else ""
                print(f"  [{d['index']}] {d['name']}{mark}")
        return

    console = Console()

    def emit(event: dict) -> None:
        match event["type"]:
            case "status":
                console.print(f"[dim]{event['text']}[/dim]")
            case "utterance":
                who = (
                    "[bold red]интервьюер[/bold red]"
                    if event["source"] == "loopback"
                    else "[green]я[/green]"
                )
                console.print(f"{who}: {event['text']}")
            case "answer_start":
                console.print("[bold cyan]ответ →[/bold cyan] ", end="")
            case "answer_delta":
                console.print(event["text"], end="")
            case "answer_end":
                console.print()

    stop = threading.Event()
    try:
        run_pipeline(
            emit,
            stop,
            input_file=args.input_file,
            model_size=args.model,
            mic_device=args.mic_device,
            loopback_device=args.loopback_device,
        )
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
