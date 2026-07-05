"""Stop-хук: если в рабочем дереве изменён код, но не доки — просим прогнать /docs-check."""

import json
import subprocess
import sys

data = json.load(sys.stdin)
if data.get("stop_hook_active"):  # уже сработали в этом цикле — не зацикливаемся
    sys.exit(0)

r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
files = [line[3:].strip() for line in r.stdout.splitlines()]
code_changed = any(f.startswith("src/") or f == "pyproject.toml" for f in files)
docs_changed = any(f in ("README.md", "CLAUDE.md") or f.startswith("docs/") for f in files)

if code_changed and not docs_changed:
    print(
        json.dumps(
            {
                "decision": "block",
                "reason": "Код изменён, документация — нет. Выполни скилл docs-check: "
                "сверь README.md, CLAUDE.md и docs/ с изменениями и поправь расхождения "
                "(или подтверди, что доки актуальны).",
            }
        )
    )
sys.exit(0)
