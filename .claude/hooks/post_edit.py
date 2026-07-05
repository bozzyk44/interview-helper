"""PostToolUse: ruff format + lint --fix для отредактированных .py файлов."""

import json
import subprocess
import sys

data = json.load(sys.stdin)
path = data.get("tool_input", {}).get("file_path", "")
if path.endswith(".py"):
    subprocess.run(["ruff", "format", path], capture_output=True)
    r = subprocess.run(["ruff", "check", "--fix", path], capture_output=True, text=True)
    if r.returncode != 0:
        # неисправимые замечания показываем Claude
        print(r.stdout + r.stderr, file=sys.stderr)
        sys.exit(2)
