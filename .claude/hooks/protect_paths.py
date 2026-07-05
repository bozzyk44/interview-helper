"""PreToolUse: блокирует правки персональных данных (context/, sessions/, profile.md)."""

import json
import sys
from pathlib import PurePath

data = json.load(sys.stdin)
path = data.get("tool_input", {}).get("file_path", "")
if not path:
    sys.exit(0)
parts = PurePath(path.replace("\\", "/")).parts
if "context" in parts or "sessions" in parts or PurePath(path).name == "profile.md":
    print(
        "Файлы в context/, sessions/ и profile.md содержат персональные данные "
        "и управляются runtime-кодом. Не редактируй их напрямую.",
        file=sys.stderr,
    )
    sys.exit(2)
