"""PreToolUse(Bash): живой захват звука (main.py без --input-file) требует подтверждения."""

import json
import sys

data = json.load(sys.stdin)
cmd = data.get("tool_input", {}).get("command", "")
if ("interview_helper.main" in cmd or "main.py" in cmd) and "--input-file" not in cmd:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "ask",
                    "permissionDecisionReason": "Команда запускает живой захват звука (микрофон + loopback).",
                }
            }
        )
    )
sys.exit(0)
