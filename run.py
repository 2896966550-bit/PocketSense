#!/usr/bin/env python3
"""任务入口（Windows 无 make 时的主通道）。用法：python run.py <task>"""
import subprocess
import sys

TASKS = {
    "test": [sys.executable, "-m", "pytest", "tests", "-q"],
    "smoke": [sys.executable, "-m", "psense.pipeline.smoke"],
    "data-fetch": [sys.executable, "-m", "psense.data.fetch"],
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print("用法: python run.py {%s}" % "|".join(sorted(TASKS)))
        return 2
    return subprocess.call(TASKS[sys.argv[1]])


if __name__ == "__main__":
    raise SystemExit(main())
