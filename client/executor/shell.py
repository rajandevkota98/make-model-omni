"""Run desktop commands: pick the first installed candidate, fill placeholders, never use a shell.

Commands are argv lists (no `sh -c`), so tool arguments like a typed sentence can't inject anything.
"""
import os
import shlex
import shutil
import subprocess

Command = list[str]


def fill(command: Command, values: dict) -> Command:
    return [os.path.expanduser(part.format(**values)) if "{" in part or part.startswith("~") else part for part in command]


def first_installed(candidates: list[Command]) -> Command | None:
    return next((c for c in candidates if shutil.which(c[0])), None)


def run(command: Command, env: dict, detach: bool = False) -> bool:
    """True when the command started (detach) or exited 0. Dry run only prints."""
    if env["dry_run"]:
        env["log"](f"[dry-run] {shlex.join(command)}")
        return True
    try:
        if detach:
            subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            return True
        return subprocess.run(command, capture_output=True, timeout=15).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def run_first(candidates: list[Command], values: dict, env: dict, missing: str, failed: str, detach: bool = False) -> dict:
    """Run the first installed candidate → a tool result: {"ok": true} or {"ok": false, "error": ...}."""
    command = first_installed(candidates)
    if command is None:
        return {"ok": False, "error": missing}
    return {"ok": True} if run(fill(command, values), env, detach) else {"ok": False, "error": failed}
