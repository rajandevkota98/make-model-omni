"""set_timer: a background countdown that notifies, plays a sound and speaks when it ends.

Timers live in the client process, so they stop if you quit the client.
"""
import threading

from client.executor.shell import first_installed, fill, run


def ring(label: str, env: dict) -> None:
    tcfg = env["cfg"]["timer"]
    for candidates in (tcfg["notify"], tcfg["sound"]):
        command = first_installed(candidates)
        if command:
            run(fill(command, {"label": label}), env)
    env["log"](f"⏰ timer done: {label}")
    env["speak"](f"Your {label} timer is done." if label != "Timer" else "Your timer is done.")


def set_timer(args: dict, env: dict) -> dict:
    label = args.get("label") or "Timer"
    if env["dry_run"]:
        env["log"](f"[dry-run] timer {args['seconds']}s ({label})")
        return {"ok": True}
    timer = threading.Timer(args["seconds"], ring, args=(label, env))
    timer.daemon = True
    timer.start()
    return {"ok": True}
