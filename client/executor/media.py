"""media_control over MPRIS (the D-Bus interface Chrome, Firefox, Spotify, VLC... all expose).

Uses playerctl when installed, otherwise talks to the players directly with gdbus.
"""
import re
import shutil
import subprocess

from client.executor.shell import run

MPRIS = "org.mpris.MediaPlayer2."
METHODS = {"play": "Play", "pause": "Pause", "next": "Next", "previous": "Previous"}


def players() -> list[str]:
    try:
        out = subprocess.run(["gdbus", "call", "--session", "--dest", "org.freedesktop.DBus", "--object-path",
                              "/org/freedesktop/DBus", "--method", "org.freedesktop.DBus.ListNames"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    return re.findall(r"'(org\.mpris\.MediaPlayer2\.[^']+)'", out)


def mpris_command(player: str, action: str) -> list[str]:
    return ["gdbus", "call", "--session", "--dest", player, "--object-path", "/org/mpris/MediaPlayer2",
            "--method", f"org.mpris.MediaPlayer2.Player.{METHODS[action]}"]


def media_control(args: dict, env: dict) -> dict:
    action = args["action"]
    if shutil.which("playerctl"):
        command = ["playerctl", "--all-players", action] if action == "pause" else ["playerctl", action]
        return {"ok": True} if run(command, env) else {"ok": False, "error": "no_player"}
    found = players()
    if not found:
        return {"ok": False, "error": "no_player"}
    targets = found if action == "pause" else found[:1]   # pause everything; play / skip the first player
    return {"ok": True} if all([run(mpris_command(p, action), env) for p in targets]) else {"ok": False, "error": "no_player"}
