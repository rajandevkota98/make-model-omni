"""Volume, brightness, settings toggles, screenshots, typing and screen lock."""
import subprocess
import time
from pathlib import Path

from client.executor.shell import run_first


def set_volume(args: dict, env: dict) -> dict:
    vcfg = env["cfg"]["volume"]
    return run_first(vcfg[args["action"]], {"level": args.get("level", 0), "step": vcfg["step"]}, env, "no_audio_device", "no_audio_device")


def set_brightness(args: dict, env: dict) -> dict:
    bcfg = env["cfg"]["brightness"]
    return run_first(bcfg[args["action"]], {"level": args.get("level", 0), "step": bcfg["step"]}, env, "not_supported", "not_supported")


def toggle_setting(args: dict, env: dict) -> dict:
    scfg = env["cfg"]["settings"][args["setting"]]
    value = scfg.get("values", {}).get(args["state"], args["state"])
    return run_first(scfg["commands"], {"value": value}, env, "not_supported", "permission_denied")


def active_window() -> str:
    """X11 id of the focused window, e.g. '0x3a00007' (falls back to the whole screen)."""
    try:
        out = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"], capture_output=True, text=True, timeout=5).stdout
        return out.strip().split()[-1]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "root"


def screenshot_path(directory: str) -> Path:
    folder = Path(directory).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"omni-{time.strftime('%Y%m%d-%H%M%S')}.png"


def take_screenshot(args: dict, env: dict) -> dict:
    scfg = env["cfg"]["screenshot"]
    area = args["area"]
    values = {"path": str(screenshot_path(scfg["dir"])), "window": active_window() if area == "window" else "root"}
    result = run_first(scfg[area], values, env, "failed", "failed")
    if result["ok"]:
        env["log"](f"screenshot saved to {values['path']}")
    return result


def type_text(args: dict, env: dict) -> dict:
    result = run_first(env["cfg"]["type_text"], {"text": args["text"]}, env, "not_installed", "failed")
    if result.get("error") == "not_installed":
        env["log"]("typing needs xdotool: sudo apt install xdotool")
    return result


def lock_screen(args: dict, env: dict) -> dict:
    return run_first(env["cfg"]["lock_screen"], {}, env, "not_supported", "failed")
