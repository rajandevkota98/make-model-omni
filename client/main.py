"""omni-assistant POC: type a command, the model on Modal decides, your desktop does it.

    uv run python -m client.main             # real actions
    uv run python -m client.main --dry-run   # print the commands instead of running them
    uv run python -m client.main --no-speak  # text only

Needs the server deployed first: uv run modal deploy -m apps.serve
"""
import argparse
import json
import threading

from client.conversation import handle, next_history
from client.executor.registry import build_validators, confirm_tools, execute
from client.speech import make_speaker
from client.transport import connect
from src.utils.config import load_yaml

DIM, BOLD, GREEN, RED, YELLOW, RESET = "\033[2m", "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def show_call(call: dict, result: dict) -> str:
    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in call["arguments"].items())
    status = f"{GREEN}ok{RESET}" if result.get("ok") else f"{RED}{result.get('error')}{RESET}"
    extra = {k: v for k, v in result.items() if k != "ok"} if result.get("ok") and len(result) > 1 else ""
    return f"  {DIM}▸{RESET} {call['name']}({args}) → {status} {DIM}{extra or ''}{RESET}"


def show(event: dict, speak) -> None:
    if "say" in event:
        print(f"{BOLD}assistant>{RESET} {event['say']}")
        speak(event["say"])
    elif "call" in event:
        print(show_call(event["call"], event["result"]))
    elif "blocked" in event:
        print(f"  {YELLOW}✋ blocked {event['blocked']['name']}: it needs a spoken yes first{RESET}")
    elif "ignored" in event:
        print(f"  {YELLOW}ignored a follow-up call to {event['ignored']['name']}{RESET}")
    elif "model" in event:
        print(f"  {DIM}model {event['model']['ms']} ms · {event['model']['prompt_tokens']} prompt tokens{RESET}")


def warm_up(respond) -> None:
    """Start the GPU container while the user types (a cold start takes ~30-60 s)."""
    threading.Thread(target=lambda: respond([{"role": "user", "content": "hello"}]), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/client.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print desktop commands instead of running them")
    parser.add_argument("--no-speak", action="store_true", help="don't speak replies")
    opts = parser.parse_args()

    cfg = load_yaml(opts.config)
    tools_cfg = load_yaml(cfg["tools_config"])
    validators, confirm = build_validators(tools_cfg), confirm_tools(tools_cfg)
    speak = make_speaker(cfg["speech"], not opts.no_speak)
    env = {"cfg": cfg, "dry_run": opts.dry_run, "log": lambda text: print(f"  {DIM}{text}{RESET}"), "speak": speak}
    respond = connect(cfg["server"])
    warm_up(respond)

    print(f"{BOLD}omni-assistant{RESET} {DIM}(type a command · 'quit' to exit{' · DRY RUN' if opts.dry_run else ''}){RESET}")
    history: list[dict] = []
    while True:
        try:
            text = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "bye"}:
            break
        if not text:
            continue
        try:
            messages, events = handle(history, text, respond, lambda call: execute(call, validators, env), confirm)
        except Exception as error:  # network / server trouble: report and keep the REPL alive
            print(f"  {RED}server error: {error}{RESET}")
            history = []
            continue
        for event in events:
            show(event, speak)
        history = next_history(messages, cfg["keep_history_after_question"])


if __name__ == "__main__":
    main()
