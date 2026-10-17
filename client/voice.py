"""omni-assistant by voice: you speak, the model hears the audio directly, your desktop does it, and it
answers out loud (Kokoro on the server).

    uv run python -m client.voice             # real actions
    uv run python -m client.voice --dry-run   # print the desktop commands instead of running them

Press Enter, speak, press Enter. Needs: uv run modal deploy -m apps.playground
"""
import argparse
import threading
from pathlib import Path

from client.audio_io import new_recording_path, play, record_until_enter, wav_to_pcm
from client.conversation import AUDIO, Respond, handle, next_history
from client.executor.registry import build_validators, confirm_tools, execute
from client.main import BOLD, DIM, RED, RESET, show
from client.speech import make_speaker
from src.utils.config import load_yaml

RECORDINGS = Path("data/recordings")


def connect_voice(server_cfg: dict, audio: list[bytes], spoken: dict) -> Respond:
    """respond(messages) for the playground server. `audio` holds this conversation's recordings in order;
    each reply's Kokoro audio is kept in `spoken` (reply text → (pcm, rate)) for the speaker."""
    import modal

    remote = modal.Cls.from_name(server_cfg["app_name"], server_cfg["class_name"])()

    def respond(messages: list[dict]) -> dict:
        result = remote.turn.remote("tools", messages, list(audio), True)
        spoken[result["reply"]] = (result["speech"], result["speech_rate"])
        return result

    return respond


def make_voice_speaker(spoken: dict, fallback):
    """Model replies play their Kokoro audio; anything else (a timer going off) uses the system voice."""
    def speak(text: str) -> None:
        if text in spoken:
            play(*spoken.pop(text))
        else:
            fallback(text)

    return speak


def warm_up(server_cfg: dict) -> None:
    """Start the GPU container while the user gets ready (a cold start takes ~1 min)."""
    def ping() -> None:
        import modal

        modal.Cls.from_name(server_cfg["app_name"], server_cfg["class_name"])().turn.remote("tools", [{"role": "user", "content": "hello"}], [], False)

    threading.Thread(target=ping, daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/client.yaml")
    parser.add_argument("--dry-run", action="store_true", help="print desktop commands instead of running them")
    opts = parser.parse_args()

    cfg = load_yaml(opts.config)
    tools_cfg = load_yaml(cfg["tools_config"])
    validators, confirm = build_validators(tools_cfg), confirm_tools(tools_cfg)
    audio: list[bytes] = []
    spoken: dict = {}
    speak = make_voice_speaker(spoken, make_speaker(cfg["speech"], True))
    env = {"cfg": cfg, "dry_run": opts.dry_run, "log": lambda text: print(f"  {DIM}{text}{RESET}"), "speak": speak}
    respond = connect_voice(cfg["voice_server"], audio, spoken)
    warm_up(cfg["voice_server"])

    print(f"{BOLD}omni-assistant · voice{RESET} {DIM}Enter = talk · q + Enter = quit"
          f"{' · DRY RUN' if opts.dry_run else ''} · first reply takes ~1 min (server waking up){RESET}")
    history: list[dict] = []
    while True:
        try:
            key = input(f"{BOLD}[Enter to talk]{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if key in {"q", "quit", "exit"}:
            break
        if not history:
            audio.clear()
        path = record_until_enter(new_recording_path(RECORDINGS), f"{BOLD}🎙  listening… press Enter when done{RESET}")
        audio.append(wav_to_pcm(path))
        try:
            messages, events = handle(history, AUDIO, respond, lambda call: execute(call, validators, env), confirm)
        except Exception as error:  # network / server trouble: report and keep the loop alive
            print(f"  {RED}server error: {error}{RESET}")
            history = []
            continue
        for event in events:
            show(event, speak)
        history = next_history(messages, cfg["keep_history_after_question"])


if __name__ == "__main__":
    main()
