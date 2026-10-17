"""Try the model by hand: typed text, a wav file, or your microphone. Shows the reply and the tool calls
it WOULD make — nothing is executed. --talk also speaks the reply back (Kokoro, on the server).

    uv run python -m scripts.try_model --text "turn the volume down"
    uv run python -m scripts.try_model --audio data/speech/samples/test-62c484c994-bm_fable-c.wav
    uv run python -m scripts.try_model --record 4                # speak for 4 s
    uv run python -m scripts.try_model --record 4 --transcribe   # only: what did the ears hear?
    uv run python -m scripts.try_model --chat                    # type lines, or /rec 4, /hear 4, /new, /quit
    uv run python -m scripts.try_model --talk                    # Enter = start/stop talking; it answers aloud

Recordings are kept in data/recordings/ (a start on a self-recorded test set).
Needs: uv run modal deploy -m apps.playground
"""
import argparse
import json
from pathlib import Path

from client.audio_io import new_recording_path, play, record_seconds, record_until_enter, wav_to_pcm
from client.conversation import assistant, next_history, user
from src.utils.config import load_yaml

CFG = load_yaml("configs/playground.yaml")
RECORDINGS = Path("data/recordings")
BOLD, DIM, GREEN, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[0m"


def record(seconds: float) -> str:
    print(f"{BOLD}🎙  speak now ({seconds:g} s)…{RESET}", flush=True)
    path = record_seconds(new_recording_path(RECORDINGS), seconds)
    print(f"{DIM}   saved {path}{RESET}")
    return path


def record_until_done() -> str:
    return record_until_enter(new_recording_path(RECORDINGS), f"{BOLD}🎙  listening… press Enter when done{RESET}")


def connect():
    import modal

    remote = modal.Cls.from_name(CFG["app_name"], CFG["class_name"])()
    return lambda mode, messages, audio, speak=False: remote.turn.remote(mode, messages, audio, speak)


def show(result: dict, mode: str) -> None:
    if mode == "transcribe":
        print(f"{BOLD}heard>{RESET} {result['reply']}")
    else:
        print(f"{BOLD}assistant>{RESET} {result['reply'] or DIM + '(silent)' + RESET}")
        for call in result["tool_calls"]:
            print(f"  {GREEN}▸ {call['name']}{RESET}({json.dumps(call['arguments'])})")
    print(f"  {DIM}{result['ms']} ms · audio {result['audio_seconds']} s · {result['prompt_tokens']} prompt tokens{RESET}")


def as_message(result: dict) -> dict:
    return assistant(result["reply"], result["tool_calls"])


def chat(turn) -> None:
    print(f"{BOLD}playground{RESET} {DIM}type a command · /rec N speak N s · /hear N transcribe only · /new · /quit{RESET}")
    history, audio = [], []
    while True:
        try:
            line = input(f"{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return
        if line == "/new":
            history, audio = [], []
            continue
        if line.startswith("/hear"):
            seconds = float(line.split()[1]) if len(line.split()) > 1 else 4
            show(turn("transcribe", [], [wav_to_pcm(record(seconds))]), "transcribe")
            continue
        if line.startswith("/rec"):
            seconds = float(line.split()[1]) if len(line.split()) > 1 else 4
            audio = audio + [wav_to_pcm(record(seconds))]
            message = user("audio")
        else:
            message = user(line)
        result = turn("tools", [*history, message], audio)
        show(result, "tools")
        history = next_history([*history, message, as_message(result)], True)
        if not history:
            audio = []


def talk(turn) -> None:
    """Voice in, voice out. History is kept for follow-ups like "yes, do it"; /new-style reset = type n."""
    print(f"{BOLD}talk{RESET} {DIM}Enter = start talking · n + Enter = new conversation · q + Enter = quit{RESET}")
    history, audio = [], []
    while True:
        try:
            key = input(f"{BOLD}[Enter to talk]{RESET} ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if key == "q":
            return
        if key == "n":
            history, audio = [], []
            continue
        audio = audio + [wav_to_pcm(record_until_done())]
        message = user("audio")
        result = turn("tools", [*history, message], audio, True)
        show(result, "tools")
        print(f"  {DIM}voice {result['speech_ms']} ms for {result['speech_seconds']} s of speech{RESET}")
        play(result["speech"], result["speech_rate"])
        history = next_history([*history, message, as_message(result)], True)
        if not history:
            audio = []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text")
    parser.add_argument("--audio", help="path to a wav file")
    parser.add_argument("--record", type=float, metavar="SECONDS")
    parser.add_argument("--transcribe", action="store_true", help="show what the ears heard instead of acting")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--talk", action="store_true", help="speak, and hear the reply")
    opts = parser.parse_args()
    turn = connect()
    if opts.chat:
        chat(turn)
        return
    if opts.talk:
        talk(turn)
        return
    mode = "transcribe" if opts.transcribe else "tools"
    if opts.text:
        show(turn("tools", [user(opts.text)], []), "tools")
    elif opts.audio or opts.record:
        pcm = wav_to_pcm(opts.audio or record(opts.record))
        show(turn(mode, [user("audio")], [pcm]), mode)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
