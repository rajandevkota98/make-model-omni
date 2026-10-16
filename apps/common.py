"""Shared Modal pieces: the app, the container image, the Volume and secrets."""
import modal

APP_NAME = "make-model-omni"
VOLUME_PATH = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("omni-lab", create_if_missing=True)
secrets = [modal.Secret.from_name("comet")]

train_packages = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.19.0", "peft==0.21.0", "bitsandbytes==0.50.2",
        "accelerate==1.15.0", "comet-ml==3.58.8", "pyyaml==6.0.3", "jsonschema==4.26.0",
    )
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "TOKENIZERS_PARALLELISM": "false"})
)


def with_code(image: modal.Image) -> modal.Image:
    """Project code and configs go last so code edits don't rebuild the package layers."""
    return image.workdir("/root").add_local_dir("configs", "/root/configs").add_local_python_source("src", "apps")


train_image = with_code(train_packages)
# Step 5: same stack + WER scoring
audio_train_image = with_code(train_packages.uv_pip_install("jiwer==4.0.0"))

KOKORO_REPO = "hexgrad/Kokoro-82M"
SPACY_EN_SM = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Step 4: Kokoro TTS + Whisper round-trip QA. Kokoro weights and voices are baked into the image so
# parallel containers never race on the shared HF cache.
tts_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("espeak-ng")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.19.0", "kokoro==0.9.4", "misaki[en]==0.9.4", "spacy==3.8.16",
        f"en_core_web_sm @ {SPACY_EN_SM}", "numpy==2.5.4", "scipy==1.18.1", "jiwer==4.0.0", "pyyaml==6.0.3",
    )
    .env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false"})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; "
                  f"snapshot_download('{KOKORO_REPO}', allow_patterns=['*.json', '*.pth', 'voices/*.pt'])\"")
    .workdir("/root")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src", "apps")
)
