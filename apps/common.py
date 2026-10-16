"""Shared Modal pieces: the app, the container image, the Volume and secrets."""
import modal

APP_NAME = "make-model-omni"
VOLUME_PATH = "/vol"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("omni-lab", create_if_missing=True)
secrets = [modal.Secret.from_name("comet")]

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.14.0", "transformers==5.19.0", "peft==0.21.0", "bitsandbytes==0.50.2",
        "accelerate==1.15.0", "comet-ml==3.58.8", "pyyaml==6.0.3", "jsonschema==4.26.0",
    )
    .env({"HF_HOME": f"{VOLUME_PATH}/hf", "TOKENIZERS_PARALLELISM": "false"})
    .workdir("/root")
    .add_local_dir("configs", "/root/configs")
    .add_local_python_source("src", "apps")
)
