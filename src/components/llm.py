"""Qwen3 loading: 4-bit NF4 base (QLoRA), LoRA adapters, tokenizer."""
import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def quant_config(qcfg: dict):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(load_in_4bit=qcfg["load_in_4bit"], bnb_4bit_quant_type=qcfg["quant_type"],
                              bnb_4bit_use_double_quant=qcfg["double_quant"], bnb_4bit_compute_dtype=DTYPES[qcfg["compute_dtype"]])


def load_tokenizer(mcfg: dict, padding_side: str = "right"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(mcfg["id"], revision=mcfg["revision"], padding_side=padding_side)


def load_base(mcfg: dict):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        mcfg["id"], revision=mcfg["revision"], quantization_config=quant_config(mcfg["quant"]),
        dtype=DTYPES[mcfg["quant"]["compute_dtype"]], device_map={"": 0}, attn_implementation="sdpa")


def add_lora(model, lcfg: dict, gradient_checkpointing: bool):
    """QLoRA recipe: stabilise the 4-bit model for training, then wrap the linear layers with LoRA."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
                        target_modules=lcfg["target_modules"], bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, config)


def load_adapter(model, path: str):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, path, is_trainable=False)


def decoder_and_head(model):
    """(transformer body, lm_head) — LoRA layers live inside the body, so calling it directly keeps them active."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.lm_head


def param_counts(model) -> dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": sum(p.numel() for p in model.parameters())}
