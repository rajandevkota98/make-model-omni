"""Qwen3 loading: 4-bit NF4 base (QLoRA), LoRA adapters, tokenizer, and the merged inference weights."""
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


def load_lora(model, path: str, trainable: bool, gradient_checkpointing: bool = True):
    """Continue from a saved LoRA. Trainable → same QLoRA preparation as add_lora."""
    from peft import PeftModel, prepare_model_for_kbit_training

    if trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    return PeftModel.from_pretrained(model, path, is_trainable=trainable)


WEIGHTS = ("qlora", "merged_bf16", "merged_8bit", "merged_nf4_bf16", "merged_nf4_8bit")


def load_bf16(model_id: str, revision: str | None = None):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16, device_map={"": 0},
                                                attn_implementation="sdpa")


def merge_lora(mcfg: dict, lora_path: str, from_nf4: bool = False):
    """LoRA folded into a bf16 base: one plain model, no adapter layers left.
    from_nf4: start from the 4-bit base dequantized to bf16, i.e. the exact weights the LoRA was trained on
    (the original bf16 weights differ from them by the NF4 rounding error)."""
    base = load_base(mcfg).dequantize().to(torch.bfloat16) if from_nf4 else load_bf16(mcfg["id"], mcfg["revision"])
    return load_adapter(base, lora_path).merge_and_unload()


def load_8bit(path: str):
    """LLM.int8: linear weights stored in int8, outlier features kept in bf16; embeddings and lm_head stay bf16."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    return AutoModelForCausalLM.from_pretrained(path, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                                dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")


def load_inference(mcfg: dict, lora_path: str, weights: str):
    """qlora: 4-bit base + LoRA, as trained · merged_bf16: LoRA merged into the original bf16 base ·
    merged_8bit: that merged model saved, then reloaded in 8-bit (int8 weights can't be merged into) ·
    merged_nf4_*: the same, but merged into the dequantized 4-bit base the LoRA was trained on."""
    import tempfile

    if weights == "qlora":
        return load_adapter(load_base(mcfg), lora_path).eval()
    from_nf4 = weights.startswith("merged_nf4")
    if weights in ("merged_bf16", "merged_nf4_bf16"):
        return merge_lora(mcfg, lora_path, from_nf4).eval()
    if weights in ("merged_8bit", "merged_nf4_8bit"):
        with tempfile.TemporaryDirectory() as tmp:
            merged = merge_lora(mcfg, lora_path, from_nf4)
            merged.save_pretrained(tmp)
            del merged
            torch.cuda.empty_cache()
            return load_8bit(tmp).eval()
    raise ValueError(f"unknown weights {weights!r}, expected one of {WEIGHTS}")
