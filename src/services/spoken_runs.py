"""Step 6 runs: train LoRA + audio adapter on spoken commands, evaluate tool calls from audio.

Runs inside a Modal GPU container; everything comes from the merged config dict.
"""
import random
import time
from pathlib import Path

from src.utils.config import deep_merge, load_yaml
from src.utils.io import read_jsonl, write_json
from src.utils.logging import get_logger
from src.utils.tracking import end_run, log_metrics, log_table, start_run

log = get_logger("spoken")


def full_config(cfg: dict) -> dict:
    """Step 6 config + the pieces it borrows from steps 3 and 5."""
    text, audio = load_yaml(cfg["text_config"]), load_yaml(cfg["audio_config"])
    return deep_merge({"model": text["model"], "lora": text["lora"], "tokens": text["data"], "encoder": audio["encoder"],
                       "adapter": audio["adapter"], "prompt": audio["prompt"]}, cfg)


# ── setup ───────────────────────────────────────────────────────────────────

def load_parts(cfg: dict, lora_path: str, adapter_path: str, train: bool) -> dict:
    import torch

    from src.components.adapters.registry import build_adapter, load_adapter_weights
    from src.components.encoders import load_whisper
    from src.components.llm import load_base, load_lora, load_tokenizer
    from src.components.tokens import add_placeholder
    from src.services.audio_data import prompt_parts

    tokenizer = load_tokenizer(cfg["model"], padding_side="left")
    base = load_base(cfg["model"])
    n_rows = base.get_input_embeddings().weight.shape[0]
    target_norm = float(base.get_input_embeddings().weight.float().norm(dim=-1).mean())
    model = load_lora(base, lora_path, trainable=train)
    model.train(train)
    placeholder_id = add_placeholder(tokenizer, cfg["prompt"]["placeholder"], n_rows)
    adapter = build_adapter(cfg["adapter"], cfg["encoder"]["dim"], target_norm).to("cuda", torch.float32)
    load_adapter_weights(adapter, Path(adapter_path) / "adapter.safetensors")
    adapter.train(train)
    return {"model": model, "adapter": adapter, "tokenizer": tokenizer, "whisper": load_whisper(cfg["encoder"]),
            "placeholder_id": placeholder_id, "asr_prompt": prompt_parts(tokenizer, cfg["prompt"], placeholder_id),
            "stop_ids": cfg["tokens"]["stop_ids"]}


def load_split(cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    rows = read_jsonl(Path(cfg["data"]["spoken_dir"]) / f"{split}.jsonl")
    return rows[:limit] if limit else rows


def item_fn(cfg: dict, parts: dict, pools: dict, lists: dict, epoch: int):
    """(source, index) → one training item."""
    from src.services.spoken_data import asr_item, spoken_item, text_item

    ecfg, k, pid = cfg["encoder"], cfg["adapter"]["frame_stack"], parts["placeholder_id"]

    def make(source: str, index: int) -> dict:
        if source == "spoken":
            return spoken_item(lists["spoken"][index], pools["train"], f"{cfg['train']['seed']}:{epoch}", pid, ecfg, k)
        if source == "text":
            return text_item(lists["text"][index])
        return asr_item(lists["asr"][index], pools["asr"], parts["asr_prompt"], parts["tokenizer"], ecfg, k)

    return make


def replay_lists(cfg: dict, n_spoken: int, epoch: int, pools: dict) -> dict:
    """Fresh random replay picks each epoch."""
    from src.services.spoken_data import replay_count

    d, rng = cfg["data"], random.Random(f"{cfg['train']['seed']}:replay:{epoch}")
    text_all = pools.get("text_examples", [])
    asr_all = pools["asr"]["rows"] if "asr" in pools else []
    n_text, n_asr = replay_count(n_spoken, d["text_replay"]["fraction"]), replay_count(n_spoken, d["asr_replay"]["fraction"])
    return {"text": rng.sample(text_all, min(n_text, len(text_all))), "asr": rng.sample(asr_all, min(n_asr, len(asr_all)))}


def collate(items: list[dict], cfg: dict, parts: dict) -> dict:
    from src.services.spoken_data import collate_items

    t = cfg["tokens"]
    return collate_items(items, parts["tokenizer"].pad_token_id, t["tool_call_open_id"], t["tool_call_close_id"])


# ── checkpoints ─────────────────────────────────────────────────────────────

def save_checkpoint(parts: dict, optimizer, path: Path, state: dict, cfg: dict, with_optimizer: bool) -> None:
    import torch

    from src.components.adapters.registry import save_adapter

    path.mkdir(parents=True, exist_ok=True)
    parts["model"].save_pretrained(path / "lora")
    save_adapter(parts["adapter"], path / "adapter.safetensors")
    write_json(path / "adapter_config.json", {"adapter": cfg["adapter"], "encoder": cfg["encoder"], "prompt": cfg["prompt"]})
    if with_optimizer:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
    write_json(path / "state.json", state)


def try_resume(parts: dict, optimizer, path: Path) -> dict:
    if not (path / "state.json").exists():
        return {}
    import json

    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    from src.components.adapters.registry import load_adapter_weights

    set_peft_model_state_dict(parts["model"], load_file(path / "lora" / "adapter_model.safetensors"))
    load_adapter_weights(parts["adapter"], path / "adapter.safetensors")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cuda"))
    return json.loads((path / "state.json").read_text())


# ── training ────────────────────────────────────────────────────────────────

def run_training(cfg: dict, run_id: str, tracking_cfg: dict, commit=lambda: None) -> dict:
    import torch

    from src.components.thinker_step import evaluate_loss, train_step
    from src.services.spoken_data import load_pool, plan_epoch, spoken_item

    cfg = full_config(cfg)
    t, o, d = cfg["train"], cfg["optim"], cfg["data"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    parts = load_parts(cfg, cfg["init"]["lora"], cfg["init"]["adapter"], train=True)
    spoken = load_split(cfg, "train", t["limit_train"])
    dev = spoken if t["dev_is_train"] else load_split(cfg, "dev")
    started = time.time()
    pools = {"train": load_pool(d["speech_audio_dir"], ["train"]), "dev": load_pool(d["speech_audio_dir"], ["dev"])}
    if t["dev_is_train"]:
        pools["dev"] = pools["train"]
    if d["text_replay"]["fraction"]:
        pools["text_examples"] = read_jsonl(Path(d["text_replay"]["dir"]) / "train.jsonl")
    if d["asr_replay"]["fraction"]:
        pools["asr"] = load_pool(d["asr_replay"]["audio_dir"], d["asr_replay"]["splits"], d["asr_replay"]["max_shards"])
    log.info("clips loaded in %.0f s", time.time() - started)

    ecfg, k, pid = cfg["encoder"], cfg["adapter"]["frame_stack"], parts["placeholder_id"]
    dev_items = [spoken_item(e, pools["dev"], "dev", pid, ecfg, k) for e in dev]
    dev_text = [] if t["dev_is_train"] else [{**e, "waveforms": [], "source": "text"} for e in read_jsonl(Path(d["text_replay"]["dir"]) / "dev.jsonl")]
    bs = cfg["eval"]["batch_size"]
    dev_batches = [collate(chunk, cfg, parts) for items in (dev_items, dev_text) for chunk in (items[i:i + bs] for i in range(0, len(items), bs))]

    counts = {"spoken": len(spoken), **{k_: len(v) for k_, v in replay_lists(cfg, len(spoken), 0, pools).items()}}
    per_epoch = -(-len(plan_epoch(counts, t["micro_batch"], t["seed"], 0)) // t["grad_accum"])
    total_steps = t["max_steps"] or per_epoch * t["epochs"]
    lora_params = [p for p in parts["model"].parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": lora_params, "peak_lr": o["lr_lora"], "lr": o["lr_lora"]},
                                   {"params": list(parts["adapter"].parameters()), "peak_lr": o["lr_adapter"], "lr": o["lr_adapter"]}],
                                  betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
    run_dir = Path(cfg["paths"]["ckpt_dir"]) / run_id
    resumed = try_resume(parts, optimizer, run_dir / "last")
    step, best = resumed.get("step", 0), resumed.get("best_dev_loss", float("inf"))
    params = {"lora": sum(p.numel() for p in lora_params), "adapter": sum(p.numel() for p in parts["adapter"].parameters())}
    run = start_run(tracking_cfg, run_id, {**cfg, "params": params, "counts": counts, "total_steps": total_steps},
                    Path(cfg["paths"]["logs_dir"]) / f"{run_id}.jsonl")
    log.info("params %s · per epoch %s · steps %d%s", params, counts, total_steps, f" · resumed at {step}" if step else "")
    started = time.time()
    while step < total_steps:
        epoch = step // per_epoch
        lists = {"spoken": spoken, **replay_lists(cfg, len(spoken), epoch, pools)}
        make = item_fn(cfg, parts, pools, lists, epoch)
        plan = plan_epoch({s: len(v) for s, v in lists.items()}, t["micro_batch"], t["seed"], epoch)
        groups = [plan[i:i + t["grad_accum"]] for i in range(0, len(plan), t["grad_accum"])]
        for group in groups[step - epoch * per_epoch:]:
            if step >= total_steps:
                break
            micro = [collate([make(source, i) for i in idx], cfg, parts) for source, idx in group]
            tick = time.time()
            metrics = train_step(parts["model"], parts["adapter"], parts["whisper"], optimizer, micro, step, total_steps, pid, cfg)
            step += 1
            if step % tracking_cfg["log_every_steps"] == 0 or step in (1, total_steps):
                metrics.update({"epoch": step / per_epoch, "sec_per_step": time.time() - tick,
                                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                log_metrics(run, metrics, step, "train/")
                log.info("step %d/%d loss %.4f tool %.4f reply %.4f · %.1f s/step", step, total_steps, metrics["loss"],
                         metrics["loss_tool_call"], metrics["loss_reply"], metrics["sec_per_step"])
            if step % t["eval_every"] == 0 or (step == total_steps and t.get("final_eval", True)):
                dev_metrics = evaluate_loss(parts["model"], parts["adapter"], parts["whisper"], dev_batches, pid, cfg)
                log_metrics(run, dev_metrics, step, "dev/")
                log.info("dev %s (best %.4f)", {k_: round(v, 4) for k_, v in dev_metrics.items() if isinstance(v, float)}, best)
                if dev_metrics["loss"] < best:
                    best = dev_metrics["loss"]
                    save_checkpoint(parts, optimizer, run_dir / "best", {"step": step, "dev_loss": best}, cfg, False)
                    commit()
            if step % t["save_every"] == 0:
                save_checkpoint(parts, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, cfg, True)
                commit()
    save_checkpoint(parts, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, cfg, True)
    if best == float("inf"):
        save_checkpoint(parts, optimizer, run_dir / "best", {"step": step, "dev_loss": None}, cfg, False)
    commit()
    summary = {"run_id": run_id, "steps": step, "best_dev_loss": best, "hours": round((time.time() - started) / 3600, 3),
               "params": params, "best": str(run_dir / "best")}
    write_json(run_dir / "summary.json", summary)
    end_run(run)
    return summary


# ── evaluation ──────────────────────────────────────────────────────────────

def generate_items(parts: dict, items: list[dict], cfg: dict) -> list[str]:
    import torch

    from src.components.generate import generate_from_embeds
    from src.components.thinker_step import batch_embeds

    tok, out, bs = parts["tokenizer"], [], cfg["eval"]["batch_size"]
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        prompts = [it["input_ids"][: it["prompt_len"]] for it in chunk]
        width = max(map(len, prompts))
        batch = {"input_ids": torch.tensor([[tok.pad_token_id] * (width - len(p)) + p for p in prompts]),
                 "waveforms": [w for it in chunk for w in it["waveforms"]]}
        mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device="cuda")
        with torch.no_grad():
            embeds = batch_embeds(parts["model"], parts["adapter"], parts["whisper"], batch, parts["placeholder_id"], cfg)
        out += generate_from_embeds(parts["model"], tok, embeds, mask, cfg["eval"]["max_new_tokens"], parts["stop_ids"])
        log.info("eval %d/%d", min(i + bs, len(items)), len(items))
    return out


def prompt_text(example: dict, tokenizer) -> str:
    """Prompt as text, with each audio slot shown as what was said (for scoring and samples)."""
    pieces = [tokenizer.decode(example["segments"][0])]
    for said, segment in zip(example["audio_texts"], example["segments"][1:]):
        pieces += [said, tokenizer.decode(segment)]
    return "".join(pieces)


def run_evaluation(cfg: dict, lora_path: str, adapter_path: str, split: str, name: str, tracking_cfg: dict,
                   limit: int | None = None) -> dict:
    """lora_path / adapter_path: a step 6 checkpoint (<ckpt>/lora, <ckpt>) or the untouched step 3 + step 5 inits."""
    from src.services.data.validate import build_context
    from src.services.spoken_data import load_pool, spoken_item
    from src.utils.metrics import aggregate, gate, score_turn

    cfg = full_config(cfg)
    parts = load_parts(cfg, lora_path, adapter_path, train=False)
    tok, pid, ecfg, k = parts["tokenizer"], parts["placeholder_id"], cfg["encoder"], cfg["adapter"]["frame_stack"]
    examples = load_split(cfg, "train", cfg["train"]["limit_train"]) if cfg["train"]["dev_is_train"] else load_split(cfg, split, limit)
    pool = load_pool(cfg["data"]["speech_audio_dir"], ["train" if cfg["train"]["dev_is_train"] else split])
    data_cfg = load_yaml("configs/data.yaml")
    ctx = build_context(load_yaml(data_cfg["tools_config"]), data_cfg["validate"])
    summary, all_samples = {"name": name, "lora": lora_path, "adapter": adapter_path, "split": split}, []
    for variant in cfg["eval"]["variants"]:
        items = [spoken_item(e, pool, "eval", pid, ecfg, k, variant) for e in examples]
        outputs = generate_items(parts, items, cfg)
        scores = []
        for e, out in zip(examples, outputs):
            expected, prompt = tok.decode(e["target"]), prompt_text(e, tok)
            scores.append({**score_turn(out, expected, prompt, e["kind"], ctx["schemas"], ctx["confirm_tools"]), "id": e["id"]})
            all_samples.append({"variant": variant, "id": e["id"], "kind": e["kind"], "said": " | ".join(e["audio_texts"]),
                                "expected": expected.strip(), "predicted": out.strip(), "exact": scores[-1]["exact"]})
        result = aggregate(scores)
        result["gate"] = {**gate(result), "within_5_of_text": result["tool_exact"] is not None
                          and result["tool_exact"] >= cfg["eval"]["text_tool_exact"] - 0.05}
        summary[variant] = result
    out_dir = Path(cfg["paths"]["evals_dir"])
    write_json(out_dir / f"{name}.json", summary)
    write_json(out_dir / f"{name}.samples.json", all_samples)
    run = start_run(tracking_cfg, f"eval-{name}", {"lora": lora_path, "adapter": adapter_path, "split": split})
    for variant in cfg["eval"]["variants"]:
        log_metrics(run, {k_: v for k_, v in summary[variant].items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, 0, f"eval/{variant}/")
    log_table(run, f"samples-{name}", random.Random(0).sample(all_samples, min(cfg["eval"]["samples_to_log"], len(all_samples))))
    end_run(run)
    return summary
