"""Training and evaluation runs: wiring of model, data, loop, checkpoints and tracking.

Runs inside a Modal GPU container; everything it needs comes from the merged config dict.
"""
import json
import math
import random
import time
from pathlib import Path

from src.utils.io import read_jsonl, write_json
from src.utils.logging import get_logger
from src.utils.tracking import end_run, log_metrics, log_table, start_run

log = get_logger("runs")


# ── data ────────────────────────────────────────────────────────────────────

def load_split(cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    examples = read_jsonl(Path(cfg["data"]["dir"]) / f"{split}.jsonl")
    return examples[:limit] if limit else examples


def make_collate(cfg: dict, tokenizer):
    from src.components.batching import collate

    d = cfg["data"]
    return lambda examples: collate(examples, tokenizer.pad_token_id, d["tool_call_open_id"], d["tool_call_close_id"])


def dev_batches(examples: list[dict], batch_size: int, collate_fn) -> list[tuple[list[str], dict]]:
    chunks = [examples[i:i + batch_size] for i in range(0, len(examples), batch_size)]
    return [([e["kind"] for e in chunk], collate_fn(chunk)) for chunk in chunks]


def steps_per_epoch(cfg: dict, n_train: int) -> int:
    t = cfg["train"]
    return math.ceil(math.ceil(n_train / t["micro_batch"]) / t["grad_accum"])


def total_steps_for(cfg: dict, n_train: int) -> int:
    return cfg["train"]["max_steps"] or steps_per_epoch(cfg, n_train) * cfg["train"]["epochs"]


# ── checkpoints ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, path: Path, state: dict, with_optimizer: bool) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if with_optimizer:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
    write_json(path / "state.json", state)


def try_resume(model, optimizer, path: Path) -> dict | None:
    if not (path / "state.json").exists():
        return None
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    set_peft_model_state_dict(model, load_file(path / "adapter_model.safetensors"))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cuda"))
    return json.loads((path / "state.json").read_text())


# ── training ────────────────────────────────────────────────────────────────

def run_training(cfg: dict, run_id: str, tracking_cfg: dict, commit=lambda: None) -> dict:
    import torch

    from src.components.batching import epoch_batches
    from src.components.llm import add_lora, load_base, load_tokenizer, param_counts
    from src.components.train_step import evaluate_loss, train_step

    t, o = cfg["train"], cfg["optim"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    tokenizer = load_tokenizer(cfg["model"])
    model = add_lora(load_base(cfg["model"]), cfg["lora"], cfg["model"]["gradient_checkpointing"])
    model.train()
    counts = param_counts(model)
    log.info("params: %s trainable of %s", f"{counts['trainable']:,}", f"{counts['total']:,}")

    train = load_split(cfg, "train", t["limit_train"])
    dev = train if t["dev_is_train"] else load_split(cfg, "dev")
    collate_fn = make_collate(cfg, tokenizer)
    dev_set = dev_batches(dev, cfg["eval"]["batch_size"], collate_fn)
    total_steps = total_steps_for(cfg, len(train))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
    run_dir = Path(cfg["paths"]["ckpt_dir"]) / run_id
    resumed = try_resume(model, optimizer, run_dir / "last") or {}
    step, best = resumed.get("step", 0), resumed.get("best_dev_loss", float("inf"))
    if resumed:
        log.info("resumed from step %d", step)

    run = start_run(tracking_cfg, run_id, {**cfg, "params": counts, "n_train": len(train), "n_dev": len(dev), "total_steps": total_steps},
                    Path(cfg["paths"]["logs_dir"]) / f"{run_id}.jsonl")
    log.info("train=%d dev=%d steps=%d", len(train), len(dev), total_steps)
    started, tokens_seen = time.time(), 0
    per_epoch = steps_per_epoch(cfg, len(train))
    epoch = step // per_epoch
    while step < total_steps:
        batches = epoch_batches(len(train), t["micro_batch"], t["seed"], epoch)
        groups = [batches[i:i + t["grad_accum"]] for i in range(0, len(batches), t["grad_accum"])]
        skip = step - epoch * per_epoch  # resume mid-epoch
        for group in groups[max(0, skip):]:
            if step >= total_steps:
                break
            micro = [collate_fn([train[i] for i in idx]) for idx in group]
            tick = time.time()
            metrics = train_step(model, optimizer, micro, step, total_steps, o)
            step += 1
            tokens = sum(int(b["attention_mask"].sum()) for b in micro)
            tokens_seen += tokens
            if step % tracking_cfg["log_every_steps"] == 0 or step == 1 or step == total_steps:
                metrics.update({"epoch": step / len(groups), "tokens_per_sec": tokens / (time.time() - tick),
                                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                log_metrics(run, metrics, step, "train/")
                log.info("step %d/%d loss %.4f tool %.4f reply %.4f lr %.2e", step, total_steps, metrics["loss"],
                         metrics["loss_tool_call"], metrics["loss_reply"], metrics["lr"])
            if step % t["eval_every"] == 0 or (step == total_steps and t.get("final_eval", True)):
                dev_metrics = evaluate_loss(model, dev_set)
                log_metrics(run, dev_metrics, step, "dev/")
                log.info("dev loss %.4f (best %.4f)", dev_metrics["loss"], best)
                if dev_metrics["loss"] < best:
                    best = dev_metrics["loss"]
                    save_checkpoint(model, optimizer, run_dir / "best", {"step": step, "dev_loss": best}, with_optimizer=False)
                    commit()
            if step % t["save_every"] == 0:
                save_checkpoint(model, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, with_optimizer=True)
                commit()
        epoch += 1
    save_checkpoint(model, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, with_optimizer=True)
    commit()
    summary = {"run_id": run_id, "steps": step, "best_dev_loss": best, "hours": round((time.time() - started) / 3600, 3),
               "tokens_seen": tokens_seen, "params": counts, "best_adapter": str(run_dir / "best")}
    write_json(run_dir / "summary.json", summary)
    end_run(run)
    return summary


# ── evaluation ──────────────────────────────────────────────────────────────

def run_evaluation(cfg: dict, adapter: str | None, split: str, name: str, tracking_cfg: dict, limit: int | None = None,
                   weights: str = "qlora") -> dict:
    """weights: qlora (4-bit base + LoRA), merged_bf16 or merged_8bit, see components/llm.py:load_inference."""
    from src.components.generate import generate_batch
    from src.components.llm import load_base, load_tokenizer
    from src.services.assistant import load_weights
    from src.services.data.validate import build_context
    from src.utils.config import load_yaml
    from src.utils.metrics import aggregate, gate, score_turn

    tokenizer = load_tokenizer(cfg["model"], padding_side="left")
    model = load_weights(cfg["model"], adapter, weights) if adapter else load_base(cfg["model"]).eval()
    data_cfg = load_yaml("configs/data.yaml")
    ctx = build_context(load_yaml(data_cfg["tools_config"]), data_cfg["validate"])
    examples = load_split(cfg, split, limit)
    e = cfg["eval"]
    scores, samples, started = [], [], time.time()
    for i in range(0, len(examples), e["batch_size"]):
        chunk = examples[i:i + e["batch_size"]]
        prompts = [ex["input_ids"][:ex["prompt_len"]] for ex in chunk]
        outputs = generate_batch(model, tokenizer, prompts, e["max_new_tokens"], cfg["data"]["stop_ids"])
        for ex, out in zip(chunk, outputs):
            expected = tokenizer.decode(ex["input_ids"][ex["prompt_len"]:], skip_special_tokens=False)
            prompt_text = tokenizer.decode(ex["input_ids"][:ex["prompt_len"]], skip_special_tokens=False)
            score = score_turn(out, expected, prompt_text, ex["kind"], ctx["schemas"], ctx["confirm_tools"])
            scores.append({**score, "id": ex["id"]})
            samples.append({"id": ex["id"], "kind": ex["kind"], "user": prompt_text.rsplit("<|im_start|>user\n", 1)[-1][:200],
                            "expected": expected.strip(), "predicted": out.strip(), "exact": score["exact"]})
        log.info("eval %d/%d", min(i + e["batch_size"], len(examples)), len(examples))
    summary = {**aggregate(scores), "name": name, "adapter": adapter, "weights": weights, "split": split, "minutes": round((time.time() - started) / 60, 2)}
    summary["gate"] = gate(summary)
    out_dir = Path(cfg["paths"]["evals_dir"])
    write_json(out_dir / f"{name}.json", summary)
    write_json(out_dir / f"{name}.samples.json", samples)
    run = start_run(tracking_cfg, f"eval-{name}", {"adapter": adapter or "none", "weights": weights, "split": split, "name": name})
    log_metrics(run, {k: v for k, v in summary.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}, 0, f"eval/{split}/")
    rng = random.Random(0)
    log_table(run, f"samples-{name}", rng.sample(samples, min(e["samples_to_log"], len(samples))))
    log_table(run, f"by-kind-{name}", [{"kind": k, **v} for k, v in summary["by_kind"].items()])
    end_run(run)
    return summary
