"""Stage 1b runs: train the audio adapter, evaluate transcription (WER), probe its outputs.

Runs inside a Modal GPU container; everything comes from the merged config dict.
"""
import json
import random
import time
from collections import Counter
from pathlib import Path

from src.utils.io import write_json
from src.utils.logging import get_logger
from src.utils.tracking import end_run, log_metrics, log_table, start_run

log = get_logger("align")


# ── setup ───────────────────────────────────────────────────────────────────

def llm_model_cfg(cfg: dict) -> dict:
    from src.utils.config import load_yaml

    return load_yaml(cfg["llm_config"])["model"]


def load_parts(cfg: dict, adapter_path: str | None = None, train: bool = False) -> dict:
    """Frozen Qwen3 (4-bit) + frozen Whisper encoder + adapter + tokenizer with the placeholder token."""
    import torch

    from src.components.adapters.registry import build_adapter, load_adapter_weights
    from src.components.encoders import load_whisper
    from src.components.llm import load_base, load_tokenizer
    from src.components.tokens import add_placeholder
    from src.services.audio_data import prompt_parts

    mcfg = llm_model_cfg(cfg)
    tokenizer = load_tokenizer(mcfg, padding_side="left")
    llm = load_base(mcfg)
    llm.requires_grad_(False)
    if train and cfg["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train(train)  # checkpointing only runs in train mode; Qwen3 has no dropout, so this changes nothing else
    placeholder_id = add_placeholder(tokenizer, cfg["prompt"]["placeholder"], llm.get_input_embeddings().weight.shape[0])
    target_norm = float(llm.get_input_embeddings().weight.float().norm(dim=-1).mean())
    adapter = build_adapter(cfg["adapter"], cfg["encoder"]["dim"], target_norm).to("cuda", torch.float32)
    log.info("mean Qwen embedding length %.3f", target_norm)
    if adapter_path:
        load_adapter_weights(adapter, Path(adapter_path) / "adapter.safetensors")
    adapter.train(train)
    return {"llm": llm, "tokenizer": tokenizer, "whisper": load_whisper(cfg["encoder"]), "adapter": adapter,
            "placeholder_id": placeholder_id, "prompt": prompt_parts(tokenizer, cfg["prompt"], placeholder_id),
            "stop_ids": [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]}


def rows_for(cfg: dict, split: str) -> list[dict]:
    from src.services.audio_data import load_rows

    return load_rows(cfg["data"]["sources"][split], cfg["data"]["max_seconds"], cfg["train"]["seed"])


def train_rows(cfg: dict) -> list[dict]:
    """All train rows, or `limit_train` of them taken as whole random shards (a random scatter of rows
    would force reading nearly every shard)."""
    from src.services.audio_data import shard_key

    t = cfg["train"]
    rows = rows_for(cfg, "train")
    if not t["limit_train"]:
        return rows
    by_shard: dict = {}
    for row in rows:
        by_shard.setdefault(shard_key(row), []).append(row)
    keys = sorted(by_shard)
    random.Random(t["seed"]).shuffle(keys)
    picked: list[dict] = []
    for key in keys:
        picked += by_shard[key]
        if len(picked) >= t["limit_train"]:
            break
    return picked[: t["limit_train"]]


def batches_for(rows: list[dict], cfg: dict, parts: dict, micro_batch: int, seed: int, epoch: int = 0):
    """Iterator of collated batches over `rows` in plan order."""
    from src.services.audio_data import collate, iterate_audio, plan_epoch

    plan = plan_epoch(rows, micro_batch, cfg["data"]["group_shards"], seed, epoch)
    return plan, (collate(r, w, parts["prompt"], parts["tokenizer"], cfg) for r, w in iterate_audio(rows, plan))


# ── checkpoints ─────────────────────────────────────────────────────────────

def save_checkpoint(adapter, optimizer, path: Path, state: dict, cfg: dict, with_optimizer: bool) -> None:
    import torch

    from src.components.adapters.registry import save_adapter

    path.mkdir(parents=True, exist_ok=True)
    save_adapter(adapter, path / "adapter.safetensors")
    write_json(path / "adapter_config.json", {"adapter": cfg["adapter"], "encoder": cfg["encoder"], "prompt": cfg["prompt"]})
    if with_optimizer:
        torch.save(optimizer.state_dict(), path / "optimizer.pt")
    write_json(path / "state.json", state)


def try_resume(adapter, optimizer, path: Path) -> dict:
    if not (path / "state.json").exists():
        return {}
    import torch

    from src.components.adapters.registry import load_adapter_weights

    load_adapter_weights(adapter, path / "adapter.safetensors")
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cuda"))
    return json.loads((path / "state.json").read_text())


# ── training ────────────────────────────────────────────────────────────────

def run_training(cfg: dict, run_id: str, tracking_cfg: dict, commit=lambda: None) -> dict:
    import torch

    from src.components.align_step import evaluate_loss, train_step
    from src.services.audio_data import steps_per_epoch

    t, o = cfg["train"], cfg["optim"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    parts = load_parts(cfg, train=True)
    adapter, pid = parts["adapter"], parts["placeholder_id"]
    train = train_rows(cfg)
    dev = train if t["dev_is_train"] else rows_for(cfg, "dev")
    _, dev_iter = batches_for(dev, cfg, parts, cfg["eval"]["batch_size"], seed=0)
    dev_set = list(dev_iter)
    n_micro = len(batches_for(train, cfg, parts, t["micro_batch"], t["seed"])[0])
    per_epoch = steps_per_epoch(n_micro, t["grad_accum"])
    total_steps = t["max_steps"] or per_epoch * t["epochs"]

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"])
    run_dir = Path(cfg["paths"]["ckpt_dir"]) / run_id
    resumed = try_resume(adapter, optimizer, run_dir / "last")
    step, best = resumed.get("step", 0), resumed.get("best_dev_loss", float("inf"))
    counts = {"adapter": sum(p.numel() for p in adapter.parameters())}
    hours = sum(r["n_samples"] for r in train) / 16000 / 3600
    run = start_run(tracking_cfg, run_id, {**cfg, "params": counts, "n_train": len(train), "train_hours": round(hours, 1),
                                           "n_dev": len(dev), "total_steps": total_steps}, Path(cfg["paths"]["logs_dir"]) / f"{run_id}.jsonl")
    log.info("adapter params %s · train %d clips (%.0f h) · dev %d · steps %d%s", f"{counts['adapter']:,}", len(train), hours,
             len(dev), total_steps, f" · resumed at {step}" if step else "")
    started = time.time()
    while step < total_steps:
        epoch = step // per_epoch
        _, batches = batches_for(train, cfg, parts, t["micro_batch"], t["seed"], epoch)
        skip = (step - epoch * per_epoch) * t["grad_accum"]
        group: list[dict] = []
        for index, batch in enumerate(batches):
            if index < skip:
                continue
            group.append(batch)
            if len(group) < t["grad_accum"]:
                continue
            tick = time.time()
            metrics = train_step(parts["llm"], adapter, parts["whisper"], optimizer, group, step, total_steps, pid, cfg)
            audio_s = sum(len(w) for b in group for w in b["waveforms"]) / 16000
            group, step = [], step + 1
            if step % tracking_cfg["log_every_steps"] == 0 or step in (1, total_steps):
                metrics.update({"epoch": step / per_epoch, "audio_sec_per_sec": audio_s / (time.time() - tick),
                                "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9})
                log_metrics(run, metrics, step, "train/")
                log.info("step %d/%d loss %.4f norm %.2f lr %.2e · %.0f× real time", step, total_steps, metrics["loss"],
                         metrics["norm_ratio"], metrics["lr"], metrics["audio_sec_per_sec"])
            if step % t["eval_every"] == 0 or (step == total_steps and t.get("final_eval", True)):
                dev_metrics = evaluate_loss(parts["llm"], adapter, parts["whisper"], dev_set, pid, cfg)
                log_metrics(run, dev_metrics, step, "dev/")
                log.info("dev %s (best %.4f)", {k: round(v, 4) for k, v in dev_metrics.items()}, best)
                if dev_metrics["loss"] < best:
                    best = dev_metrics["loss"]
                    save_checkpoint(adapter, optimizer, run_dir / "best", {"step": step, "dev_loss": best}, cfg, False)
                    commit()
            if step % t["save_every"] == 0:
                save_checkpoint(adapter, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, cfg, True)
                commit()
            if step >= total_steps:
                break
        else:
            if group:  # leftover micro-batches at the end of an epoch still make a step
                train_step(parts["llm"], adapter, parts["whisper"], optimizer, group, step, total_steps, pid, cfg)
                step += 1
    save_checkpoint(adapter, optimizer, run_dir / "last", {"step": step, "best_dev_loss": best}, cfg, True)
    if best == float("inf"):
        save_checkpoint(adapter, optimizer, run_dir / "best", {"step": step, "dev_loss": None}, cfg, False)
    commit()
    summary = {"run_id": run_id, "steps": step, "best_dev_loss": best, "hours": round((time.time() - started) / 3600, 3),
               "train_hours_audio": round(hours, 1), "params": counts, "best_adapter": str(run_dir / "best")}
    write_json(run_dir / "summary.json", summary)
    end_run(run)
    return summary


# ── evaluation ──────────────────────────────────────────────────────────────

def transcribe_rows(parts: dict, rows: list[dict], cfg: dict) -> list[str]:
    import torch

    from src.components.align_step import adapter_tokens
    from src.components.generate import generate_from_embeds
    from src.components.merge import audio_tokens, merge_embeddings
    from src.services.audio_data import iterate_audio, plan_epoch

    p, tok, k = parts["prompt"], parts["tokenizer"], cfg["adapter"]["frame_stack"]
    out: dict[str, str] = {}
    plan = plan_epoch(rows, cfg["eval"]["batch_size"], cfg["data"]["group_shards"], 0, 0)
    for picked, waveforms in iterate_audio(rows, plan):
        prompts = [p["before"] + [p["placeholder_id"]] * audio_tokens(len(w), cfg["encoder"], k) + p["after"] for w in waveforms]
        width = max(map(len, prompts))
        ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in prompts], device="cuda")
        mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in prompts], device="cuda")
        with torch.no_grad():
            audio = adapter_tokens(parts["adapter"], parts["whisper"], waveforms, cfg)
            embeds = merge_embeddings(parts["llm"].get_input_embeddings()(ids), ids, p["placeholder_id"], audio)
        texts = generate_from_embeds(parts["llm"], tok, embeds, mask, cfg["eval"]["max_new_tokens"], parts["stop_ids"])
        out.update({r["clip_id"] + "|" + r["audio_dir"]: t.strip() for r, t in zip(picked, texts)})
    return [out[r["clip_id"] + "|" + r["audio_dir"]] for r in rows]


def probe(parts: dict, rows: list[dict], cfg: dict) -> dict:
    """Nearest Qwen token to each adapter output. Healthy: many different tokens, not one repeated."""
    import torch
    import torch.nn.functional as F

    from src.components.align_step import adapter_tokens
    from src.services.audio_data import iterate_audio, plan_epoch

    plan = plan_epoch(rows[:64], 64, cfg["data"]["group_shards"], 0, 0)
    _, waveforms = next(iterate_audio(rows[:64], plan))
    with torch.no_grad():
        audio = adapter_tokens(parts["adapter"], parts["whisper"], waveforms, cfg)[: cfg["eval"]["probe_tokens"]]
        table = F.normalize(parts["llm"].get_input_embeddings().weight.float(), dim=-1)
        nearest = (F.normalize(audio.float(), dim=-1) @ table.T).argmax(dim=-1).tolist()
    counts = Counter(nearest)
    return {"tokens": len(nearest), "distinct": len(counts), "distinct_ratio": round(len(counts) / max(1, len(nearest)), 3),
            "top": [(parts["tokenizer"].decode([i]), c) for i, c in counts.most_common(10)]}


def wer_table(rows: list[dict], hyps: list[str], normalise) -> dict:
    import jiwer

    by: dict[str, tuple[list[str], list[str]]] = {}
    for row, hyp in zip(rows, hyps):
        for key in ("all", row["source"], f"{row['source']}:{'clean' if not row.get('noise') else 'noisy'}"):
            refs, outs = by.setdefault(key, ([], []))
            refs.append(normalise(row["text"]))
            outs.append(normalise(hyp) or "-")
    return {k: {"n": len(r), "wer": round(jiwer.wer(r, h), 4)} for k, (r, h) in sorted(by.items())}


def run_evaluation(cfg: dict, adapter_path: str, split: str, name: str, tracking_cfg: dict, limit: int | None = None) -> dict:
    from transformers import WhisperTokenizer

    started = time.time()
    parts = load_parts(cfg, adapter_path)
    rows = train_rows(cfg) if cfg["train"]["dev_is_train"] else rows_for(cfg, split)   # overfit: the same clips
    if limit:
        rows = random.Random(0).sample(rows, min(limit, len(rows)))
    hyps = transcribe_rows(parts, rows, cfg)
    normalise = WhisperTokenizer.from_pretrained(cfg["encoder"]["id"], revision=cfg["encoder"]["revision"]).normalize
    summary = {"name": name, "adapter": adapter_path, "split": split, "wer": wer_table(rows, hyps, normalise),
               "probe": probe(parts, rows, cfg), "minutes": round((time.time() - started) / 60, 2)}
    samples = [{"source": r["source"], "noise": r.get("noise"), "text": r["text"], "heard": h} for r, h in zip(rows, hyps)]
    write_json(Path(cfg["paths"]["evals_dir"]) / f"{name}.json", summary)
    write_json(Path(cfg["paths"]["evals_dir"]) / f"{name}.samples.json", samples)
    run = start_run(tracking_cfg, f"eval-{name}", {"adapter": adapter_path, "split": split})
    log_metrics(run, {f"wer_{k}": v["wer"] for k, v in summary["wer"].items()}, 0, f"eval/{split}/")
    log_table(run, f"samples-{name}", random.Random(0).sample(samples, min(cfg["eval"]["samples_to_log"], len(samples))))
    end_run(run)
    return summary
