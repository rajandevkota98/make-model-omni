import json

from src.utils.tracking import end_run, flatten, log_metrics, log_table, start_run


def test_flatten_nested_config():
    assert flatten({"lora": {"r": 32, "alpha": 64}, "lr": 2e-4}) == {"lora.r": 32, "lora.alpha": 64, "lr": 2e-4}


def test_local_log_without_comet(tmp_path, monkeypatch):
    monkeypatch.delenv("COMET_API_KEY", raising=False)
    path = tmp_path / "run.jsonl"
    run = start_run({"comet": {"enabled": True, "project": "x"}}, "test-run", {"lora": {"r": 32}}, path)
    assert run["comet"] is None  # no key → Comet silently off
    log_metrics(run, {"loss": 1.5}, step=10, prefix="train/")
    log_table(run, "samples", [{"prompt": "hi", "output": "hello"}])
    end_run(run)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["type"] for r in records] == ["params", "metrics", "table"]
    assert records[0]["lora.r"] == 32 and records[1]["train/loss"] == 1.5 and records[1]["step"] == 10
