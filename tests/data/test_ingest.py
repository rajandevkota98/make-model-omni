from src.services.data.ingest import ingest_file


def test_ingest_skips_broken_lines(tmp_path):
    path = tmp_path / "batch.jsonl"
    path.write_text('{"id": "a"}\nnot json\n\n[1, 2]\n{"id": "b"}\n')
    rows, rejected = ingest_file(path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[1]["_source"] == {"file": "batch.jsonl", "line": 5}
    assert [r["reasons"][0].split(":")[0] for r in rejected] == ["bad_json", "not_an_object"]
