import pytest

torch = pytest.importorskip("torch")  # CPU-only check; skipped where torch isn't installed

from src.components.adapters.ops import frame_stack  # noqa: E402
from src.components.adapters.registry import build_adapter  # noqa: E402
from src.components.merge import IGNORE_INDEX, audio_tokens, build_example, merge_embeddings, split_prompt  # noqa: E402

ECFG = {"frames_per_second": 50, "max_frames": 1500}


def test_frame_stack_pads_and_widens():
    frames = torch.arange(10 * 3, dtype=torch.float32).reshape(10, 3)
    out = frame_stack(frames, 4)
    assert out.shape == (3, 12)
    assert torch.equal(out[0], frames[:4].reshape(-1)) and torch.equal(out[2, 6:], torch.zeros(6))


def test_audio_token_count():
    assert audio_tokens(16000, ECFG, 4) == 13          # 1 s → 50 frames → 13 tokens
    assert audio_tokens(16000 * 40, ECFG, 4) == 375    # capped at 30 s


def test_build_example_masks_prompt():
    before, after = split_prompt([1, 2, 99, 3], 99)
    ex = build_example(before, after, 99, 3, [7, 8])
    assert ex["input_ids"] == [1, 2, 99, 99, 99, 3, 7, 8]
    assert ex["labels"] == [IGNORE_INDEX] * 6 + [7, 8] and ex["prompt_len"] == 6


def test_split_prompt_needs_one_placeholder():
    with pytest.raises(ValueError):
        split_prompt([99, 1, 99], 99)


def test_merge_fills_placeholders_in_batch_order():
    ids = torch.tensor([[5, 99, 99, 6], [99, 7, 0, 0]])
    text = torch.zeros(2, 4, 2)
    audio = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    merged = merge_embeddings(text, ids, 99, audio)
    assert merged[0, 1].tolist() == [1, 1] and merged[0, 2].tolist() == [2, 2] and merged[1, 0].tolist() == [3, 3]
    assert merged[0, 0].tolist() == [0, 0]


def test_merge_rejects_count_mismatch():
    with pytest.raises(ValueError):
        merge_embeddings(torch.zeros(1, 3, 2), torch.tensor([[99, 99, 1]]), 99, torch.zeros(1, 2))


def test_adapter_maps_to_llm_width_and_gets_gradients():
    adapter = build_adapter({"type": "mlp", "frame_stack": 4, "hidden": 16, "out_dim": 8}, 6)
    out = adapter(frame_stack(torch.randn(9, 6), 4))
    assert out.shape == (3, 8)
    out.sum().backward()
    assert all(p.grad is not None for p in adapter.parameters())


def test_out_norm_matches_target_length():
    adapter = build_adapter({"type": "mlp", "frame_stack": 4, "hidden": 16, "out_dim": 64, "out_norm": True}, 6, target_norm=0.7)
    lengths = adapter(torch.randn(5, 24) * 100).norm(dim=-1)
    assert torch.allclose(lengths, torch.full((5,), 0.7), atol=1e-3)
