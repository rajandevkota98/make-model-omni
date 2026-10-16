from src.components.batching import PROMPT, REPLY, TOOL_CALL, epoch_batches, token_types
from src.utils.config import deep_merge
from src.utils.schedule import lr_at


def test_lr_warmup_then_cosine():
    lrs = [lr_at(s, 100, 1.0, 0.1, 0.1) for s in range(100)]
    assert lrs[0] == 0.1 and lrs[9] == 1.0          # 10 warmup steps
    assert lrs[-1] > 0.1 and lrs[-1] < 0.11         # ends near min_lr_ratio
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))


def test_deep_merge():
    assert deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}}) == {"a": {"x": 1, "y": 3}, "b": 1}


def test_token_types_marks_tool_span():
    # prompt(2) | "Ok" | <tool_call> json </tool_call> | <|im_end|>
    ids = [9, 9, 5, 100, 7, 101, 2]
    assert token_types(ids, 2, open_id=100, close_id=101) == [PROMPT, PROMPT, REPLY, TOOL_CALL, TOOL_CALL, TOOL_CALL, REPLY]


def test_epoch_batches_cover_everything_once():
    batches = epoch_batches(10, 3, seed=1, epoch=0)
    assert sorted(i for b in batches for i in b) == list(range(10))
    assert batches != epoch_batches(10, 3, seed=1, epoch=1)
