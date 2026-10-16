"""Placeholder tokens: a slot in the prompt that the adapter's vectors replace.

Qwen3's embedding table has 151,936 rows but the tokenizer only uses 151,669, so new special tokens
fit in spare rows. The table is never resized (resizing would change the frozen model).
"""


def add_placeholder(tokenizer, token: str, n_rows: int) -> int:
    tokenizer.add_tokens([token], special_tokens=True)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id >= n_rows:
        raise ValueError(f"{token} got id {token_id}; it must fit in the existing {n_rows}-row embedding table")
    return token_id
