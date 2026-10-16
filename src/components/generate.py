"""Greedy generation from already-tokenized prompts (left-padded batches)."""
import torch


@torch.no_grad()
def generate_batch(model, tokenizer, prompts: list[list[int]], max_new_tokens: int, stop_ids: list[int]) -> list[str]:
    pad_id = tokenizer.pad_token_id
    width = max(len(p) for p in prompts)
    input_ids = torch.tensor([[pad_id] * (width - len(p)) + p for p in prompts], device="cuda")
    attention_mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in prompts], device="cuda")
    output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                            do_sample=False, eos_token_id=stop_ids, pad_token_id=pad_id, use_cache=True)
    return [tokenizer.decode(row[width:], skip_special_tokens=False) for row in output]


@torch.no_grad()
def generate_from_embeds(model, tokenizer, embeds: torch.Tensor, attention_mask: torch.Tensor, max_new_tokens: int,
                         stop_ids: list[int]) -> list[str]:
    """Greedy generation from left-padded input embeddings (audio prompts); returns only the new text."""
    output = model.generate(inputs_embeds=embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                            do_sample=False, eos_token_id=stop_ids, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    return tokenizer.batch_decode(output, skip_special_tokens=True)
