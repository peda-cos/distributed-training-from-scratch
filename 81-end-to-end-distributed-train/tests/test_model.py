"""Model shape, causal-mask, and deterministic-corpus tests."""

from __future__ import annotations

import torch

from end_to_end_distributed import main as demo


def test_mini_gpt_parameter_count_and_untied_head() -> None:
    model = demo.make_model(demo.DEFAULT_SEED)

    assert demo.parameter_count(model) == demo.PARAM_COUNT == 30144
    assert model.head.weight is not model.token_emb.weight
    assert model.head.out_features == demo.VOCAB_SIZE

    inputs, _ = demo.make_batch(demo.make_corpus(11, 128))
    with torch.no_grad():
        logits = model(inputs)

    assert logits.shape == (demo.BATCH_SIZE, demo.SEQ_LEN, demo.VOCAB_SIZE)


def test_causal_attention_blocks_future_tokens() -> None:
    model = demo.make_model(demo.DEFAULT_SEED)
    model.eval()
    inputs, _ = demo.make_batch(demo.make_corpus(11, 128))
    first = inputs[:1].clone()
    changed = first.clone()
    changed[0, -1] = (changed[0, -1] + 1) % demo.VOCAB_SIZE

    with torch.no_grad():
        before = model(first)
        after = model(changed)

    assert torch.equal(before[:, :-1], after[:, :-1])
    assert not torch.allclose(before[:, -1], after[:, -1])


def test_corpus_and_batch_are_deterministic() -> None:
    corpus = demo.make_corpus(11, demo.CORPUS_TOKENS)
    same = demo.make_corpus(11, demo.CORPUS_TOKENS)
    other = demo.make_corpus(12, demo.CORPUS_TOKENS)
    inputs, targets = demo.make_batch(corpus, demo.BATCH_SIZE)

    assert torch.equal(corpus, same)
    assert not torch.equal(corpus, other)
    assert inputs.shape == targets.shape == (demo.BATCH_SIZE, demo.SEQ_LEN)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])
    assert len({tuple(row.tolist()) for row in inputs}) == demo.BATCH_SIZE
    for start in range(0, demo.CORPUS_TOKENS - demo.SEQ_LEN, demo.BATCH_PHASE_STRIDE):
        window = corpus[start : start + demo.SEQ_LEN + 1]
        assert torch.equal(window[1:], (window[:-1] + 1) % demo.VOCAB_SIZE)
