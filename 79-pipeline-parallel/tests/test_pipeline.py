"""Real stagewise execution, gradient equality, and validation tests."""

from __future__ import annotations

import pytest
import torch

from pipeline_parallel import main as pipeline


def _small_config(**overrides: int) -> pipeline.RunConfig:
    values = {
        "batch_size": 16,
        "num_microbatches": 4,
        "num_stages": 4,
        "hidden_dim": 8,
        "seed": 11,
    }
    values.update(overrides)
    return pipeline.RunConfig(**values)


def test_pipeline_stage_api() -> None:
    config = _small_config()
    stages = pipeline.build_pipeline(config, dtype=torch.float64).stages
    inputs, _ = pipeline.make_batch(config, dtype=torch.float64)

    assert all(isinstance(stage, torch.nn.Module) for stage in stages)
    with torch.no_grad():
        expected = inputs
        for stage in stages:
            expected = stage(expected)
        actual = pipeline.build_pipeline(config, dtype=torch.float64)(inputs)

    assert torch.equal(actual, expected)
    assert sum(parameter.numel() for stage in stages for parameter in stage.parameters()) > 0


def test_gpipe_matches_full_batch() -> None:
    config = _small_config()
    inputs, targets = pipeline.make_batch(config, dtype=torch.float64)
    gpipe = pipeline.build_pipeline(config, "gpipe", dtype=torch.float64)

    reference_loss, reference_grads = pipeline.reference_gradients(
        gpipe, inputs, targets
    )
    result = gpipe.step(inputs, targets)

    assert result.loss == pytest.approx(reference_loss, abs=1e-10)
    assert pipeline.max_grad_delta(gpipe, reference_grads) <= 1e-9
    assert len(result.losses) == config.num_microbatches
    assert result.loss == pytest.approx(
        sum(result.losses) / config.num_microbatches, abs=1e-12
    )
    assert result.trace.idle_fraction == pytest.approx(
        pipeline.bubble_fraction(config.num_stages, config.num_microbatches)
    )


def test_1f1b_matches_full_batch_and_gpipe() -> None:
    config = _small_config()
    inputs, targets = pipeline.make_batch(config, dtype=torch.float64)
    gpipe = pipeline.build_pipeline(config, "gpipe", dtype=torch.float64)
    onef1b = pipeline.build_pipeline(config, "1f1b", dtype=torch.float64)

    reference_loss, reference_grads = pipeline.reference_gradients(
        gpipe, inputs, targets
    )
    gpipe_result = gpipe.step(inputs, targets)
    onef1b_result = onef1b.step(inputs, targets)

    assert gpipe_result.loss == pytest.approx(reference_loss, abs=1e-10)
    assert onef1b_result.loss == pytest.approx(reference_loss, abs=1e-10)
    assert pipeline.max_grad_delta(gpipe, reference_grads) <= 1e-9
    assert pipeline.max_grad_delta(onef1b, reference_grads) <= 1e-9

    gpipe_grads = [parameter.grad for parameter in gpipe.parameters()]
    for expected, actual in zip(gpipe_grads, onef1b.parameters()):
        assert expected is not None
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected, rtol=1e-9, atol=1e-12)


def test_measured_high_watermark_matches_trace() -> None:
    config = _small_config(batch_size=32, num_microbatches=8)
    inputs, targets = pipeline.make_batch(config, dtype=torch.float64)

    for schedule, expected_peak in (("gpipe", 8), ("1f1b", 4)):
        model = pipeline.build_pipeline(config, schedule, dtype=torch.float64)
        result = model.step(inputs, targets)
        assert (
            result.measured_activation_high_watermark
            == result.trace.activation_high_watermark
        )
        assert max(result.measured_activation_high_watermark) == expected_peak


def test_step_rejects_bad_microbatch_split() -> None:
    config = _small_config()
    model = pipeline.build_pipeline(config)
    inputs, targets = pipeline.make_batch(config)

    with pytest.raises(ValueError, match="divisible"):
        model.step(torch.randn(10, config.hidden_dim), torch.randint(0, 4, (10,)))
    with pytest.raises(ValueError, match="divisible"):
        model.step(torch.randn(2, config.hidden_dim), torch.randint(0, 4, (2,)))
    with pytest.raises(ValueError, match="non-empty"):
        model.step(torch.empty(0, config.hidden_dim), torch.empty(0, dtype=torch.long))
    with pytest.raises(ValueError, match="same batch size"):
        model.step(inputs, targets[:-1])
    with pytest.raises(ValueError, match="two-dimensional"):
        model.step(inputs[:, 0], targets)
    with pytest.raises(ValueError, match="one-dimensional"):
        model.step(inputs, targets.unsqueeze(1))
