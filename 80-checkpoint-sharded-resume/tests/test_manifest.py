"""Manifest schema, serialization, and rejection tests."""

from __future__ import annotations

import dataclasses
import json

import pytest

from sharded_checkpoint import main as ckpt


def _entry(
    rank: int,
    offset: int,
    numel: int,
    *,
    path: str | None = None,
    sha256: str = "a" * 64,
) -> ckpt.ShardEntry:
    return ckpt.ShardEntry(
        rank=rank,
        path=path if path is not None else f"rank{rank}.bin",
        sha256=sha256,
        param_shard_offset=offset,
        param_shard_numel=numel,
    )


def _manifest() -> ckpt.ShardManifest:
    return ckpt.ShardManifest(
        world_size=2,
        step=123,
        wall_clock_seconds=45,
        shards=(_entry(0, 0, 10), _entry(1, 10, 5)),
    )


def test_manifest_json_round_trip_and_exact_keys() -> None:
    manifest = _manifest()
    parsed = ckpt.ShardManifest.from_json(manifest.to_json())

    assert parsed == manifest
    payload = json.loads(manifest.to_json())
    assert list(payload) == [
        "world_size",
        "step",
        "wall_clock_seconds",
        "shards",
        "schema_version",
    ]
    assert list(payload["shards"][0]) == [
        "rank",
        "path",
        "sha256",
        "param_shard_offset",
        "param_shard_numel",
    ]


def test_bad_schema_and_shard_count_are_rejected() -> None:
    manifest = _manifest()

    with pytest.raises(ckpt.ManifestError, match="schema_version"):
        dataclasses.replace(manifest, schema_version=2)
    with pytest.raises(ckpt.ManifestError, match="shards for world_size"):
        ckpt.ShardManifest(
            world_size=3,
            step=1,
            wall_clock_seconds=0,
            shards=manifest.shards,
        )
    payload = json.loads(manifest.to_json())
    del payload["schema_version"]
    with pytest.raises(ckpt.ManifestError, match="missing fields"):
        ckpt.ShardManifest.from_json(json.dumps(payload))


def test_rank_and_offset_contiguity_are_enforced() -> None:
    with pytest.raises(ckpt.ManifestError, match="param_shard_offset"):
        ckpt.ShardManifest(
            world_size=2,
            step=1,
            wall_clock_seconds=0,
            shards=(_entry(0, 1, 10), _entry(1, 11, 5)),
        )
    with pytest.raises(ckpt.ManifestError, match="param_shard_offset"):
        ckpt.ShardManifest(
            world_size=2,
            step=1,
            wall_clock_seconds=0,
            shards=(_entry(0, 0, 10), _entry(1, 11, 5)),
        )
    with pytest.raises(ckpt.ManifestError, match="rank"):
        ckpt.ShardManifest(
            world_size=2,
            step=1,
            wall_clock_seconds=0,
            shards=(_entry(0, 0, 10), _entry(1, 10, 5, path="rank2.bin")),
        )
    with pytest.raises(ckpt.ManifestError, match="rank"):
        ckpt.ShardManifest(
            world_size=2,
            step=1,
            wall_clock_seconds=0,
            shards=(_entry(0, 0, 10), _entry(0, 10, 5)),
        )


def test_entry_path_and_digest_defenses() -> None:
    manifest = _manifest()
    with pytest.raises(ckpt.ManifestError, match="path"):
        ckpt.ShardManifest(
            world_size=2,
            step=1,
            wall_clock_seconds=0,
            shards=(
                _entry(0, 0, 10, path="../rank0.bin"),
                _entry(1, 10, 5),
            ),
        )
    bad_payload = {
        "rank": 0,
        "path": "rank0.bin",
        "sha256": "A" * 64,
        "param_shard_offset": 0,
        "param_shard_numel": 1,
    }
    with pytest.raises(ckpt.ManifestError, match="sha256"):
        ckpt.ShardEntry.from_dict(bad_payload)
    with pytest.raises(ckpt.ManifestError, match="rank"):
        ckpt.ShardEntry.from_dict({**bad_payload, "rank": True, "sha256": "a" * 64})
    with pytest.raises(ckpt.ManifestError, match="param_shard_numel"):
        ckpt.ShardEntry.from_dict({**bad_payload, "param_shard_numel": 0, "sha256": "a" * 64})
    assert manifest.world_size == 2


def test_manifest_json_is_deterministic() -> None:
    first = _manifest().to_json()
    second = _manifest().to_json()
    assert first == second
    assert first.endswith("\n")
    assert "schema_version" in first
