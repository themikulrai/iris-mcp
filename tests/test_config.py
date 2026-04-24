"""Tests for configuration loading."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from slurm_mcp.config import SlurmConfig


def test_defaults_derived_from_username():
    with patch.dict("os.environ", {"SLURM_MCP_USERNAME": "alice"}, clear=False):
        cfg = SlurmConfig(username="alice")
    assert cfg.resolved_working_dir == "/iris/u/alice"
    assert cfg.resolved_conda_dir == "/iris/u/alice/data/miniforge3"
    assert cfg.resolved_home_dir == "/sailhome/alice"


def test_explicit_overrides():
    with patch.dict("os.environ", {"SLURM_MCP_USERNAME": "alice"}, clear=False):
        cfg = SlurmConfig(
            username="alice",
            working_dir="/custom/path",
            conda_dir="/custom/conda",
            home_dir="/custom/home",
        )
    assert cfg.resolved_working_dir == "/custom/path"
    assert cfg.resolved_conda_dir == "/custom/conda"
    assert cfg.resolved_home_dir == "/custom/home"


def test_partition_list():
    with patch.dict("os.environ", {"SLURM_MCP_USERNAME": "alice"}, clear=False):
        cfg = SlurmConfig(username="alice", partitions="a, b, c")
    assert cfg.partition_list == ["a", "b", "c"]


def test_default_slurm_values():
    with patch.dict("os.environ", {"SLURM_MCP_USERNAME": "alice"}, clear=False):
        cfg = SlurmConfig(username="alice")
    assert cfg.default_account == "iris"
    assert cfg.default_partition == "iris"
    assert cfg.default_time == "24:00:00"
    assert cfg.default_mem == "32G"
    assert cfg.default_gpus == 1
    assert cfg.default_cpus == 4


def test_missing_username_raises():
    # _env_file=None disables .env discovery so the test doesn't pick up a
    # real .env in the project root.
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(Exception):
            SlurmConfig(_env_file=None)
