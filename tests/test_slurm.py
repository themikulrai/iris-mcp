"""Tests for the SSH executor and command builder."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from mcp.server.fastmcp.exceptions import ToolError

from slurm_mcp.slurm import SlurmClient


def test_build_sbatch_args_defaults(client):
    args = client.build_sbatch_args()
    assert args[0] == "sbatch"
    assert "--partition" in args
    assert args[args.index("--partition") + 1] == "iris"
    assert "--account" in args
    assert args[args.index("--account") + 1] == "iris"
    assert "--gres" in args
    assert args[args.index("--gres") + 1] == "gpu:1"


def test_build_sbatch_args_custom(client):
    args = client.build_sbatch_args(
        job_name="test_job",
        partition="iris-hi",
        gpus=4,
        gpu_type="h100",
        mem="64G",
        time_limit="48:00:00",
        array="0-7",
    )
    assert "--job-name" in args
    assert args[args.index("--job-name") + 1] == "test_job"
    assert args[args.index("--partition") + 1] == "iris-hi"
    assert args[args.index("--gres") + 1] == "gpu:h100:4"
    assert args[args.index("--mem") + 1] == "64G"
    assert args[args.index("--time") + 1] == "48:00:00"
    assert "--array" in args
    assert args[args.index("--array") + 1] == "0-7"


def test_build_sbatch_args_extra_args(client):
    args = client.build_sbatch_args(extra_args="--exclude=node1 --nice=100")
    assert "--exclude=node1" in args
    assert "--nice=100" in args


@pytest.mark.asyncio
async def test_run_remote_success(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"sc.stanford.edu\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await client.run_remote(["hostname"], timeout=5)

    assert result == "sc.stanford.edu"


@pytest.mark.asyncio
async def test_run_remote_kerberos_error(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"Permission denied (GSSAPI)"))
    mock_proc.returncode = 255
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(ToolError, match="Kerberos ticket likely expired"):
            await client.run_remote(["hostname"])


@pytest.mark.asyncio
async def test_run_remote_timeout(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(ToolError, match="timed out"):
            await client.run_remote(["sleep", "999"], timeout=1)


@pytest.mark.asyncio
async def test_run_remote_generic_error(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
    mock_proc.returncode = 1
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(ToolError, match="some error"):
            await client.run_remote(["bad_command"])


# ── Phase 1b/1d: SSH ControlMaster + run_remote refactor ─────────────


@pytest.mark.asyncio
async def test_run_remote_adds_controlmaster(config):
    """ControlMaster flags are injected so subsequent SSH calls multiplex."""
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"ok\n", b""))
    mock_proc.returncode = 0
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as spawn:
        await client.run_remote(["hostname"])

    argv = list(spawn.call_args.args)
    assert "ControlMaster=auto" in argv
    # ControlPersist default per plan = 600s
    assert any(a == "ControlPersist=600s" for a in argv)
    # ControlPath ends in cm-%C (fixed-length hash filename)
    assert any(a.startswith("ControlPath=") and a.endswith("cm-%C") for a in argv)


@pytest.mark.asyncio
async def test_run_remote_rejects_str(config):
    """run_remote no longer accepts a raw str command; list-only."""
    client = SlurmClient(config)
    with pytest.raises(TypeError):
        await client.run_remote("hostname")


@pytest.mark.asyncio
async def test_run_remote_kills_on_cancel(config):
    """Client cancellation must kill the SSH subprocess — no orphans."""
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.CancelledError)
    mock_proc.kill = AsyncMock()
    mock_proc.returncode = None

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(asyncio.CancelledError):
            await client.run_remote(["sleep", "999"])

    assert mock_proc.kill.called


# ── Phase 4a: parse_sgpu structured output ───────────────────────────


def test_parse_sgpu_typical():
    from slurm_mcp.slurm import parse_sgpu

    text = (
        "--------------------------------------------------------------------\n"
        "iris GPU Status\n"
        "--------------------------------------------------------------------\n"
        "There are a total of 99 gpus [up]\n"
        "8 h100 gpus\n"
        "8 h200 gpus"
    )
    result = parse_sgpu(text)
    assert result["partition"] == "iris"
    assert result["total"] == 99
    assert result["state"] == "up"
    names = {t["name"]: t for t in result["types"]}
    assert names["h100"]["count"] == 8
    assert names["h100"]["vram_gb"] == 80
    assert names["h200"]["vram_gb"] == 141
    assert text in result["raw"]


def test_parse_sgpu_unknown_type():
    from slurm_mcp.slurm import parse_sgpu

    text = "iris GPU Status\nThere are a total of 1 gpus [up]\n1 weirdgpu gpus"
    result = parse_sgpu(text)
    types = {t["name"]: t for t in result["types"]}
    assert types["weirdgpu"]["vram_gb"] is None


def test_parse_sgpu_malformed():
    from slurm_mcp.slurm import parse_sgpu

    result = parse_sgpu("total garbage\nno structure here")
    assert result["total"] is None
    assert result["types"] == []
    assert result["raw"]  # raw always preserved


# ── Phase 4b: build_sbatch_args new params ───────────────────────────


def test_build_sbatch_args_new_params(client):
    args = client.build_sbatch_args(
        dependency="afterok:100",
        nodelist="iris-hgx-2",
        exclude="iris-1",
        constraint="a100",
    )
    assert args[args.index("--dependency") + 1] == "afterok:100"
    assert args[args.index("--nodelist") + 1] == "iris-hgx-2"
    assert args[args.index("--exclude") + 1] == "iris-1"
    assert args[args.index("--constraint") + 1] == "a100"
