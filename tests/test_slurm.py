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
        result = await client.run_remote("hostname", timeout=5)

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
            await client.run_remote("hostname")


@pytest.mark.asyncio
async def test_run_remote_timeout(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError)
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(ToolError, match="timed out"):
            await client.run_remote("sleep 999", timeout=1)


@pytest.mark.asyncio
async def test_run_remote_generic_error(config):
    client = SlurmClient(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"some error"))
    mock_proc.returncode = 1
    mock_proc.kill = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        with pytest.raises(ToolError, match="some error"):
            await client.run_remote("bad_command")
