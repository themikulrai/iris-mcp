"""Tests for MCP tool functions."""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from slurm_mcp.server import create_server


@pytest.fixture
def mock_env():
    """Environment with required config."""
    with patch.dict("os.environ", {
        "SLURM_MCP_USERNAME": "testuser",
        "SLURM_MCP_LOGIN_NODE": "test-sc.example.com",
        "SLURM_MCP_WORKING_DIR": "/iris/u/testuser",
        "SLURM_MCP_CONDA_DIR": "/iris/u/testuser/data/miniforge3",
        "SLURM_MCP_HOME_DIR": "/sailhome/testuser",
    }, clear=False):
        yield


@pytest.fixture
def mcp_server(mock_env):
    return create_server()


# ── list_jobs ─────────────────────────────────────────────────────────

SQUEUE_OUTPUT = """12345|train_gpt|iris|RUNNING|1:23:45|24:00:00|1|4|gpu:1|iris-hgx-2
12346|eval_model|iris-hi|PENDING|0:00:00|12:00:00|1|4|gpu:2|Priority"""


@pytest.mark.asyncio
async def test_list_jobs(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=SQUEUE_OUTPUT):
        result = await mcp_server.call_tool("list_jobs", {})
        text = result[0][0].text
        assert "12345" in text
        assert "train_gpt" in text
        assert "RUNNING" in text
        assert "12346" in text


@pytest.mark.asyncio
async def test_list_jobs_empty(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=""):
        result = await mcp_server.call_tool("list_jobs", {})
        text = result[0][0].text
        assert "No jobs found" in text


# ── cancel_job ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cancel_job(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=""):
        result = await mcp_server.call_tool("cancel_job", {"job_id": "12345"})
        text = result[0][0].text
        assert "cancelled" in text


@pytest.mark.asyncio
async def test_cancel_job_invalid_id(mcp_server, mock_env):
    with pytest.raises(Exception, match="Invalid job ID"):
        await mcp_server.call_tool("cancel_job", {"job_id": "12345; rm -rf /"})


# ── job_status ────────────────────────────────────────────────────────

SACCT_OUTPUT = "12345|train_gpt|iris|RUNNING|0:0|1:23:45|86400|4|billing=4,gres/gpu=1|4096M|2026-04-09T10:00:00|Unknown|/iris/u/testuser/project"


@pytest.mark.asyncio
async def test_job_status(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=SACCT_OUTPUT):
        result = await mcp_server.call_tool("job_status", {"job_id": "12345"})
        text = result[0][0].text
        assert "RUNNING" in text
        assert "train_gpt" in text


# ── gpu_status ────────────────────────────────────────────────────────

GPU_OUTPUT = """--------------------------------------------------------------------
iris GPU Status
--------------------------------------------------------------------
There are a total of 99 gpus [up]
8 h100 gpus
8 h200 gpus"""


@pytest.mark.asyncio
async def test_gpu_status(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=GPU_OUTPUT):
        result = await mcp_server.call_tool("gpu_status", {})
        text = result[0][0].text
        assert "99 gpus" in text
        assert "h100" in text


# ── cluster_info ──────────────────────────────────────────────────────

SINFO_OUTPUT = """iris|up|21-00:00:00|1|mixed|gpu:h100:8
iris-hi|up|21-00:00:00|1|mixed|gpu:h200:8"""


@pytest.mark.asyncio
async def test_cluster_info(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=SINFO_OUTPUT):
        result = await mcp_server.call_tool("cluster_info", {})
        text = result[0][0].text
        assert "iris" in text
        assert "h100" in text


# ── interactive_session ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_interactive_session(mcp_server, mock_env):
    result = await mcp_server.call_tool("interactive_session", {"gpus": 2, "mem": "64GB"})
    text = result[0][0].text
    assert "srun" in text
    assert "gpu:2" in text
    assert "64GB" in text
    assert "tmux" in text


@pytest.mark.asyncio
async def test_interactive_session_gpu_type(mcp_server, mock_env):
    result = await mcp_server.call_tool("interactive_session", {"gpu_type": "h100", "gpus": 4})
    text = result[0][0].text
    assert "gpu:h100:4" in text


# ── check_auth ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_auth(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value="test-sc.example.com"):
        result = await mcp_server.call_tool("check_auth", {})
        text = result[0][0].text
        assert "Connected" in text
        assert "testuser" in text


# ── tail_output ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tail_output_direct_path(mcp_server, mock_env):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".out", delete=False) as f:
        for i in range(100):
            f.write(f"line {i}\n")
        f.flush()
        path = f.name

    try:
        result = await mcp_server.call_tool("tail_output", {"file_path": path, "lines": 10})
        text = result[0][0].text
        assert "line 99" in text
        assert "last 10 of 100" in text
    finally:
        os.unlink(path)


# ── env_list ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_env_list(mcp_server, mock_env):
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = json.dumps({"envs": ["/iris/u/testuser/data/miniforge3", "/iris/u/testuser/envs/myenv"]})
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result), \
         patch("os.path.exists", return_value=True):
        result = await mcp_server.call_tool("env_list", {})
        text = result[0][0].text
        assert "base" in text
        assert "myenv" in text
