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


# ── Phase 1c: tightened _JOB_ID_RE ──────────────────────────────────


@pytest.mark.asyncio
async def test_cancel_job_rejects_doubled_separators(mcp_server, mock_env):
    for bad in ("12345,,67", "12345---67", ",,,", "---"):
        with pytest.raises(Exception, match="Invalid job ID"):
            await mcp_server.call_tool("cancel_job", {"job_id": bad})


# ── Phase 2a: cluster_info / gpu_status reject shell injection ─────


@pytest.mark.asyncio
async def test_cluster_info_rejects_injection(mcp_server, mock_env):
    with pytest.raises(Exception, match="[Ii]nvalid|[Pp]artition"):
        await mcp_server.call_tool("cluster_info", {"partition": "iris; rm -rf /"})


# ── Phase 2c: preemption warning generalized ─────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "partition,expect_warn",
    [
        ("iris", True),
        ("iris-hi", False),
        ("iris-interactive", False),
        ("iris-hi-interactive", False),
        # When partition is None, auto-routing kicks in. The mock returns the
        # same "Submitted batch job 42" string for every remote call, including
        # the squeue probe — splitlines() gives count=1 which is well under the
        # cap, so iris-hi is selected (no preemption warning).
        (None, False),
    ],
)
async def test_submit_job_preemption_warning(mcp_server, mock_env, tmp_path, partition, expect_warn):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value="Submitted batch job 42"):
        args = {"script_content": "#!/bin/bash\necho ok", "working_dir": str(tmp_path)}
        if partition is not None:
            args["partition"] = partition
        result = await mcp_server.call_tool("submit_job", args)
        text = result[0][0].text
        if expect_warn:
            assert "preemptible" in text
        else:
            assert "preemptible" not in text


# ── Phase 4a: gpu_status returns structured JSON + summary ──────────


@pytest.mark.asyncio
async def test_gpu_status_returns_json_and_summary(mcp_server, mock_env):
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=GPU_OUTPUT):
        result = await mcp_server.call_tool("gpu_status", {})
        text = result[0][0].text

    parsed = json.loads(text)
    # structured fields
    assert parsed["total"] == 99
    types = {t["name"]: t for t in parsed["types"]}
    assert types["h100"]["vram_gb"] == 80
    assert types["h200"]["vram_gb"] == 141
    # summary preserves the original sgpu text so users still see the human format
    assert "h100" in parsed["summary"]
    assert "99" in parsed["summary"]


@pytest.mark.asyncio
async def test_gpu_status_caches(mcp_server, mock_env):
    """Two consecutive calls within TTL should hit run_remote once."""
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=GPU_OUTPUT) as rr:
        await mcp_server.call_tool("gpu_status", {})
        await mcp_server.call_tool("gpu_status", {})
    assert rr.call_count == 1


# ── Auto-routing to iris-hi unless at cap ────────────────────────────


def _captured_partition(captured_argv: list[list[str]]) -> str | None:
    """Extract the --partition value from the captured sbatch invocation."""
    for argv in captured_argv:
        if argv and argv[0] == "sbatch" and "--partition" in argv:
            return argv[argv.index("--partition") + 1]
    return None


@pytest.mark.asyncio
async def test_submit_job_auto_routes_to_hi_when_under_cap(mcp_server, mock_env, tmp_path):
    """Caller omits partition → server checks iris-hi count → < cap → uses iris-hi."""
    captured = []

    async def fake_run(cmd, timeout=None):
        captured.append(cmd)
        if cmd[0] == "squeue":
            # 3 user jobs in iris-hi, well under cap of 6
            return "111\n222\n333"
        return "Submitted batch job 42"

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run):
        result = await mcp_server.call_tool(
            "submit_job",
            {"script_content": "#!/bin/bash\necho ok", "working_dir": str(tmp_path)},
        )

    text = result[0][0].text
    assert _captured_partition(captured) == "iris-hi"
    assert "auto-routed to iris-hi" in text
    assert "preemptible" not in text  # iris-hi is not preemptible


@pytest.mark.asyncio
async def test_submit_job_falls_back_when_at_cap(mcp_server, mock_env, tmp_path):
    """Caller omits partition → iris-hi at cap (6 jobs) → falls back to default (iris)."""
    captured = []

    async def fake_run(cmd, timeout=None):
        captured.append(cmd)
        if cmd[0] == "squeue":
            return "\n".join(str(i) for i in range(6))  # 6 jobs == cap
        return "Submitted batch job 99"

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run):
        result = await mcp_server.call_tool(
            "submit_job",
            {"script_content": "#!/bin/bash\necho ok", "working_dir": str(tmp_path)},
        )

    text = result[0][0].text
    assert _captured_partition(captured) == "iris"
    assert "iris-hi at cap" in text
    assert "preemptible" in text  # falling back to iris triggers the warning


@pytest.mark.asyncio
async def test_submit_job_explicit_partition_skips_auto_route(mcp_server, mock_env, tmp_path):
    """Caller pins partition=iris → no squeue probe, request honored verbatim."""
    captured = []

    async def fake_run(cmd, timeout=None):
        captured.append(cmd)
        return "Submitted batch job 7"

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run):
        result = await mcp_server.call_tool(
            "submit_job",
            {
                "script_content": "#!/bin/bash\necho ok",
                "working_dir": str(tmp_path),
                "partition": "iris",
            },
        )

    text = result[0][0].text
    # Exactly one remote call (the sbatch); no squeue probe for auto-route.
    assert len(captured) == 1
    assert captured[0][0] == "sbatch"
    assert _captured_partition(captured) == "iris"
    assert "auto-routed" not in text


# ── Phase 4b: submit_job forwards new sbatch params ──────────────────


@pytest.mark.asyncio
async def test_submit_job_new_params_passed_through(mcp_server, mock_env, tmp_path):
    captured = []

    async def fake_run(cmd, timeout=None):
        captured.append(cmd)
        return "Submitted batch job 42"

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run):
        await mcp_server.call_tool(
            "submit_job",
            {
                "script_content": "#!/bin/bash\necho ok",
                "working_dir": str(tmp_path),
                "dependency": "afterok:100",
                "nodelist": "iris-hgx-2",
                "exclude": "iris-1",
                "constraint": "a100",
            },
        )

    assert captured, "submit_job did not invoke run_remote"
    sbatch_argv = next((c for c in captured if c and c[0] == "sbatch"), None)
    assert sbatch_argv is not None, "sbatch was not invoked"
    argv = sbatch_argv
    assert "--dependency" in argv and argv[argv.index("--dependency") + 1] == "afterok:100"
    assert "--nodelist" in argv and argv[argv.index("--nodelist") + 1] == "iris-hgx-2"
    assert "--exclude" in argv and argv[argv.index("--exclude") + 1] == "iris-1"
    assert "--constraint" in argv and argv[argv.index("--constraint") + 1] == "a100"


# ── Phase 4c: wait_for_job (squeue first, sacct fallback) ────────────


@pytest.mark.asyncio
async def test_wait_for_job_squeue_first_running(mcp_server, mock_env):
    """Job is still running in squeue: poll again (we force a short timeout)."""
    # squeue returns a RUNNING line forever; set a very short timeout so we exit fast.
    async def fake_run(cmd, timeout=None):
        if "squeue" in cmd:
            return "RUNNING|5:00|24:00:00"
        return ""

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await mcp_server.call_tool("wait_for_job", {"job_id": "42", "poll_interval": 5, "timeout": 1})

    data = json.loads(result[0][0].text)
    # Should eventually time out (terminal=False) while state stays RUNNING.
    assert data["terminal"] is False
    assert data.get("State", "").startswith("RUNNING") or data.get("reason") == "timeout"


@pytest.mark.asyncio
async def test_wait_for_job_sacct_fallback_completed(mcp_server, mock_env):
    """squeue empty → fall back to sacct, which reports COMPLETED → terminal."""
    async def fake_run(cmd, timeout=None):
        if "squeue" in cmd:
            return ""
        if "sacct" in cmd:
            return "42|COMPLETED|0:0|0:10|2026-04-23T12:00:00"
        return ""

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await mcp_server.call_tool("wait_for_job", {"job_id": "42", "poll_interval": 5})

    data = json.loads(result[0][0].text)
    assert data["terminal"] is True
    assert data["State"] == "COMPLETED"


@pytest.mark.asyncio
async def test_wait_for_job_unknown_after_n_empty(mcp_server, mock_env):
    """Both squeue and sacct empty for N polls → UNKNOWN, don't poll forever."""
    async def fake_run(cmd, timeout=None):
        return ""  # always empty

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run), \
         patch("asyncio.sleep", new_callable=AsyncMock):
        result = await mcp_server.call_tool("wait_for_job", {"job_id": "42", "poll_interval": 5})

    data = json.loads(result[0][0].text)
    assert data["terminal"] is False
    assert data.get("state", data.get("State")) == "UNKNOWN"


# ── Phase 4d: wandb_for_job — metadata.json primary, log-grep fallback, mtime cache ──


@pytest.mark.asyncio
async def test_wandb_for_job_from_metadata_json(mcp_server, mock_env, tmp_path):
    workdir = tmp_path
    wandb_dir = workdir / "wandb" / "latest-run" / "files"
    wandb_dir.mkdir(parents=True)
    (wandb_dir / "wandb-metadata.json").write_text(json.dumps({"url": "https://wandb.ai/me/proj/runs/abc123"}))

    # sacct returns the workdir; _locate_job_log falls through to the fallback (no log needed when metadata.json wins)
    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=f"{workdir}|"):
        result = await mcp_server.call_tool("wandb_for_job", {"job_id": "42"})

    text = result[0][0].text
    assert "https://wandb.ai/me/proj/runs/abc123" in text


@pytest.mark.asyncio
async def test_wandb_for_job_regex_fallback(mcp_server, mock_env, tmp_path):
    """If no metadata.json, grep the first 64 KB of the job log for a wandb URL."""
    workdir = tmp_path
    slurm_dir = workdir / "slurm"
    slurm_dir.mkdir()
    log_file = slurm_dir / "42.out"
    log_file.write_text(
        "Starting run\n"
        "wandb: View run at https://wandb.ai/me/proj/runs/xyz789\n"
        "Epoch 1 ...\n"
    )

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=f"{workdir}|"):
        result = await mcp_server.call_tool("wandb_for_job", {"job_id": "42"})
    assert "https://wandb.ai/me/proj/runs/xyz789" in result[0][0].text


@pytest.mark.asyncio
async def test_wandb_for_job_caches_by_mtime(mcp_server, mock_env, tmp_path):
    workdir = tmp_path
    slurm_dir = workdir / "slurm"
    slurm_dir.mkdir()
    log_file = slurm_dir / "42.out"
    log_file.write_text("wandb: View run at https://wandb.ai/a/b/runs/c\n")

    opens = []
    real_open = open

    def counting_open(path, *a, **kw):
        opens.append(str(path))
        return real_open(path, *a, **kw)

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", new_callable=AsyncMock, return_value=f"{workdir}|"), \
         patch("builtins.open", side_effect=counting_open):
        await mcp_server.call_tool("wandb_for_job", {"job_id": "42"})
        await mcp_server.call_tool("wandb_for_job", {"job_id": "42"})

    # The log file should be opened at most once (cache keyed on (path, mtime))
    log_opens = [p for p in opens if p.endswith("42.out")]
    assert len(log_opens) <= 1


# ── Phase 4e: quota_check dispatches on filesystem type ──────────────


@pytest.mark.asyncio
async def test_quota_check_dispatches_by_fs_type(mcp_server, mock_env):
    """Detect FS type via `stat -f -c %T` then dispatch to the right quota command."""
    calls = []

    async def fake_run(cmd, timeout=None):
        calls.append(cmd)
        # first call: FS type probe → return "nfs"
        if "stat" in cmd and "-f" in cmd:
            return "nfs"
        # home quota (always)
        if cmd[:2] == ["quota", "-s"]:
            return "Disk quotas for user testuser\nhome: 5G  (limit 10G)"
        # lfs should NOT be called on an nfs FS
        if cmd and cmd[0] == "lfs":
            raise AssertionError("lfs quota should not run on non-Lustre FS")
        return ""

    with patch("slurm_mcp.slurm.SlurmClient.run_remote", side_effect=fake_run):
        result = await mcp_server.call_tool("quota_check", {})

    text = result[0][0].text
    # structured output expected
    parsed = json.loads(text)
    assert "home" in parsed
    assert "iris" in parsed
    # evidence the FS-type probe ran before the quota call
    probe_idx = next((i for i, c in enumerate(calls) if "stat" in c and "-f" in c), None)
    assert probe_idx is not None and probe_idx == 0
