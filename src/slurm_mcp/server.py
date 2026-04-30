"""IRIS MCP Server — Slurm tools for HPC clusters."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .config import SlurmConfig
from .slurm import SlurmClient, parse_sgpu

_JOB_ID_RE = re.compile(r"^\d+(_(\d+|\*))?(-\d+)?(%\d+)?(,\d+(_(\d+|\*))?)*$")
_PARTITION_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
_WANDB_URL_RE = re.compile(r"https://wandb\.ai/[^/\s]+/[^/\s]+/runs/[^\s)>\]]+")

_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "PREEMPTED",
}

_GPU_STATUS_TTL_SECONDS = 15.0


def _validate_partition(name: str) -> None:
    if not _PARTITION_RE.match(name):
        raise ToolError(f"Invalid partition name: {name!r}")


def _write_inline_script(tmp_dir: Path, content: str) -> str:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".sh", dir=str(tmp_dir), prefix="job_")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    os.chmod(path, 0o755)
    return path


def create_server() -> FastMCP:
    config = SlurmConfig()
    client = SlurmClient(config)
    mcp = FastMCP("iris-mcp")

    # gpu_status cache: {partition_key: (timestamp, rendered_json_text)}
    gpu_cache: dict[str, tuple[float, str]] = {}

    # wandb_for_job cache: {log_path: (mtime, url_or_none)}
    wandb_cache: dict[str, tuple[float, Optional[str]]] = {}

    # ── Helpers (close over client/config) ────────────────────────────

    async def _locate_job_log(job_id: str) -> str:
        cmd = ["sacct", "-j", job_id, "--format=WorkDir", "--parsable2", "--noheader"]
        stdout = await client.run_remote(cmd)
        if not stdout:
            raise ToolError(f"Could not find working directory for job {job_id}.")
        work_dir = stdout.splitlines()[0].split("|")[0].strip()

        def _find() -> Optional[str]:
            candidates = [
                os.path.join(work_dir, config.output_dir, f"{job_id}.out"),
                os.path.join(work_dir, f"slurm-{job_id}.out"),
                os.path.join(work_dir, f"{job_id}.out"),
            ]
            for c in candidates:
                if os.path.exists(c):
                    return c
            return None

        found = await asyncio.to_thread(_find)
        if not found:
            raise ToolError(f"No output file found for job {job_id} under {work_dir}.")
        return found

    async def _fetch_job_state(job_id: str) -> dict[str, str]:
        """squeue-first (fast, authoritative while job is active),
        sacct-fallback (accounting, once job has left the queue)."""
        squeue_cmd = ["squeue", "-j", job_id, "-h", "-o", "%T|%M|%l"]
        try:
            squeue_out = await client.run_remote(squeue_cmd)
        except ToolError:
            squeue_out = ""
        if squeue_out.strip():
            parts = squeue_out.splitlines()[0].split("|")
            if parts and parts[0].strip():
                out = {"State": parts[0].strip(), "source": "squeue"}
                if len(parts) > 1:
                    out["Elapsed"] = parts[1].strip()
                if len(parts) > 2:
                    out["TimeLimit"] = parts[2].strip()
                return out

        fmt = "JobID,State,ExitCode,Elapsed,End"
        sacct_cmd = [
            "sacct", "-j", job_id,
            f"--format={fmt}",
            "--parsable2", "--noheader", "--allocations",
        ]
        try:
            sacct_out = await client.run_remote(sacct_cmd)
        except ToolError:
            sacct_out = ""
        if sacct_out.strip():
            fields = fmt.split(",")
            for line in sacct_out.splitlines():
                cols = [c.strip() for c in line.split("|")]
                if cols and cols[0]:
                    return dict(zip(fields, cols)) | {"source": "sacct"}
        return {}

    # ── Auth ──────────────────────────────────────────────────────────

    @mcp.tool()
    async def check_auth() -> str:
        """Verify SSH and Kerberos connectivity to the login node.

        Call this first if other tools fail with authentication errors.
        """
        hostname = await client.check_connection()
        return f"Connected to {hostname} as {config.username}"

    # ── Job submission ────────────────────────────────────────────────

    async def _auto_select_partition() -> tuple[str, Optional[str]]:
        """Pick a partition when the caller did not specify one.

        When `auto_route_to_hi` is enabled, prefer the high-priority partition
        (`auto_route_target`) until the user already has `auto_route_cap`
        jobs there (running + pending). Otherwise fall back to
        `default_partition` so the new job can start immediately.

        Returns (partition, note) where `note` is a human-readable explanation
        appended to the submission response (None when no auto-routing fired).
        """
        if not config.auto_route_to_hi:
            return config.default_partition, None
        target = config.auto_route_target
        cap = config.auto_route_cap
        cmd = [
            "squeue",
            "--noheader",
            "--user", config.username,
            "--partition", target,
            "-o", "%i",
        ]
        try:
            out = await client.run_remote(cmd)
        except Exception:
            return config.default_partition, None
        count = sum(1 for line in out.splitlines() if line.strip())
        if count < cap:
            return target, f"auto-routed to {target} ({count}/{cap} of your jobs already there)"
        return config.default_partition, f"{target} at cap ({count}/{cap}); using {config.default_partition}"

    @mcp.tool()
    async def submit_job(
        script_path: Optional[str] = None,
        script_content: Optional[str] = None,
        job_name: Optional[str] = None,
        partition: Optional[str] = None,
        gpus: Optional[int] = None,
        gpu_type: Optional[str] = None,
        nodes: int = 1,
        cpus: Optional[int] = None,
        mem: Optional[str] = None,
        time_limit: Optional[str] = None,
        working_dir: Optional[str] = None,
        array: Optional[str] = None,
        dependency: Optional[str] = None,
        nodelist: Optional[str] = None,
        exclude: Optional[str] = None,
        constraint: Optional[str] = None,
        extra_args: Optional[str] = None,
        script_args: Optional[str] = None,
    ) -> str:
        """Submit a Slurm batch job via sbatch.

        Provide either script_path (existing .sh on the shared filesystem) or
        script_content (inline text, written to a temp file on the shared FS).

        Uses configured defaults for partition, account, GPUs, memory, and time
        when not specified. Automatically creates the output directory before
        submission.

        `extra_args` passes additional flags to sbatch itself (e.g. '--nice=10000').
        `script_args` passes arguments to the script after the script path
        (e.g. '--num_train_steps=20 --lr=1e-4').

        `dependency` accepts standard Slurm syntax (e.g. 'afterok:12345',
        'afternotok:12345', 'afterany:12345:12346', 'singleton').
        `nodelist`/`exclude` take comma-separated node lists. `constraint` takes
        a Slurm feature expression (e.g. 'a100' or 'a100&ib').
        """
        if not script_path and not script_content:
            raise ToolError("Provide either script_path or script_content.")

        # For script_path jobs the script already contains #SBATCH headers, so
        # command-line resource flags must NOT be emitted unless the caller
        # explicitly provides them (sbatch CLI flags always override #SBATCH
        # directives). For inline script_content there are no headers, so we
        # fall back to config defaults.
        inline = bool(script_content)

        work_dir = working_dir or config.resolved_working_dir
        output_dir = config.output_dir
        output_pattern = f"{output_dir}/%j.out"

        out_path = Path(work_dir) / output_dir
        await asyncio.to_thread(out_path.mkdir, parents=True, exist_ok=True)

        if script_content:
            if not script_content.startswith("#!"):
                script_content = "#!/bin/bash\n" + script_content
            tmp_dir = Path(work_dir) / ".slurm_scripts"
            script_path = await asyncio.to_thread(_write_inline_script, tmp_dir, script_content)

        # Auto-routing kicks in only when caller did not pin a partition.
        auto_note: Optional[str] = None
        if partition is None:
            partition, auto_note = await _auto_select_partition()

        args = client.build_sbatch_args(
            job_name=job_name,
            partition=partition,
            gpus=gpus if gpus is not None else (config.default_gpus if inline else None),
            gpu_type=gpu_type,
            nodes=nodes,
            cpus=cpus if cpus is not None else (config.default_cpus if inline else None),
            mem=mem if mem is not None else (config.default_mem if inline else None),
            time_limit=time_limit if time_limit is not None else (config.default_time if inline else None),
            output_pattern=output_pattern,
            working_dir=work_dir,
            array=array,
            dependency=dependency,
            nodelist=nodelist,
            exclude=exclude,
            constraint=constraint,
            extra_args=extra_args,
        )
        args.append(script_path)
        if script_args:
            args += shlex.split(script_args)

        stdout = await client.run_remote(args)

        msg = f"Job submitted: {stdout}"
        if script_content:
            msg += f"\nScript saved to: {script_path}"

        if auto_note:
            msg += f"\nPartition: {auto_note}"

        effective = partition or config.default_partition
        if not (effective.endswith("-hi") or effective.endswith("-interactive")):
            msg += (
                f"\nNote: {effective} partition is preemptible — "
                "ensure your job supports checkpointing."
            )
        return msg

    # ── Job monitoring ────────────────────────────────────────────────

    @mcp.tool()
    async def list_jobs(
        user: Optional[str] = None,
        partition: Optional[str] = None,
        state: Optional[str] = None,
    ) -> str:
        """List Slurm jobs for the current user via squeue.

        Optionally filter by partition or state (RUNNING, PENDING, etc.).
        """
        target_user = user or config.username
        cmd = [
            "squeue",
            "--format=%i|%j|%P|%T|%M|%l|%D|%C|%b|%R",
            "--noheader",
            "--user", target_user,
        ]
        if partition:
            _validate_partition(partition)
            cmd += ["--partition", partition]
        if state:
            cmd += ["--state", state]

        stdout = await client.run_remote(cmd)
        if not stdout:
            return f"No jobs found for user '{target_user}'."

        headers = "JobID | Name | Partition | State | Elapsed | TimeLimit | Nodes | CPUs | GPUs | Reason/NodeList"
        lines = [headers, "-" * 110]
        for line in stdout.splitlines():
            if line.strip():
                lines.append(" | ".join(f.strip() for f in line.split("|")))
        return "\n".join(lines)

    @mcp.tool()
    async def cancel_job(job_id: str) -> str:
        """Cancel one or more Slurm jobs.

        job_id can be a single ID (e.g. '12345'), comma-separated ('12345,12346'),
        or an array range ('12345_0-7').
        """
        if not _JOB_ID_RE.match(job_id):
            raise ToolError(f"Invalid job ID format: {job_id}")

        await client.run_remote(["scancel", job_id])
        return f"Job {job_id} cancelled."

    @mcp.tool()
    async def job_status(job_id: str, detailed: bool = False) -> str:
        """Get detailed status for a job via sacct. Works for running and completed jobs."""
        if not _JOB_ID_RE.match(job_id.split(".")[0]):
            raise ToolError(f"Invalid job ID format: {job_id}")

        fmt = "JobID,JobName,Partition,State,ExitCode,Elapsed,TimelimitRaw,AllocCPUS,AllocTRES,MaxRSS,Start,End,WorkDir"
        if detailed:
            fmt += ",Submit,Eligible,NNodes,NTasks,ReqMem,AveRSS,MaxVMSize"

        cmd = ["sacct", "-j", job_id, f"--format={fmt}", "--parsable2", "--noheader"]
        stdout = await client.run_remote(cmd)

        if not stdout:
            return f"No information found for job {job_id}."

        fields = fmt.split(",")
        output_lines = [f"Job {job_id} Status:"]
        for line in stdout.splitlines():
            if not line.strip():
                continue
            values = line.split("|")
            output_lines.append("-" * 40)
            for i, field in enumerate(fields):
                if i < len(values) and values[i]:
                    output_lines.append(f"  {field}: {values[i]}")
        return "\n".join(output_lines)

    @mcp.tool()
    async def wait_for_job(
        job_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
    ) -> str:
        """Block until a Slurm job reaches a terminal state (or timeout).

        Polls squeue first (fast, ~50ms) and falls back to sacct once the job
        has left the queue. If both are empty for 6 consecutive polls, returns
        state=UNKNOWN rather than waiting forever. poll_interval is clamped to
        a 5-second floor.
        """
        if not _JOB_ID_RE.match(job_id.split(".")[0]):
            raise ToolError(f"Invalid job ID format: {job_id}")
        if timeout is not None and timeout <= 0:
            raise ToolError("timeout must be positive")
        poll_interval = max(5, poll_interval)

        MAX_EMPTY = 6
        start = time.monotonic()
        empty_polls = 0
        last_state: dict[str, str] = {}

        while True:
            state = await _fetch_job_state(job_id)

            if state:
                empty_polls = 0
                last_state = state
                raw_state = state.get("State", "").split()[0] if state.get("State") else ""
                if raw_state in _TERMINAL_STATES:
                    return json.dumps({"job_id": job_id, **state, "terminal": True}, indent=2)
            else:
                empty_polls += 1
                if empty_polls >= MAX_EMPTY:
                    return json.dumps({
                        "job_id": job_id,
                        "state": "UNKNOWN",
                        "terminal": False,
                        "reason": "not visible to scheduler or accounting",
                    }, indent=2)

            if timeout is not None and (time.monotonic() - start) >= timeout:
                return json.dumps({
                    "job_id": job_id,
                    **last_state,
                    "terminal": False,
                    "reason": "timeout",
                }, indent=2)

            await asyncio.sleep(poll_interval)

    @mcp.tool()
    async def tail_output(
        job_id: Optional[str] = None,
        file_path: Optional[str] = None,
        lines: int = 50,
    ) -> str:
        """Read the output log of a Slurm job.

        Provide job_id to auto-locate the log file, or file_path for a direct path.
        Reads from the shared filesystem (no SSH needed for file access).
        """
        if not job_id and not file_path:
            raise ToolError("Provide either job_id or file_path.")

        if job_id and not file_path:
            file_path = await _locate_job_log(job_id)

        def _tail() -> tuple[list[str], int]:
            if not os.path.exists(file_path):
                raise ToolError(f"File not found: {file_path}")
            with open(file_path, errors="replace") as f:
                all_lines = f.readlines()
            return all_lines[-lines:], len(all_lines)

        tail, total = await asyncio.to_thread(_tail)
        header = f"=== {file_path} (last {len(tail)} of {total} lines) ==="
        return header + "\n" + "".join(tail)

    # ── Cluster info ──────────────────────────────────────────────────

    @mcp.tool()
    async def gpu_status(partition: Optional[str] = None) -> str:
        """Show GPU availability across the cluster as structured JSON.

        Parses `sgpu` output into {partition, total, state, types[name,count,vram_gb]}.
        VRAM is looked up from a static map (h100=80, h200=141, a100=80, l40s=48,
        a6000=48, titanrtx=24). The original sgpu text is preserved under `summary`
        for human display. Cached for 15 seconds.
        """
        key = partition or "__default__"
        now = time.monotonic()
        entry = gpu_cache.get(key)
        if entry and (now - entry[0]) < _GPU_STATUS_TTL_SECONDS:
            return entry[1]

        base_cmd = shlex.split(config.gpu_command)
        if partition:
            _validate_partition(partition)
            if "-p" in base_cmd:
                idx = base_cmd.index("-p")
                base_cmd[idx + 1] = partition
            else:
                base_cmd += ["-p", partition]

        raw = await client.run_remote(base_cmd)
        parsed = parse_sgpu(raw)
        rendered = json.dumps({
            **parsed,
            "summary": raw,
            "source": "sgpu+static_map",
        }, indent=2)
        gpu_cache[key] = (now, rendered)
        return rendered

    @mcp.tool()
    async def cluster_info(partition: Optional[str] = None) -> str:
        """Show Slurm cluster overview: partitions, node states, and GPU resources."""
        cmd = ["sinfo", "--format=%P|%a|%l|%D|%T|%G", "--noheader"]
        if partition:
            _validate_partition(partition)
            cmd += ["-p", partition]

        stdout = await client.run_remote(cmd)
        if not stdout:
            return "Could not retrieve cluster info."

        lines = [
            "Partition | Avail | TimeLimit | Nodes | State | GPUs",
            "-" * 80,
        ]
        for line in stdout.splitlines():
            if line.strip():
                lines.append(" | ".join(f.strip() for f in line.split("|")))
        return "\n".join(lines)

    # ── Interactive sessions ──────────────────────────────────────────

    @mcp.tool()
    async def interactive_session(
        partition: str = "iris-interactive",
        gpus: int = 1,
        gpu_type: Optional[str] = None,
        mem: str = "32GB",
        time: str = "4:00:00",
    ) -> str:
        """Generate the commands to start an interactive Slurm session.

        Returns the exact commands to run — does not start the session itself,
        since interactive sessions require a TTY on the login node.
        """
        gres = f"gpu:{gpu_type}:{gpus}" if gpu_type else f"gpu:{gpus}"
        srun_cmd = (
            f"srun -p {partition} --mem={mem} --gres={gres} "
            f"--account={config.default_account} --time={time} --pty bash"
        )

        return "\n".join([
            "Run these commands to start an interactive session:",
            "",
            f"  ssh {config.login_node}",
            "  tmux                    # so session survives SSH drops",
            f"  {srun_cmd}",
            "",
            "After landing on the compute node:",
            "  hostname                # note the node name (e.g. iris-hgx-2)",
            "",
            "To connect your IDE to the compute node:",
            "  Set SSH Host to the node name (e.g. iris-hgx-2)",
            f"  Ensure your SSH config has: Host iris-hgx-* with ProxyJump {config.login_node}",
            "",
            "To reconnect after SSH drop:",
            f"  ssh {config.login_node}",
            "  tmux attach",
        ])

    # ── Wandb cross-reference ─────────────────────────────────────────

    @mcp.tool()
    async def wandb_for_job(job_id: str) -> str:
        """Find the wandb run URL for a Slurm job.

        Prefers ${workdir}/wandb/latest-run/files/wandb-metadata.json (structured).
        Falls back to regex-scanning the first 64 KB of the job log. Caches
        results keyed on (log_path, mtime).
        """
        if not _JOB_ID_RE.match(job_id.split(".")[0]):
            raise ToolError(f"Invalid job ID format: {job_id}")

        # Locate workdir via sacct (reuse the log-locator's first step inline —
        # we need workdir specifically for the wandb/ subdirectory).
        sacct_cmd = ["sacct", "-j", job_id, "--format=WorkDir", "--parsable2", "--noheader"]
        stdout = await client.run_remote(sacct_cmd)
        if not stdout.strip():
            raise ToolError(f"Could not find working directory for job {job_id}.")
        work_dir = stdout.splitlines()[0].split("|")[0].strip()

        metadata_path = os.path.join(work_dir, "wandb", "latest-run", "files", "wandb-metadata.json")

        def _read_metadata() -> Optional[str]:
            if not os.path.exists(metadata_path):
                return None
            try:
                with open(metadata_path) as f:
                    data = json.load(f)
                url = data.get("url")
                return url if isinstance(url, str) else None
            except (OSError, ValueError):
                return None

        url = await asyncio.to_thread(_read_metadata)
        if url:
            return url

        # Fallback: regex-scan first 64 KB of the log file.
        log_path = await _locate_job_log(job_id)

        try:
            mtime = await asyncio.to_thread(os.path.getmtime, log_path)
        except OSError:
            mtime = 0.0

        cached = wandb_cache.get(log_path)
        if cached and cached[0] == mtime:
            return cached[1] or "not found"

        def _grep() -> Optional[str]:
            with open(log_path, "rb") as f:
                blob = f.read(65536)
            text = blob.decode(errors="replace")
            m = _WANDB_URL_RE.search(text)
            return m.group(0) if m else None

        found = await asyncio.to_thread(_grep)
        wandb_cache[log_path] = (mtime, found)
        return found or "not found"

    # ── Environment ───────────────────────────────────────────────────

    @mcp.tool()
    async def env_list() -> str:
        """List available conda/mamba environments."""
        conda_bin = os.path.join(config.resolved_conda_dir, "bin", "conda")
        if not os.path.exists(conda_bin):
            raise ToolError(f"Conda not found at {conda_bin}. Set SLURM_MCP_CONDA_DIR.")

        def _run() -> subprocess.CompletedProcess:
            return subprocess.run(
                [conda_bin, "env", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

        result = await asyncio.to_thread(_run)
        if result.returncode != 0:
            raise ToolError(f"conda env list failed: {result.stderr}")

        data = json.loads(result.stdout)
        envs = data.get("envs", [])
        if not envs:
            return "No conda environments found."

        lines = ["Conda environments:"]
        for env_path in envs:
            name = os.path.basename(env_path) if env_path != config.resolved_conda_dir else "base"
            lines.append(f"  {name:20s}  {env_path}")
        return "\n".join(lines)

    # ── Disk / quota ──────────────────────────────────────────────────

    @mcp.tool()
    async def disk_usage(path: Optional[str] = None) -> str:
        """Check disk usage for a directory.

        Defaults to the shared working directory. Also shows home directory usage
        as a reminder to keep it small.
        """
        target = path or config.resolved_working_dir

        def _du(p: str) -> Optional[str]:
            r = subprocess.run(
                ["du", "-sh", p],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip().split()[0]
            return None

        results: list[str] = []
        target_size = await asyncio.to_thread(_du, target)
        if target_size:
            results.append(f"{target}: {target_size}")

        home = config.resolved_home_dir
        if home != target and os.path.exists(home):
            home_size = await asyncio.to_thread(_du, home)
            if home_size:
                results.append(f"{home}: {home_size} (keep this small!)")

        return "\n".join(results) if results else "Could not determine disk usage."

    @mcp.tool()
    async def quota_check() -> str:
        """Report filesystem quota for home and shared working directories.

        Probes filesystem type via `stat -f -c %T` first, then dispatches to the
        correct quota command (Lustre → lfs quota, BeeGFS → beegfs-ctl,
        NFS/other → df). Home directory always uses `quota -s`.
        """
        work_dir = config.resolved_working_dir
        try:
            fs_type = (await client.run_remote(["stat", "-f", "-c", "%T", work_dir])).strip().lower()
        except ToolError as e:
            fs_type = f"<error: {e}>"

        async def _safe(cmd: list[str]) -> str:
            try:
                return await client.run_remote(cmd)
            except ToolError as e:
                return f"<error: {e}>"

        home_raw = await _safe(["quota", "-s"])

        if "lustre" in fs_type:
            iris_raw = await _safe(["lfs", "quota", "-hu", config.username, work_dir])
            iris_source = "lfs"
        elif "beegfs" in fs_type:
            iris_raw = await _safe(["beegfs-ctl", "--getquota", "--uid", config.username])
            iris_source = "beegfs-ctl"
        else:
            iris_raw = await _safe(["df", "-h", work_dir])
            iris_source = "df"

        return json.dumps({
            "fs_type": fs_type,
            "home": {"source": "quota -s", "raw": home_raw},
            "iris": {"source": iris_source, "path": work_dir, "raw": iris_raw},
        }, indent=2)

    return mcp
