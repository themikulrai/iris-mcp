"""IRIS MCP Server — Slurm tools for HPC clusters."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError

from .config import SlurmConfig
from .slurm import SlurmClient

_JOB_ID_RE = re.compile(r"^[\d,_\-\%]+$")


def create_server() -> FastMCP:
    config = SlurmConfig()
    client = SlurmClient(config)
    mcp = FastMCP("iris-mcp")

    # ── Auth ──────────────────────────────────────────────────────────

    @mcp.tool()
    async def check_auth() -> str:
        """Verify SSH and Kerberos connectivity to the login node.

        Call this first if other tools fail with authentication errors.
        """
        hostname = await client.check_connection()
        return f"Connected to {hostname} as {config.username}"

    # ── Job submission ────────────────────────────────────────────────

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
        extra_args: Optional[str] = None,
    ) -> str:
        """Submit a Slurm batch job via sbatch.

        Provide either script_path (path to an existing .sh file on the shared filesystem)
        or script_content (inline script text that will be written to a temp file).

        Uses configured defaults for partition, account, GPUs, memory, and time if not specified.
        Automatically creates the output directory before submission.

        Set array (e.g. '0-7' or '0-31%8') for job array submissions.
        """
        if not script_path and not script_content:
            raise ToolError("Provide either script_path or script_content.")

        work_dir = working_dir or config.resolved_working_dir
        output_dir = config.output_dir
        output_pattern = f"{output_dir}/%j.out"

        # Ensure output directory exists (on shared filesystem, so local mkdir works)
        out_path = Path(work_dir) / output_dir
        out_path.mkdir(parents=True, exist_ok=True)

        # Write inline script to a temp file on shared filesystem
        if script_content:
            if not script_content.startswith("#!"):
                script_content = "#!/bin/bash\n" + script_content
            tmp_dir = Path(work_dir) / ".slurm_scripts"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            import tempfile
            fd, tmp_path = tempfile.mkstemp(suffix=".sh", dir=str(tmp_dir), prefix="job_")
            with os.fdopen(fd, "w") as f:
                f.write(script_content)
            os.chmod(tmp_path, 0o755)
            script_path = tmp_path

        args = client.build_sbatch_args(
            job_name=job_name,
            partition=partition,
            gpus=gpus,
            gpu_type=gpu_type,
            nodes=nodes,
            cpus=cpus,
            mem=mem,
            time_limit=time_limit,
            output_pattern=output_pattern,
            working_dir=work_dir,
            array=array,
            extra_args=extra_args,
        )
        args.append(script_path)

        stdout = await client.run_remote(args)

        msg = f"Job submitted: {stdout}"
        if script_content:
            msg += f"\nScript saved to: {script_path}"
        if partition and "interactive" not in partition and partition == "iris":
            msg += "\nNote: iris partition is preemptible — ensure your job supports checkpointing."
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
            # Look up working directory from sacct
            cmd = ["sacct", "-j", job_id, "--format=WorkDir", "--parsable2", "--noheader"]
            stdout = await client.run_remote(cmd)
            if not stdout:
                raise ToolError(f"Could not find working directory for job {job_id}.")

            work_dir = stdout.splitlines()[0].split("|")[0].strip()
            output_dir = config.output_dir
            candidates = [
                os.path.join(work_dir, output_dir, f"{job_id}.out"),
                os.path.join(work_dir, f"slurm-{job_id}.out"),
                os.path.join(work_dir, f"{job_id}.out"),
            ]

            found = None
            for candidate in candidates:
                if os.path.exists(candidate):
                    found = candidate
                    break
            if not found:
                raise ToolError(f"No output file found for job {job_id}. Tried:\n" + "\n".join(candidates))
            file_path = found

        if not os.path.exists(file_path):
            raise ToolError(f"File not found: {file_path}")

        with open(file_path, errors="replace") as f:
            all_lines = f.readlines()

        tail = all_lines[-lines:]
        header = f"=== {file_path} (last {len(tail)} of {len(all_lines)} lines) ==="
        return header + "\n" + "".join(tail)

    # ── Cluster info ──────────────────────────────────────────────────

    @mcp.tool()
    async def gpu_status(partition: Optional[str] = None) -> str:
        """Show GPU availability across the cluster.

        Uses the cluster's GPU status command (sgpu by default).
        """
        if partition:
            cmd = config.gpu_command.replace("-p iris", f"-p {partition}")
        else:
            cmd = config.gpu_command
        return await client.run_remote(cmd)

    @mcp.tool()
    async def cluster_info(partition: Optional[str] = None) -> str:
        """Show Slurm cluster overview: partitions, node states, and GPU resources."""
        cmd = "sinfo --format='%P|%a|%l|%D|%T|%G' --noheader"
        if partition:
            cmd += f" -p {partition}"

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

    # ── Environment ───────────────────────────────────────────────────

    @mcp.tool()
    async def env_list() -> str:
        """List available conda/mamba environments."""
        conda_bin = os.path.join(config.resolved_conda_dir, "bin", "conda")
        if not os.path.exists(conda_bin):
            raise ToolError(f"Conda not found at {conda_bin}. Set SLURM_MCP_CONDA_DIR.")

        result = subprocess.run(
            [conda_bin, "env", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
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

    # ── Disk usage ────────────────────────────────────────────────────

    @mcp.tool()
    async def disk_usage(path: Optional[str] = None) -> str:
        """Check disk usage for a directory.

        Defaults to the shared working directory. Also shows home directory usage
        as a reminder to keep it small.
        """
        results: list[str] = []
        target = path or config.resolved_working_dir

        # Check target
        du = subprocess.run(
            ["du", "-sh", target],
            capture_output=True, text=True, timeout=30,
        )
        if du.returncode == 0 and du.stdout.strip():
            results.append(f"{target}: {du.stdout.strip().split()[0]}")

        # Also check home dir if different from target
        home = config.resolved_home_dir
        if home != target and os.path.exists(home):
            du = subprocess.run(
                ["du", "-sh", home],
                capture_output=True, text=True, timeout=30,
            )
            if du.returncode == 0 and du.stdout.strip():
                results.append(f"{home}: {du.stdout.strip().split()[0]} (keep this small!)")

        return "\n".join(results) if results else "Could not determine disk usage."

    return mcp
