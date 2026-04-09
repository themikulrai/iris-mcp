"""SSH-based Slurm command executor.

Runs commands on the login node via SSH, with Kerberos-aware error handling.
"""

from __future__ import annotations

import asyncio
import shlex

from mcp.server.fastmcp.exceptions import ToolError

from .config import SlurmConfig


class SlurmClient:
    def __init__(self, config: SlurmConfig) -> None:
        self.config = config

    async def run_remote(self, cmd: str | list[str], timeout: int | None = None) -> str:
        """Run a command on the login node via SSH.

        Returns stdout on success. Raises ToolError with actionable messages on failure.
        """
        if isinstance(cmd, list):
            cmd_str = " ".join(shlex.quote(c) for c in cmd)
        else:
            cmd_str = cmd

        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.config.ssh_timeout}",
            self.config.login_node,
            cmd_str,
        ]

        proc = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        effective_timeout = timeout or self.config.command_timeout
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), effective_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise ToolError(f"Command timed out after {effective_timeout}s: {cmd_str}")

        stdout_str = stdout.decode().strip()
        stderr_str = stderr.decode().strip()

        if proc.returncode != 0:
            if any(hint in stderr_str for hint in ("Permission denied", "GSSAPI", "No Kerberos", "credential")):
                raise ToolError(
                    f"SSH authentication failed — Kerberos ticket likely expired.\n"
                    f"Fix: kinit {self.config.username}@CS.STANFORD.EDU"
                )
            raise ToolError(f"Command failed (exit {proc.returncode}): {stderr_str or stdout_str}")

        return stdout_str

    async def check_connection(self) -> str:
        """Verify SSH + Kerberos connectivity to the login node."""
        hostname = await self.run_remote("hostname", timeout=5)
        return hostname

    def build_sbatch_args(
        self,
        *,
        job_name: str | None = None,
        partition: str | None = None,
        account: str | None = None,
        gpus: int | None = None,
        gpu_type: str | None = None,
        nodes: int = 1,
        cpus: int | None = None,
        mem: str | None = None,
        time_limit: str | None = None,
        output_pattern: str | None = None,
        working_dir: str | None = None,
        array: str | None = None,
        extra_args: str | None = None,
    ) -> list[str]:
        """Build sbatch argument list with config defaults."""
        cfg = self.config
        args: list[str] = ["sbatch"]

        if job_name:
            args += ["--job-name", job_name]
        args += ["--partition", partition or cfg.default_partition]
        args += ["--account", account or cfg.default_account]
        args += ["--nodes", str(nodes)]
        args += ["--cpus-per-task", str(cpus or cfg.default_cpus)]
        args += ["--mem", mem or cfg.default_mem]
        args += ["--time", time_limit or cfg.default_time]

        gres = f"gpu:{gpu_type}:{gpus or cfg.default_gpus}" if gpu_type else f"gpu:{gpus or cfg.default_gpus}"
        args += ["--gres", gres]

        if output_pattern:
            args += ["--output", output_pattern]

        if working_dir:
            args += ["--chdir", working_dir]

        if array:
            args += ["--array", array]

        if extra_args:
            args += shlex.split(extra_args)

        return args
