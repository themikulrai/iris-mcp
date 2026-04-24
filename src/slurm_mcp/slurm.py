"""SSH-based Slurm command executor.

Runs commands on the login node via SSH, with Kerberos-aware error handling
and OpenSSH ControlMaster-based connection reuse.
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
from pathlib import Path

from mcp.server.fastmcp.exceptions import ToolError

from .config import SlurmConfig


GPU_VRAM_GB: dict[str, int] = {
    "h100": 80,
    "h200": 141,
    "a100": 80,
    "a100_40": 40,
    "a100_80": 80,
    "l40s": 48,
    "a6000": 48,
    "titanrtx": 24,
}


_SGPU_PARTITION_HEADER_RE = re.compile(r"^\s*(\S+)\s+GPU\s+Status", re.IGNORECASE)
_SGPU_TOTAL_RE = re.compile(r"There\s+are\s+a\s+total\s+of\s+(\d+)\s+gpus?\s*\[(\w+)\]", re.IGNORECASE)
_SGPU_TYPE_RE = re.compile(r"^\s*(\d+)\s+([a-zA-Z0-9_\-]+)\s+gpus?\s*$", re.IGNORECASE)


def parse_sgpu(text: str) -> dict:
    """Parse `sgpu` text output into a structured dict.

    Lines that don't match are ignored; the original text is always preserved
    under the `raw` key as a fallback.
    """
    partition: str | None = None
    total: int | None = None
    state: str | None = None
    types: list[dict] = []
    for line in text.splitlines():
        if m := _SGPU_PARTITION_HEADER_RE.match(line):
            partition = m.group(1)
            continue
        if m := _SGPU_TOTAL_RE.search(line):
            total = int(m.group(1))
            state = m.group(2)
            continue
        if m := _SGPU_TYPE_RE.match(line):
            count = int(m.group(1))
            name = m.group(2).lower()
            types.append({"name": name, "count": count, "vram_gb": GPU_VRAM_GB.get(name)})
    return {
        "partition": partition,
        "total": total,
        "state": state,
        "types": types,
        "raw": text,
    }


def _ensure_control_dir(control_path: str) -> None:
    """Create the ControlMaster socket directory with safe perms if missing."""
    expanded = os.path.expanduser(control_path)
    parent = os.path.dirname(expanded)
    if parent:
        Path(parent).mkdir(mode=0o700, parents=True, exist_ok=True)


class SlurmClient:
    def __init__(self, config: SlurmConfig) -> None:
        self.config = config
        # Prepare the ControlMaster socket dir once per client.
        try:
            _ensure_control_dir(config.ssh_control_path)
        except OSError:
            # Non-fatal: SSH will still work, just without multiplexing.
            pass

    async def run_remote(self, cmd: list[str], timeout: int | None = None) -> str:
        """Run a command on the login node via SSH.

        `cmd` must be a list of argv tokens; the caller supplies structure, we
        quote. String commands are rejected (historical str overload removed).
        Returns stdout on success. Raises ToolError with actionable messages on
        failure. On client cancellation the SSH subprocess is killed to avoid
        orphans.
        """
        if not isinstance(cmd, list):
            raise TypeError(f"run_remote requires a list of argv tokens, got {type(cmd).__name__}")

        cmd_str = " ".join(shlex.quote(c) for c in cmd)

        ssh_cmd = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.config.ssh_timeout}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={self.config.ssh_control_path}",
            "-o", f"ControlPersist={self.config.ssh_control_persist}",
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
        except asyncio.CancelledError:
            # Client disconnect or task cancellation — don't orphan ssh.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            raise

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
        hostname = await self.run_remote(["hostname"], timeout=5)
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
        dependency: str | None = None,
        nodelist: str | None = None,
        exclude: str | None = None,
        constraint: str | None = None,
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

        if dependency:
            args += ["--dependency", dependency]
        if nodelist:
            args += ["--nodelist", nodelist]
        if exclude:
            args += ["--exclude", exclude]
        if constraint:
            args += ["--constraint", constraint]

        if extra_args:
            args += shlex.split(extra_args)

        return args
