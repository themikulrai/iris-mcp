"""Configuration for the Slurm MCP server.

All settings are loaded from environment variables with the SLURM_MCP_ prefix,
or from a .env file in the working directory.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class SlurmConfig(BaseSettings):
    model_config = {"env_prefix": "SLURM_MCP_", "env_file": ".env", "env_file_encoding": "utf-8"}

    # Required
    username: str

    # Connection
    login_node: str = "sc.stanford.edu"
    ssh_timeout: int = 10
    command_timeout: int = 60

    # Slurm defaults
    default_account: str = "iris"
    default_partition: str = "iris"
    default_time: str = "24:00:00"
    default_mem: str = "32G"
    default_gpus: int = 1
    default_cpus: int = 4
    output_dir: str = "slurm"

    # Directories (auto-derived from username if not set)
    working_dir: Optional[str] = None
    conda_dir: Optional[str] = None
    home_dir: Optional[str] = None

    # Cluster-specific
    gpu_command: str = "sgpu -p iris"
    partitions: str = "iris,iris-hi,iris-interactive,iris-hi-interactive"

    @property
    def resolved_working_dir(self) -> str:
        return self.working_dir or f"/iris/u/{self.username}"

    @property
    def resolved_conda_dir(self) -> str:
        return self.conda_dir or f"/iris/u/{self.username}/data/miniforge3"

    @property
    def resolved_home_dir(self) -> str:
        return self.home_dir or f"/sailhome/{self.username}"

    @property
    def partition_list(self) -> list[str]:
        return [p.strip() for p in self.partitions.split(",")]
