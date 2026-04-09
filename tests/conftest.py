"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from slurm_mcp.config import SlurmConfig
from slurm_mcp.slurm import SlurmClient


@pytest.fixture
def config():
    """Config with test defaults — no .env file needed."""
    with patch.dict("os.environ", {"SLURM_MCP_USERNAME": "testuser"}, clear=False):
        return SlurmConfig(
            username="testuser",
            login_node="test-sc.example.com",
            working_dir="/iris/u/testuser",
            conda_dir="/iris/u/testuser/data/miniforge3",
            home_dir="/sailhome/testuser",
        )


@pytest.fixture
def client(config):
    """SlurmClient with mocked SSH execution."""
    c = SlurmClient(config)
    c.run_remote = AsyncMock()
    return c
