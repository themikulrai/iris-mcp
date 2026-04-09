# iris-mcp

An MCP server that gives AI coding assistants (Claude Code, Cursor, etc.) direct access to Slurm HPC clusters.

Designed for setups where you **code on a workstation** but **submit jobs to a login node** — the server runs locally and proxies Slurm commands over SSH.

## Features

- **Job Management** — submit, list, cancel, and monitor Slurm jobs
- **GPU Status** — see available GPUs across partitions
- **Interactive Sessions** — generate `srun` commands with IDE connection instructions
- **Kerberos-Aware** — detects expired tickets and tells you how to fix it
- **Conda Environments** — list available environments
- **Configurable** — works on any Slurm cluster, ships with Stanford IRIS defaults

## Quick Start

### 1. Clone and setup

```bash
git clone https://github.com/themikulrai/iris-mcp.git
cd iris-mcp
bash setup.sh
```

### 2. Configure

Edit `.env` and set your username:

```bash
SLURM_MCP_USERNAME=your_username
```

That's it for IRIS lab members — all other defaults are pre-configured. See `.env.example` for the full list of options.

### 3. Add to Claude Code

Add to your `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "slurm": {
      "command": "/path/to/iris-mcp/.venv/bin/python",
      "args": ["-m", "slurm_mcp"],
      "cwd": "/path/to/iris-mcp"
    }
  }
}
```

Replace `/path/to/iris-mcp` with the actual path (e.g., `/iris/u/yourname/projects/iris-mcp`).

### 4. Use it

Once configured, your AI assistant can directly interact with your cluster:

- "Check my Slurm auth"
- "Show available GPUs"
- "Submit a training job on 4 H100s"
- "List my running jobs"
- "Show the last 100 lines of job 12345's output"
- "How do I start an interactive session with 2 GPUs?"
- "What conda environments do I have?"

## Tools

| Tool | Description |
|------|-------------|
| `check_auth` | Verify SSH/Kerberos connectivity to the login node |
| `submit_job` | Submit batch jobs via sbatch (supports inline scripts, job arrays) |
| `list_jobs` | List jobs via squeue, with optional partition/state filters |
| `cancel_job` | Cancel jobs by ID, range, or comma-separated list |
| `job_status` | Detailed job info via sacct (running or completed) |
| `tail_output` | Read job stdout/stderr logs |
| `gpu_status` | GPU availability across partitions |
| `cluster_info` | Partition overview, node states, resources |
| `interactive_session` | Generate srun + IDE connection commands |
| `env_list` | List conda/mamba environments |
| `disk_usage` | Check disk usage for working and home directories |

## Configuration

All settings use environment variables with the `SLURM_MCP_` prefix. Set them in `.env` or your shell profile.

| Variable | Default | Description |
|----------|---------|-------------|
| `SLURM_MCP_USERNAME` | *(required)* | Your cluster username |
| `SLURM_MCP_LOGIN_NODE` | `sc.stanford.edu` | SSH target for Slurm commands |
| `SLURM_MCP_DEFAULT_ACCOUNT` | `iris` | Slurm account for job submission |
| `SLURM_MCP_DEFAULT_PARTITION` | `iris` | Default partition |
| `SLURM_MCP_DEFAULT_TIME` | `24:00:00` | Default time limit |
| `SLURM_MCP_DEFAULT_MEM` | `32G` | Default memory |
| `SLURM_MCP_DEFAULT_GPUS` | `1` | Default GPU count |
| `SLURM_MCP_DEFAULT_CPUS` | `4` | Default CPU count |
| `SLURM_MCP_OUTPUT_DIR` | `slurm` | Directory for job output logs |
| `SLURM_MCP_WORKING_DIR` | `/iris/u/{username}` | Primary working directory |
| `SLURM_MCP_CONDA_DIR` | `/iris/u/{username}/data/miniforge3` | Conda installation path |
| `SLURM_MCP_GPU_COMMAND` | `sgpu -p iris` | Command for GPU availability |
| `SLURM_MCP_PARTITIONS` | `iris,iris-hi,iris-interactive,iris-hi-interactive` | Available partitions |

## How It Works

The server runs on your **workstation** (where your IDE lives) and proxies Slurm commands to the **login node** via SSH:

```
IDE (Claude Code) <-> MCP Server (workstation) <--SSH--> Login Node (sc) <-> Slurm
```

- Uses Kerberos/GSSAPI for SSH auth (no keys needed)
- Reads job output files directly from the shared filesystem
- Conda environment listing runs locally (shared filesystem)

## Requirements

- Python 3.10+
- SSH access to a Slurm login node (Kerberos or key-based)
- Shared filesystem between workstation and cluster (for log reading)
- `mcp` Python package (installed automatically)

## Development

```bash
# Run tests
.venv/bin/pytest tests/ -v

# Lint
.venv/bin/ruff check src/ tests/

# Run server directly
.venv/bin/python -m slurm_mcp
```

## License

MIT
