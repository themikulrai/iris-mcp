"""Entry point for `python -m slurm_mcp`."""

from .server import create_server


def main() -> None:
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
