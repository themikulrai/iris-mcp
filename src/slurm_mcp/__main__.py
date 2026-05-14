"""Entry point for `python -m slurm_mcp`."""

import os
from .server import create_server


def _pin_krb5_ccache() -> None:
    """Force KRB5CCNAME at our DIR collection (managed by k5start). Spawned
    MCPs inherit env from the parent VSCode-server, which may hold a stale
    FILE: pointer from a long-lived session — break that chain explicitly."""
    os.environ["KRB5CCNAME"] = f"DIR:/tmp/krb5cc_{os.getuid()}.d/"


def main() -> None:
    _pin_krb5_ccache()
    server = create_server()
    server.run()


if __name__ == "__main__":
    main()
