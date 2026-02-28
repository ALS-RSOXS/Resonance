"""Entry point for running MCP server as standalone application."""

import sys

from .server import mcp


def main() -> None:
    try:
        mcp.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
