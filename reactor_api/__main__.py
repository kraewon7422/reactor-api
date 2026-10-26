"""Command-line entry point: ``reactor-api`` or ``python -m reactor_api``."""

from __future__ import annotations

import argparse


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(
        prog="reactor-api", description="Serve the reactor simulation API."
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = ap.parse_args()
    uvicorn.run(
        "reactor_api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
