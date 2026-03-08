"""
Command-line interface for minima-llm.

Provides batch management commands for Parasail batch jobs and a proxy server.
"""
import argparse
import asyncio
import sys
from pathlib import Path

from .config import MinimaLlmConfig
from .batch import (
    batch_status_overview,
    cancel_batch,
    cancel_all_batches,
    cancel_all_local_batches,
)


def main():
    parser = argparse.ArgumentParser(
        prog="minima-llm",
        description="Minima LLM - batch management and proxy CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # batch-status command
    batch_parser = subparsers.add_parser(
        "batch-status",
        help="Show status of Parasail batches",
    )
    batch_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config YAML file (default: use environment variables)",
    )
    batch_parser.add_argument(
        "--cancel",
        metavar="PREFIX",
        help="Cancel all batches matching PREFIX and delete local state",
    )
    batch_parser.add_argument(
        "--cancel-remote",
        metavar="BATCH_ID",
        help="Cancel a specific remote batch by ID",
    )
    batch_parser.add_argument(
        "--cancel-all",
        action="store_true",
        help="Cancel ALL local batches",
    )

    # proxy command
    proxy_parser = subparsers.add_parser(
        "minimallm-proxy",
        help="Start OpenAI-compatible proxy server with caching and rate limiting",
    )
    proxy_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Path to config YAML file (default: use environment variables)",
    )
    proxy_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    proxy_parser.add_argument(
        "--port",
        type=int,
        default=8990,
        help="Listen port (default: 8990)",
    )
    proxy_parser.add_argument(
        "--force-model",
        action="store_true",
        help="Override client model with configured OPENAI_MODEL",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "batch-status":
        # Load config
        if args.config:
            config = MinimaLlmConfig.from_yaml(args.config)
        else:
            config = MinimaLlmConfig.from_env()

        # Execute requested action
        if args.cancel:
            cancel_all_batches(config, prefix=args.cancel)
        elif args.cancel_remote:
            cancel_batch(args.cancel_remote, config)
        elif args.cancel_all:
            cancel_all_local_batches(config)
        else:
            batch_status_overview(config)

    elif args.command == "minimallm-proxy":
        if args.config:
            config = MinimaLlmConfig.from_yaml(args.config)
        else:
            config = MinimaLlmConfig.from_env()

        from .proxy import ProxyServer
        server = ProxyServer(
            config,
            host=args.host,
            port=args.port,
            force_model=args.force_model,
        )
        try:
            asyncio.run(server.run())
        except KeyboardInterrupt:
            print("\nProxy stopped.")


def proxy_main():
    """Entry point for the standalone minimallm-proxy command."""
    sys.argv = ["minima-llm", "minimallm-proxy"] + sys.argv[1:]
    main()


if __name__ == "__main__":
    main()
