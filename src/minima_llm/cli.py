"""
Command-line interface for minima-llm.

Provides batch management commands for Parasail batch jobs.
"""
import argparse
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
        description="Minima LLM - batch management CLI",
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


if __name__ == "__main__":
    main()