"""Ingest PDFs and Markdown into the local ChromaDB index.

Usage (from the repository root):

    python scripts/ingest.py --source ./data
    python scripts/ingest.py --source ./data --reset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.ingestion import ingest_source  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument(
        "--source",
        default="./data",
        help="File or directory of PDFs / Markdown to ingest (default: ./data)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop the existing collection before ingesting.",
    )
    args = parser.parse_args()
    summary = ingest_source(args.source, reset=args.reset)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
