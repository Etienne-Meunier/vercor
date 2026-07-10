"""Command-line smoke runner for the independently packaged plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from vercor_public_plugin.plugin import run_smoke


def main(arguments: Sequence[str] | None = None) -> None:
    """Run the installed plugin and print one compact JSON result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(arguments)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run_smoke(args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
