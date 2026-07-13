"""Command-line smoke runner for the frozen VerCOR 3.0 plugin."""

from __future__ import annotations

import json

from vercor_compat_plugin_3_0.plugin import run_smoke


def main() -> None:
    """Run the installed compatibility plugin and print compact JSON."""

    print(json.dumps(run_smoke(), sort_keys=True))


if __name__ == "__main__":
    main()
