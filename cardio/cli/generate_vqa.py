"""CLI entry point for VQA data generation.

Registered as ``cardio-generate-vqa`` in ``pyproject.toml``::

    cardio-generate-vqa --output-dir ./vqa_output --splits train val test
    cardio-generate-vqa --stage convert --format all --output-dir ./vqa_output
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    """Delegate to the pipeline orchestrator in :mod:`cardio.data.vqa.run`."""
    from cardio.data.vqa.run import main as _run_main

    _run_main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    main()
