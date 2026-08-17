"""Execute the workshop notebook into an ignored build artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "hands_on_agentes_jpr2026.ipynb"
DEFAULT_OUTPUT = ROOT / ".build" / "nbtest" / "executed.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    notebook = nbformat.read(args.input, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute(cwd=str(ROOT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(executed, args.output)
    print(f"Executed {len(executed.cells)} cells without errors: {args.output}")


if __name__ == "__main__":
    main()
