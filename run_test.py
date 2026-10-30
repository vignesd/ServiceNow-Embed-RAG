from __future__ import annotations

from pathlib import Path
import sys

import pytest


def main() -> int:
    """Run all classifier tests from testing/test_classifier.py."""
    project_root = Path(__file__).resolve().parent
    test_file = project_root / "testing" / "test_classifier.py"

    if not test_file.is_file():
        print(f"Test file not found: {test_file}", file=sys.stderr)
        return 2

    args = [
        str(test_file),
        "-v",
        "--tb=short",
    ]

    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
