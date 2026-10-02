"""Smoke checks for the source CLI's model-command surface."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))
BUILD_DIR = REPO_ROOT / "build"
LIBRARY_NAME = {
    "darwin": "libvgre.dylib",
    "win32": "vgre.dll",
}.get(sys.platform, "libvgre.so")
LIBRARY_PATH = BUILD_DIR / LIBRARY_NAME
if LIBRARY_PATH.is_file():
    os.environ["VGRE_LIB_PATH"] = str(LIBRARY_PATH)

from vgre.cli import build_parser


def test_cli_has_the_documented_model_commands():
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert {"chat", "pull", "generate", "train", "tokenize"} <= set(subparsers.choices)
