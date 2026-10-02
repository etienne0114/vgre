"""Regression tests for model CLI commands that were previously missing."""

import contextlib
import io
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
LIBRARY_NAME = {
    "darwin": "libvgre.dylib",
    "win32": "vgre.dll",
}.get(sys.platform, "libvgre.so")
LIBRARY_PATH = BUILD_DIR / LIBRARY_NAME
os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
    part for part in (str(BUILD_DIR), os.environ.get("LD_LIBRARY_PATH", "")) if part
)
if LIBRARY_PATH.is_file():
    os.environ["VGRE_LIB_PATH"] = str(LIBRARY_PATH)
sys.path.insert(0, str(REPO_ROOT / "bindings" / "python"))

from vgre.cli import build_parser, main


def _run_cli(args):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            returncode = main(args)
        except SystemExit as error:
            returncode = error.code or 0
    return returncode, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize("command", ["chat", "pull", "generate", "train", "tokenize"])
def test_model_commands_are_registered(command):
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert command in subparsers.choices


@pytest.mark.parametrize("command", ["chat", "pull"])
def test_feature_command_help(command):
    returncode, stdout, stderr = _run_cli([command, "--help"])
    output = (stdout + stderr).lower()
    assert returncode == 0
    assert f"usage: vgre {command}" in output


def test_chat_missing_model_reports_file_error(tmp_path):
    missing_model = tmp_path / "missing.gguf"
    returncode, stdout, stderr = _run_cli(["chat", str(missing_model)])
    output = (stdout + stderr).lower()
    assert returncode != 0
    assert "failed to load gguf model" in output
    assert str(missing_model).lower() in output
    assert "does not exist" in output
    assert "vgre chat smollm2" in output


def test_chat_defaults_to_downloadable_preset():
    args = build_parser().parse_args(["chat"])
    assert args.model == "smollm2"


def test_pull_uses_platform_cache_unless_overridden(tmp_path, monkeypatch):
    from vgre.cli import _model_cache_root

    override = tmp_path / "models"
    monkeypatch.setenv("VGRE_MODEL_CACHE_DIR", str(override))
    assert _model_cache_root() == override

    pull_args = build_parser().parse_args(["pull", "smollm2"])
    assert pull_args.directory is None


def test_pull_unknown_preset_is_rejected():
    returncode, stdout, stderr = _run_cli(["pull", "not-a-preset"])
    output = (stdout + stderr).lower()
    assert returncode != 0
    assert "unknown model preset" in output
    assert "smollm2" in output
