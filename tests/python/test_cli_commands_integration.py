"""Subprocess checks for the installed model and runtime CLI commands."""

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "bindings" / "python"
NATIVE_NAME = {
    "darwin": "libvgre.dylib",
    "win32": "vgre.dll",
}.get(sys.platform, "libvgre.so")
NATIVE_PATH = REPO_ROOT / "build" / NATIVE_NAME


def run_cli(args):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(PYTHON_DIR), env.get("PYTHONPATH", "")) if part
    )
    env["VGRE_IPC_MODE"] = "OFF"
    if NATIVE_PATH.is_file():
        env["VGRE_LIB_PATH"] = str(NATIVE_PATH)
    if sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (str(REPO_ROOT / "build"), env.get("DYLD_LIBRARY_PATH", "")) if part
        )
    elif sys.platform != "win32":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            part for part in (str(REPO_ROOT / "build"), env.get("LD_LIBRARY_PATH", "")) if part
        )
    return subprocess.run(
        [sys.executable, "-m", "vgre", *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def test_help_and_version_commands_return_success():
    help_result = run_cli(["--help"])
    assert help_result.returncode == 0, help_result.stderr
    assert "usage: vgre" in help_result.stdout.lower()
    assert "chat" in help_result.stdout and "generate" in help_result.stdout

    version_result = run_cli(["--version"])
    assert version_result.returncode == 0, version_result.stderr
    assert "vgre" in version_result.stdout.lower()
    assert "native backend: available" in version_result.stdout.lower()


def test_each_model_command_exposes_its_actual_parser_help():
    for command in ("chat", "pull", "generate", "train", "tokenize", "info"):
        result = run_cli([command, "--help"])
        assert result.returncode == 0, f"{command}: {result.stderr}"
        assert f"usage: vgre {command}" in result.stdout.lower()


def test_pull_rejects_unknown_model_without_network_access():
    result = run_cli(["pull", "not-a-preset"])
    output = (result.stdout + result.stderr).lower()
    assert result.returncode == 2
    assert "unknown model preset" in output
    assert "smollm2" in output


def test_generate_reports_a_missing_gguf_path(tmp_path):
    model_path = tmp_path / "missing.gguf"
    result = run_cli(["generate", "--model", str(model_path)])
    output = (result.stdout + result.stderr).lower()
    assert result.returncode == 2
    assert "failed to load gguf model" in output
    assert str(model_path).lower() in output


def test_speculative_sampling_requires_greedy_temperature():
    result = run_cli(["generate", "--speculative", "--temperature", "0.7"])
    assert result.returncode == 2
    assert "requires --temperature 0" in result.stderr
