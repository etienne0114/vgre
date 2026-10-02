"""Cross-platform subprocess coverage for the real interactive chat CLI."""

import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "bindings" / "python"
TEST_DIR = Path(__file__).resolve().parent
NATIVE_NAME = {
    "darwin": "libvgre.dylib",
    "win32": "vgre.dll",
}.get(sys.platform, "libvgre.so")
NATIVE_PATH = REPO_ROOT / "build" / NATIVE_NAME
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(PYTHON_DIR))

from test_gguf_auto_config import _write_gguf


def run_cli(args, *, input_text=""):
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
        input=input_text,
        text=True,
        capture_output=True,
        timeout=90,
        check=False,
    )


def test_chat_help_describes_model_and_generation_options():
    result = run_cli(["chat", "--help"])
    output = result.stdout.lower()
    assert result.returncode == 0, result.stderr
    assert "usage: vgre chat" in output
    assert "model" in output and "--max-tokens" in output


def test_chat_runs_a_native_session_with_a_valid_gguf(tmp_path):
    model_path = tmp_path / "tiny-chat.gguf"
    _write_gguf(model_path)

    result = run_cli(
        ["chat", str(model_path), "--max-tokens", "2", "--temperature", "0"],
        input_text="hello\n",
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "loaded GGUF model" in result.stderr
    assert "Assistant:" in result.stdout


def test_chat_rejects_malformed_gguf_with_an_actionable_error(tmp_path):
    model_path = tmp_path / "malformed.gguf"
    model_path.write_bytes(b"GGUF\x03\x00")

    result = run_cli(["chat", str(model_path)])
    output = (result.stdout + result.stderr).lower()
    assert result.returncode == 2
    assert "failed to load gguf model" in output
    assert str(model_path).lower() in output
