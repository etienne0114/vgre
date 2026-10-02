"""Exercise GGUF metadata auto-configuration through the public CLI process."""

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
        timeout=90,
        check=False,
    )


def test_generate_loads_typed_gguf_and_runs_native_inference(tmp_path):
    for architecture in ("llama", "qwen2"):
        model_path = tmp_path / f"tiny-{architecture}.gguf"
        _write_gguf(model_path, architecture)

        result = run_cli([
            "generate", "--model", str(model_path), "--prompt", "hello",
            "--max-tokens", "2", "--temperature", "0",
        ])
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "loaded GGUF model" in result.stderr
        assert len(result.stdout) >= 2
