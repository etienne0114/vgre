import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
VGRE_SCRIPT = REPO_ROOT / "scripts" / "vgre.sh"
CLI_INSTALL_SCRIPT = REPO_ROOT / "scripts" / "vgre-cli-install.sh"
NATIVE_LIBRARY_NAME = "libvgre.dylib" if sys.platform == "darwin" else "libvgre.so"
NATIVE_CUDART_NAME = "libvgre_cudart.dylib" if sys.platform == "darwin" else "libvgre_cudart.so"


class VgreCliNativeLibrarySelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.install_dir = self.root / "install"
        self.install_lib_dir = self.install_dir / "lib"
        self.install_lib_dir.mkdir(parents=True)
        self.installed_library = self.install_lib_dir / NATIVE_LIBRARY_NAME
        self.installed_library.touch()

        self.fake_python = self.root / "python"
        self.fake_python.write_text(
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = \"-c\" ]; then exit 0; fi\n"
            "printf '%s\\n' \"${VGRE_LIB_PATH:-}\"\n"
        )
        self.fake_python.chmod(0o755)

    def run_cli(self, extra_env=None):
        env = os.environ.copy()
        env.update(
            {
                "HOME": str(self.home),
                "VGRE_INSTALL_DIR": str(self.install_dir),
                "VGRE_PYTHON": str(self.fake_python),
                "VGRE_VENV": str(self.root / "missing-venv"),
            }
        )
        env.pop("VGRE_LIB_PATH", None)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["/bin/sh", str(VGRE_SCRIPT), "info"],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_prefers_fresh_source_install_over_bundled_package_library(self):
        selected = self.run_cli()
        self.assertEqual(selected, str(self.installed_library))

    def test_honors_explicit_valid_library_override(self):
        override = self.root / "explicit-libvgre.so"
        override.touch()
        selected = self.run_cli({"VGRE_LIB_PATH": str(override)})
        self.assertEqual(selected, str(override))

    def test_sync_copies_fresh_runtime_and_sibling_into_venv_package(self):
        native_dir = self.root / "native"
        native_dir.mkdir()
        (native_dir / NATIVE_LIBRARY_NAME).write_bytes(b"fresh runtime")
        (native_dir / f"{NATIVE_LIBRARY_NAME}.0").symlink_to(NATIVE_LIBRARY_NAME)
        (native_dir / NATIVE_CUDART_NAME).write_bytes(b"fresh cudart")

        venv = self.root / "venv"
        python = venv / "bin" / "python"
        python.parent.mkdir(parents=True)
        purelib = venv / "lib" / "python3.12" / "site-packages"
        package = purelib / "vgre"
        package.mkdir(parents=True)
        python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$VGRE_TEST_PURELIB\"\n"
        )
        python.chmod(0o755)
        env = os.environ.copy()
        env["VGRE_TEST_PURELIB"] = str(package / "lib")

        result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                '. "$1"; vgre_copy_python_native_libraries "$2" "$3"',
                "vgre-copy-test",
                str(CLI_INSTALL_SCRIPT),
                str(native_dir),
                str(venv),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        package_lib = package / "lib"
        self.assertTrue((package_lib / NATIVE_LIBRARY_NAME).is_file(), result.stdout)
        self.assertEqual((package_lib / NATIVE_LIBRARY_NAME).read_bytes(), b"fresh runtime")
        self.assertTrue((package_lib / f"{NATIVE_LIBRARY_NAME}.0").is_symlink())
        self.assertEqual(
            (package_lib / NATIVE_CUDART_NAME).read_bytes(), b"fresh cudart"
        )

    @unittest.skipUnless(
        (REPO_ROOT / "build" / NATIVE_LIBRARY_NAME).exists(),
        "requires the freshly-built native runtime",
    )
    def test_copied_runtime_can_construct_a_language_model(self):
        native_dir = REPO_ROOT / "build"
        venv = self.root / "runtime-venv"
        python = venv / "bin" / "python"
        python.parent.mkdir(parents=True)
        purelib = venv / "lib" / "python3.12" / "site-packages"
        package = purelib / "vgre"
        package.mkdir(parents=True)
        python.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$VGRE_TEST_PURELIB\"\n"
        )
        python.chmod(0o755)
        env = os.environ.copy()
        env["VGRE_TEST_PURELIB"] = str(package / "lib")

        copy_result = subprocess.run(
            [
                "/bin/sh",
                "-c",
                '. "$1"; vgre_copy_python_native_libraries "$2" "$3"',
                "vgre-copy-test",
                str(CLI_INSTALL_SCRIPT),
                str(native_dir),
                str(venv),
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(copy_result.returncode, 0, copy_result.stderr)

        env["PYTHONPATH"] = str(REPO_ROOT / "bindings" / "python")
        env["VGRE_LIB_PATH"] = str(package / "lib" / NATIVE_LIBRARY_NAME)
        model_result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from vgre import LanguageModel; model = LanguageModel(vocab=256, n_layer=1, d_model=8, n_head=1, d_ff=16, max_seq=8); print(model.num_parameters); model.close()",
            ],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(model_result.returncode, 0, model_result.stderr)
        self.assertEqual(model_result.stdout.strip(), "4760")


if __name__ == "__main__":
    unittest.main()
