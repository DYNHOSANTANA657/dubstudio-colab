"""Setup regressions: mocked Git and real local subprocess output, no network."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import dubstudio_bootstrap as bootstrap


class CloneMaterializationTests(unittest.TestCase):
    repository = "https://example.invalid/model.git"
    revision = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.destination = self.root / "model"

    def tearDown(self):
        self.temporary.cleanup()

    def git_directory(self, *, materialized=False):
        git = self.destination / ".git"
        git.mkdir(parents=True)
        if materialized:
            (git / "index").write_bytes(b"mock materialized index")

    def head(self, revision=None, *, returncode=0):
        return subprocess.CompletedProcess(
            ["git", "rev-parse", "HEAD"], returncode,
            stdout=(revision or self.revision) + "\n", stderr="",
        )

    def test_fresh_clone_already_at_pin_still_checks_out_files(self):
        # A real --no-checkout clone can already have the pinned HEAD but no index.
        def pretend_execute(command, **kwargs):
            if command[:2] == ["git", "clone"]:
                self.git_directory()

        with patch.object(bootstrap, "execute", side_effect=pretend_execute) as execute, patch.object(bootstrap.subprocess, "run", return_value=self.head()) as run:
            bootstrap.clone_at(self.repository, self.destination, self.revision)
        execute.assert_has_calls([
            call(["git", "clone", "--filter=blob:none", "--no-checkout", self.repository, str(self.destination)]),
            call(["git", "checkout", "--detach", self.revision], cwd=self.destination),
        ])
        self.assertEqual(execute.call_count, 2)
        run.assert_called_once_with(["git", "rev-parse", "HEAD"], cwd=self.destination, capture_output=True, text=True)

    def test_existing_unmaterialized_clone_at_pin_requires_checkout(self):
        self.git_directory()
        with patch.object(bootstrap, "execute") as execute, patch.object(bootstrap.subprocess, "run", return_value=self.head()):
            bootstrap.clone_at(self.repository, self.destination, self.revision)
        execute.assert_called_once_with(["git", "checkout", "--detach", self.revision], cwd=self.destination)

    def test_existing_materialized_clone_at_pin_preserves_local_patch(self):
        self.git_directory(materialized=True)
        patched_file = self.destination / "demoTalkNet.py"
        original_patch = b"authorized compatibility patch must remain\n"
        patched_file.write_bytes(original_patch)
        with patch.object(bootstrap, "execute") as execute, patch.object(bootstrap.subprocess, "run", return_value=self.head()):
            bootstrap.clone_at(self.repository, self.destination, self.revision)
        execute.assert_not_called()
        self.assertEqual(patched_file.read_bytes(), original_patch)

    def test_different_revision_fetches_then_checks_out_requested_pin(self):
        self.git_directory(materialized=True)
        with patch.object(bootstrap, "execute") as execute, patch.object(bootstrap.subprocess, "run", return_value=self.head("different-revision")):
            bootstrap.clone_at(self.repository, self.destination, self.revision)
        self.assertEqual(execute.call_args_list, [
            call(["git", "fetch", "--depth", "1", "origin", self.revision], cwd=self.destination),
            call(["git", "checkout", "--detach", self.revision], cwd=self.destination),
        ])

    def test_unreadable_head_fetches_then_checks_out_requested_pin(self):
        self.git_directory(materialized=True)
        with patch.object(bootstrap, "execute") as execute, patch.object(bootstrap.subprocess, "run", return_value=self.head(returncode=128)):
            bootstrap.clone_at(self.repository, self.destination, self.revision)
        self.assertEqual(execute.call_args_list, [
            call(["git", "fetch", "--depth", "1", "origin", self.revision], cwd=self.destination),
            call(["git", "checkout", "--detach", self.revision], cwd=self.destination),
        ])


class SetupOutputTests(unittest.TestCase):
    def test_stdout_and_stderr_are_forwarded_from_real_local_process(self):
        output = io.StringIO()
        command = [sys.executable, "-u", "-c", "import sys; print('setup-stdout-probe'); print('setup-stderr-probe', file=sys.stderr)"]
        with contextlib.redirect_stdout(output):
            bootstrap.execute(command)
        text = output.getvalue()
        self.assertIn("setup-stdout-probe", text)
        self.assertIn("setup-stderr-probe", text)
        self.assertIn("iniciado", text)
        self.assertIn("concluído", text)

    def test_failed_real_process_forwards_cause_and_preserves_exit_code(self):
        output = io.StringIO()
        command = [sys.executable, "-u", "-c", "import sys; print('setup-explicit-failure', file=sys.stderr); sys.exit(7)"]
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError) as error:
            bootstrap.execute(command)
        self.assertEqual(error.exception.returncode, 7)
        self.assertEqual(error.exception.cmd, command)
        self.assertIn("setup-explicit-failure", output.getvalue())
        self.assertNotIn("concluído", output.getvalue())


class DependencyContractTests(unittest.TestCase):
    """Exercise setup orchestration without installing or downloading packages."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.storage = self.root / "storage"
        self.muse = self.runtime / "musetalk-env"
        self.marker = self.muse / ".ready.json"
        self.muse_python = str(self.muse / "bin/python")
        self.muse_mim = str(self.muse / "bin/mim")
        for name in ("app-env", "musetalk-env"):
            executable = self.runtime / name / "bin/python"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"mock interpreter, never executed")
        self.commands = []

    def tearDown(self):
        self.temporary.cleanup()

    def requirements(self):
        path = Path(bootstrap.__file__).parent / "requirements-musetalk.txt"
        return path, {
            line.strip().split("==", 1)[0]: line.strip().split("==", 1)[1]
            for line in path.read_text(encoding="utf-8").splitlines()
            if "==" in line and not line.lstrip().startswith("#")
        }

    def is_muse_pip(self, command):
        return command[:3] == [self.muse_python, "-m", "pip"]

    def is_check(self, command):
        return self.is_muse_pip(command) and command[3:] == ["check"]

    def is_import_probe(self, command):
        return (
            command[:2] == [self.muse_python, "-c"]
            and "musetalk.utils.preprocessing" in command[-1]
        )

    def prepare_mocked(self, *, fail_when=None, on_execute=None):
        def execute(command, **kwargs):
            self.commands.append((list(command), kwargs.copy()))
            if on_execute:
                on_execute(command, kwargs)
            if fail_when and fail_when(command):
                raise subprocess.CalledProcessError(1, command)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(bootstrap, "gpu_precheck", return_value="mock NVIDIA T4"))
            stack.enter_context(patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=30 * 1024**3)))
            stack.enter_context(patch.object(bootstrap.shutil, "which", return_value="mock-installed-tool"))
            stack.enter_context(patch.object(bootstrap, "execute", side_effect=execute))
            stack.enter_context(patch.object(bootstrap, "clone_at"))
            stack.enter_context(patch.object(bootstrap, "download_model_weights"))
            stack.enter_context(patch.object(bootstrap, "copy_missing"))
            stack.enter_context(patch.object(bootstrap, "persist_cache"))
            stack.enter_context(patch.object(Path, "symlink_to"))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return bootstrap.prepare(self.runtime, self.storage, needs_lipsync=True)

    def assert_isolated(self, values):
        path, _ = self.requirements()
        self.assertEqual(values["PYTHONNOUSERSITE"], "1")
        self.assertEqual(values["PIP_CONSTRAINT"], str(path.resolve()))
        self.assertTrue(values["PATH"].startswith(str(self.muse / "bin") + ":"))
        self.assertNotIn(str(self.runtime / "app-env"), values["LD_LIBRARY_PATH"])
        self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", values)

    def test_numpy_and_opencv_match_pinned_official_musetalk_abi(self):
        _, requirements = self.requirements()
        self.assertEqual(requirements["numpy"], "1.23.5")
        self.assertEqual(requirements["opencv-python"], "4.9.0.80")
        # If two distributions are deliberately installed, neither can upgrade
        # the shared cv2 module to an incompatible OpenCV/NumPy ABI.
        for name, version in requirements.items():
            if name.startswith("opencv-python"):
                self.assertEqual(version, "4.9.0.80")

    def test_tomli_is_explicit_for_yapf_under_python310(self):
        _, requirements = self.requirements()
        self.assertEqual(requirements["tomli"], "2.2.1")

    def test_every_muse_pip_resolution_is_constrained_and_isolated(self):
        self.prepare_mocked()
        resolutions = [(command, options) for command, options in self.commands if self.is_muse_pip(command)]
        self.assertGreaterEqual(len(resolutions), 5)
        for command, options in resolutions:
            with self.subTest(command=command):
                self.assert_isolated(options["env"])

    def test_all_mim_nested_resolutions_use_same_constraints(self):
        self.prepare_mocked()
        resolutions = [(command, options) for command, options in self.commands if command[0] == self.muse_mim]
        self.assertEqual({command[-1] for command, _ in resolutions}, {"mmengine==0.10.7", "mmcv==2.0.1", "mmdet==3.1.0"})
        for command, options in resolutions:
            with self.subTest(command=command):
                self.assert_isolated(options["env"])

    def test_muse_keeps_colab_driver_and_own_cuda_libraries_not_app_libraries(self):
        driver = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
        cuda_runtime = self.muse / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
        cuda_runtime.mkdir(parents=True)
        with patch.dict(bootstrap.os.environ, {"LD_LIBRARY_PATH": driver, "PATH": "/usr/bin:/bin"}):
            self.prepare_mocked()
        probes = [(command, options) for command, options in self.commands if self.is_import_probe(command)]
        self.assertEqual(len(probes), 1)
        values = probes[0][1]["env"]
        self.assertEqual(values["LD_LIBRARY_PATH"], ":".join([
            str(self.muse / "lib/python3.10/site-packages/torch/lib"),
            str(cuda_runtime), driver,
        ]))
        self.assertEqual(values["PATH"], str(self.muse / "bin") + ":/usr/bin:/bin")
        self.assert_isolated(values)

    def test_chumpy_uses_seeded_build_tools_without_dependency_upgrades(self):
        self.prepare_mocked()
        commands = [command for command, _ in self.commands]
        matches = [command for command in commands if "chumpy==0.70" in command]
        self.assertEqual(len(matches), 1)
        chumpy = matches[0]
        self.assertTrue(self.is_muse_pip(chumpy))
        self.assertIn("--no-build-isolation", chumpy)
        self.assertIn("--no-deps", chumpy)
        mmpose = next(command for command in commands if "mmpose==1.1.0" in command)
        self.assertLess(commands.index(mmpose), commands.index(chumpy))
        seeded_tools = next(command for command in commands if self.is_muse_pip(command) and "setuptools==80.9.0" in command)
        self.assertLess(commands.index(seeded_tools), commands.index(chumpy))

    def test_pip_check_and_import_probe_precede_ready_marker(self):
        checked = []

        def check_marker(command, options):
            if self.is_check(command) or self.is_import_probe(command):
                self.assertFalse(self.marker.exists(), "An unchecked environment must not be advertised as ready")
                checked.append("pip_check" if self.is_check(command) else "gpu_imports")

        self.prepare_mocked(on_execute=check_marker)
        self.assertIn("pip_check", checked)
        self.assertIn("gpu_imports", checked)
        self.assertTrue(self.marker.is_file())

    def test_dependency_check_failure_leaves_muse_unready(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare_mocked(fail_when=self.is_check)
        self.assertFalse(self.marker.exists())

    def test_import_or_gpu_failure_leaves_muse_unready(self):
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare_mocked(fail_when=self.is_import_probe)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
