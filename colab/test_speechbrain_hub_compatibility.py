"""Hub 1.x/SpeechBrain regressions with isolated imports, no model or network."""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner


# Preserve the actual pinned fetch handler and from_hparams' optional module
# semantics without importing Torch or downloading a model in offline tests.
FETCH_SOURCE = '''import huggingface_hub
from requests.exceptions import HTTPError

def fetch(filename, source):
    try:
        return huggingface_hub.hf_hub_download(repo_id=source, filename=filename)
    except HTTPError as e:
        if "404 Client Error" in str(e):
            raise ValueError("File not found on HF hub") from e
        raise
'''

INTERFACE_SOURCE = '''
def from_hparams(hparams_file="hyperparams.yaml", pymodule_file="custom.py"):
    parameters = fetch(hparams_file, "speechbrain/spkrec-ecapa-voxceleb")
    try:
        custom = fetch(pymodule_file, "speechbrain/spkrec-ecapa-voxceleb")
    except ValueError:
        if pymodule_file == "custom.py":
            pass
        else:
            raise
    return parameters
'''


class RemoteEntryNotFoundError(Exception):
    pass


class LegacyHTTPError(Exception):
    pass


class SpeechBrainHubErrorContractTests(unittest.TestCase):
    def module(self, download, *, compatible=True):
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = download
        errors = types.ModuleType("huggingface_hub.errors")
        errors.RemoteEntryNotFoundError = RemoteEntryNotFoundError
        requests = types.ModuleType("requests")
        requests_errors = types.ModuleType("requests.exceptions")
        requests_errors.HTTPError = LegacyHTTPError
        source = bootstrap.speechbrain_hub_compatible_source(FETCH_SOURCE) if compatible else FETCH_SOURCE
        namespace = {}
        with patch.dict(sys.modules, {"huggingface_hub": hub, "huggingface_hub.errors": errors,
                                      "requests": requests, "requests.exceptions": requests_errors}):
            exec(compile(source + INTERFACE_SOURCE, "speechbrain-contract-fixture", "exec"), namespace)
        return namespace

    def test_missing_default_custom_py_reproduces_old_failure_and_is_optional_after_fix(self):
        missing = RemoteEntryNotFoundError("404 custom.py")

        def download(repo_id, filename):
            if filename == "custom.py":
                raise missing
            return "/cache/hyperparams.yaml"

        before = self.module(download, compatible=False)
        with self.assertRaises(RemoteEntryNotFoundError):
            before["from_hparams"]()
        after = self.module(download)
        self.assertEqual(after["from_hparams"](), "/cache/hyperparams.yaml")

    def test_missing_mandatory_parameters_are_fatal_with_original_cause(self):
        missing = RemoteEntryNotFoundError("404 hyperparams.yaml")

        def download(**kwargs):
            raise missing

        namespace = self.module(download)
        with self.assertRaisesRegex(ValueError, "File not found") as raised:
            namespace["from_hparams"]()
        self.assertIs(raised.exception.__cause__, missing)

    def test_missing_mandatory_weights_and_nondefault_python_module_are_fatal(self):
        for filename in ("embedding_model.ckpt", "required_interface.py"):
            missing = RemoteEntryNotFoundError("404 " + filename)

            def download(repo_id, filename, missing=missing):
                if filename == "hyperparams.yaml":
                    return "/cache/hyperparams.yaml"
                raise missing

            namespace = self.module(download)
            with self.subTest(filename=filename), self.assertRaises(ValueError) as raised:
                if filename.endswith(".ckpt"):
                    namespace["fetch"](filename, "speechbrain/spkrec-ecapa-voxceleb")
                else:
                    namespace["from_hparams"](pymodule_file=filename)
            self.assertIs(raised.exception.__cause__, missing)

    def test_auth_repository_revision_offline_and_network_errors_are_not_ignored(self):
        failures = [type(name, (Exception,), {})(message) for name, message in (
            ("RepositoryNotFoundError", "401 invalid credentials"),
            ("GatedRepoError", "403 forbidden"), ("RevisionNotFoundError", "404 invalid revision"),
            ("HfHubHTTPError", "429 rate limited"), ("HfHubHTTPError", "500 service error"),
            ("LocalEntryNotFoundError", "offline cache missing"),
            ("ConnectError", "network unreachable"), ("ReadTimeout", "request timed out"),
        )]
        for failure in failures:
            def download(repo_id, filename, failure=failure):
                if filename == "hyperparams.yaml":
                    return "/cache/hyperparams.yaml"
                raise failure
            namespace = self.module(download)
            with self.subTest(error=type(failure).__name__), self.assertRaises(type(failure)) as raised:
                namespace["from_hparams"]()
            self.assertIs(raised.exception, failure)

    def test_successful_fetch_preserves_repository_filename_and_return_value(self):
        seen = []

        def download(**kwargs):
            seen.append(kwargs)
            return "/cache/verified-file"

        namespace = self.module(download)
        self.assertEqual(namespace["fetch"]("weights.ckpt", "speaker/model"), "/cache/verified-file")
        self.assertEqual(seen, [{"repo_id": "speaker/model", "filename": "weights.ckpt"}])

    def test_original_requests_http_contract_is_preserved(self):
        for message, optional in (("404 Client Error", True), ("401 Client Error", False)):
            failure = LegacyHTTPError(message)

            def download(repo_id, filename, failure=failure):
                if filename == "hyperparams.yaml":
                    return "/cache/hyperparams.yaml"
                raise failure

            namespace = self.module(download)
            with self.subTest(message=message):
                if optional:
                    self.assertEqual(namespace["from_hparams"](), "/cache/hyperparams.yaml")
                else:
                    with self.assertRaises(LegacyHTTPError) as raised:
                        namespace["from_hparams"]()
                    self.assertIs(raised.exception, failure)


class SpeechBrainInstalledPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                     prefix=".test-speechbrain-hub-")
        self.root = Path(self.temporary.name)
        self.runtime, self.storage = self.root / "runtime", self.root / "storage"
        self.target = self.runtime / "app-env/lib/python3.10/site-packages/speechbrain/utils/fetching.py"
        self.target.parent.mkdir(parents=True)
        self.original = FETCH_SOURCE.replace("\n", "\r\n").encode()
        self.target.write_bytes(self.original)

    def tearDown(self):
        self.temporary.cleanup()

    def test_patch_is_idempotent_and_original_bytes_remain_recoverable(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(bootstrap.patch_speechbrain_hub_compatibility(self.runtime))
            first = self.target.read_bytes()
            self.assertFalse(bootstrap.patch_speechbrain_hub_compatibility(self.runtime))
        self.assertEqual(self.target.read_bytes(), first)
        self.assertEqual(self.target.with_name("fetching.py.dubstudio-hub-original").read_bytes(), self.original)
        self.assertEqual(first.count(b"except _DubStudioRemoteEntryNotFoundError"), 1)

    def test_unknown_or_partial_source_is_not_overwritten(self):
        for source in ("def unrelated():\n    pass\n", FETCH_SOURCE + "# _DubStudioRemoteEntryNotFoundError\n"):
            self.target.write_text(source, encoding="utf-8")
            original = self.target.read_bytes()
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "compatibilidade testada"):
                bootstrap.patch_speechbrain_hub_compatibility(self.runtime)
            self.assertEqual(self.target.read_bytes(), original)
            self.assertFalse(self.target.with_name("fetching.py.dubstudio-hub-original").exists())

    def test_missing_package_is_reported_without_installing_or_downloads(self):
        with self.assertRaisesRegex(RuntimeError, "SpeechBrain não foi instalado"):
            bootstrap.patch_speechbrain_hub_compatibility(self.root / "missing-runtime")

    def prepare(self, ready):
        app = self.runtime / "app-env"
        (app / "bin").mkdir()
        (app / "bin/python").write_bytes(b"unused interpreter protocol fixture")
        marker = app / ".ready.json"
        if ready:
            requirements = Path(bootstrap.__file__).with_name("requirements-app.txt")
            marker.write_text(json.dumps({"fingerprint": hashlib.sha256(
                requirements.read_bytes() + bootstrap.PIPELINE_VERSION.encode()).hexdigest()}), encoding="utf-8")
        commands = []

        def execute(command, **kwargs):
            commands.append(command)
            if command[:2] == [str(app / "bin/python"), "-c"]:
                self.assertIn("except _DubStudioRemoteEntryNotFoundError", self.target.read_text(encoding="utf-8"))
                self.assertFalse(marker.exists())

        with patch.object(bootstrap, "gpu_precheck", return_value="GPU protocol fixture"), \
             patch.object(bootstrap.shutil, "disk_usage", return_value=types.SimpleNamespace(free=30 * 1024**3)), \
             patch.object(bootstrap.shutil, "which", return_value="installed-tool-fixture"), \
             patch.object(bootstrap, "execute", side_effect=execute), \
             patch.object(bootstrap, "copy_missing"), patch.object(bootstrap, "persist_cache"), \
             contextlib.redirect_stdout(io.StringIO()):
            bootstrap.prepare(self.runtime, self.storage, needs_lipsync=False)
        return commands

    def test_warm_environment_is_repaired_without_reinstall_or_pipeline_hash_change(self):
        before = runner.worker_fingerprint()
        commands = self.prepare(True)
        self.assertEqual(runner.worker_fingerprint(), before)
        self.assertIn("except _DubStudioRemoteEntryNotFoundError", self.target.read_text(encoding="utf-8"))
        self.assertFalse(any("pip" in command for command in commands))

    def test_first_setup_patches_dependency_before_imports_and_ready_marker(self):
        commands = self.prepare(False)
        self.assertTrue(any("pip" in command for command in commands))
        self.assertTrue((self.runtime / "app-env/.ready.json").is_file())
        self.assertIn("except _DubStudioRemoteEntryNotFoundError", self.target.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
