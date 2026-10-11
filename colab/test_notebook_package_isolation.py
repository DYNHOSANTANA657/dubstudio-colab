"""Notebook package switching without Colab, network, heavy deps, or GPU."""
import ast
import contextlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from unittest.mock import patch

NOTEBOOK = Path(__file__).resolve().parents[1] / "DubStudio_Colab.ipynb"


class ReplaceUploadRoot(ast.NodeTransformer):
    def __init__(self, root):
        self.root = str(root)

    def visit_Constant(self, node):
        if node.value == "/content/dubstudio-packages":
            return ast.copy_location(ast.Constant(self.root), node)
        return node


class NotebookPackageIsolationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="dub-nb-")
        self.root = Path(self.folder.name)
        self.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "".join(self.notebook["cells"][2]["source"])
        syntax = ReplaceUploadRoot(self.root / "packages").visit(ast.parse(source))
        self.upload_code = compile(ast.fix_missing_locations(syntax), "notebook-upload", "exec")
        self.prepare_code = compile("".join(self.notebook["cells"][4]["source"]), "notebook-prepare", "exec")
        self.env = {"sys": sys, "zipfile": zipfile, "uuid": uuid, "Path": Path, "PurePosixPath": PurePosixPath}
        self.fake_files = types.SimpleNamespace(upload=lambda: {})
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        google.colab, colab.files = colab, self.fake_files
        self.modules = patch.dict(sys.modules, {"google": google, "google.colab": colab})
        self.modules.start()
        self.original_path = list(sys.path)
        self.output = io.StringIO()

    def tearDown(self):
        sys.path[:] = self.original_path
        self.modules.stop()
        self.folder.cleanup()

    def package(self, revision, runner_source=None):
        target = self.root / f"package-{revision}.zip"
        runner_source = runner_source or (
            f"REVISION = {revision!r}\n"
            "def validate_manifest(package):\n"
            "    return {'job_id': REVISION, 'operation':'dub_video', 'target_languages':['es-ES'], 'options': {'require_lipsync': False}}\n"
            "def needs_lipsync(manifest):\n"
            "    return manifest['options']['require_lipsync']\n"
            "def needs_face_analysis(manifest):\n"
            "    return needs_lipsync(manifest) or manifest['options'].get('timing_mode') == 'audio_master'\n"
        )
        bootstrap = (
            f"REVISION = {revision!r}\n"
            "def prepare(runtime, storage, needs, *, needs_face_analysis=False):\n"
            "    global LAST_NEEDS_FACE\n"
            "    LAST_NEEDS_FACE = needs_face_analysis\n"
            "    runtime.mkdir(parents=True, exist_ok=True)\n"
            "    (runtime/'prepared-revision.txt').write_text(REVISION)\n"
            "def environment(runtime):\n"
            "    return {'PREPARE_REVISION':REVISION}\n"
        )
        contents = {
            "worker/dubstudio_runner_v2.py": runner_source,
            "worker/dubstudio_bootstrap.py": bootstrap,
            "worker/dubstudio_colab_worker.py": f"REVISION = {revision!r}\n",
            "worker/dubstudio_audio_master.py": f"REVISION = {revision!r}\n",
            "worker/dubstudio_translate.py": f"REVISION = {revision!r}\n",
            "worker/requirements-app.txt": "# synthetic\n",
            "worker/requirements-musetalk.txt": "# synthetic\n",
        }
        with zipfile.ZipFile(target, "w") as archive:
            for name, source in contents.items():
                archive.writestr(name, source)
        return target

    def upload(self, target):
        self.fake_files.upload = lambda: {str(target): b"fixture"}
        with contextlib.redirect_stdout(self.output):
            exec(self.upload_code, self.env)

    def prepare(self):
        self.env.update(runtime_root=self.root / "runtime", storage_root=self.root / "storage")
        with contextlib.redirect_stdout(self.output):
            exec(self.prepare_code, self.env)

    def test_switching_package_uses_new_runner_and_bootstrap_in_same_python_session(self):
        self.upload(self.package("first"))
        previous = self.env["runner"]
        self.prepare()
        self.assertEqual(self.env["bootstrap"].REVISION, "first")
        self.upload(self.package("second"))
        self.prepare()
        self.assertEqual(self.env["manifest"]["job_id"], "second")
        self.assertEqual(self.env["runner"].REVISION, "second")
        self.assertIsNot(self.env["runner"], previous)
        self.assertEqual(self.env["bootstrap"].REVISION, "second")
        self.assertEqual(self.env["worker_env"]["PREPARE_REVISION"], "second")
        self.assertEqual((self.root / "runtime/prepared-revision.txt").read_text(), "second")
        for variable in ("runner", "bootstrap"):
            expected = self.env["package_root"] / "worker" / f"dubstudio_{'runner_v2' if variable == 'runner' else 'bootstrap'}.py"
            self.assertEqual(Path(self.env[variable].__file__).resolve(), expected.resolve())

    def test_preloaded_modules_from_other_directory_cannot_override_uploaded_code(self):
        for name in ("dubstudio_runner_v2", "dubstudio_bootstrap", "dubstudio_colab_worker", "dubstudio_translate", "dubstudio_audio_master"):
            stale = types.ModuleType(name)
            stale.REVISION = "poison-old-package"
            sys.modules[name] = stale
        self.upload(self.package("current"))
        self.prepare()
        self.assertEqual(self.env["runner"].REVISION, "current")
        self.assertEqual(self.env["bootstrap"].REVISION, "current")
        self.assertNotIn("dubstudio_colab_worker", sys.modules)
        self.assertNotIn("dubstudio_translate", sys.modules)
        self.assertNotIn("dubstudio_audio_master", sys.modules)

    def test_audio_master_without_lipsync_passes_face_preparation_requirement(self):
        code = (
            "def validate_manifest(package):\n"
            "    return {'job_id':'natural', 'operation':'dub_video', 'target_languages':['es-ES'], 'options':{'require_lipsync':False, 'timing_mode':'audio_master'}}\n"
            "def needs_lipsync(manifest):\n"
            "    return False\n"
            "def needs_face_analysis(manifest):\n"
            "    return manifest['options']['timing_mode'] == 'audio_master'\n"
        )
        self.upload(self.package("natural", code))
        self.prepare()
        self.assertIs(self.env["bootstrap"].LAST_NEEDS_FACE, True)

    def test_invalid_upload_does_not_leave_previous_manifest_or_worker_environment(self):
        self.upload(self.package("old"))
        self.prepare()
        invalid = self.root / "video-not-zip.mp4"
        invalid.write_bytes(b"not a zip")
        with self.assertRaisesRegex(RuntimeError, "ZIP válido"):
            self.upload(invalid)
        for variable in ("manifest", "package_root", "runner", "bootstrap", "worker_env"):
            self.assertNotIn(variable, self.env)

    def test_failed_fresh_module_import_is_removed_and_does_not_recover_stale_module(self):
        self.upload(self.package("old"))
        with self.assertRaisesRegex(RuntimeError, "synthetic fresh import error"):
            self.upload(self.package("bad", "raise RuntimeError('synthetic fresh import error')\n"))
        self.assertNotIn("dubstudio_runner_v2", sys.modules)
        self.assertNotIn("runner", self.env)
        self.assertNotIn("manifest", self.env)

    def test_loader_rejects_module_names_outside_explicit_public_worker_set(self):
        self.upload(self.package("current"))
        with self.assertRaisesRegex(RuntimeError, "fora do código esperado"):
            self.env["load_studio_module"]("../personal")

    def test_published_template_has_no_execution_output_or_private_runtime_paths(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell.get("execution_count"))
                self.assertEqual(cell.get("outputs"), [])
                compile("".join(cell["source"]), "published-notebook-cell", "exec")
        serialized = json.dumps(self.notebook)
        self.assertNotRegex(serialized, r"(?i)\b[A-Z]:[\\/]+Users[\\/]+")
        self.assertNotRegex(serialized, r"(?:runtime|validation-colab)[\\/]+(?:jobs|trash|fluxo-)")
        self.assertNotRegex(serialized, r"(?:/jobs/|fluxo-|job_id\s*[=:]\s*['\"]?)[a-f0-9]{12}\b")


if __name__ == "__main__":
    unittest.main()
