"""Safe public failure diagnostics; synthetic subprocesses, never a GPU claim."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import dubstudio_runner_v2 as runner


class FailureDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.analysis = self.shared / "active-speaker.json"
        self.config = {"directory": str(self.shared), "stage": "analysis", "output": str(self.analysis)}

    def tearDown(self):
        self.temporary.cleanup()

    def subprocess_error(self, text, exit_code=7, tail_repeat=0):
        log = self.root / "pipeline.log"
        command = [sys.executable, "-u", "-c", f"import sys; print({text!r}); print('private unrelated diagnostic\\n' * {tail_repeat}); sys.exit({exit_code})", "--stage", "analysis"]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                runner.run_command(command, log)
            except runner.PipelineError as exc:
                return exc
        return None

    def ambiguity(self):
        points = [(43.16, 43.20), (43.32, 43.36), (43.64, 43.68), (43.68, 43.72), (44.20, 44.24), (44.24, 44.28)]
        runner.write_json(self.analysis, {"ambiguity": [{"start": start, "end": end, "reason": "múltiplos rostos candidatos"} for start, end in points], "private_boxes": ["do not export"], "normalized_video": "private runtime path"})

    def test_stage_marker_survives_long_later_log_without_exporting_tail(self):
        exc = self.subprocess_error("ERRO_ETAPA=Rosto/locutor ambíguo. Corrija a seleção.", tail_repeat=3000)
        self.assertIn("Rosto/locutor ambíguo", str(exc))
        self.assertNotIn("private unrelated", str(exc))
        self.assertEqual(exc.diagnostic["stage"], "analysis")
        self.assertEqual(exc.diagnostic["exit_code"], 7)
        self.assertEqual(exc.diagnostic["log_file"], "pipeline.log")
        self.assertLessEqual(len(str(exc)), 1200)

    def test_unmarked_or_unanchored_text_never_becomes_public_cause(self):
        for text in ("secret private traceback", "DEBUG ERRO_ETAPA=secret private traceback"):
            with self.subTest(text=text):
                exc = self.subprocess_error(text)
                self.assertIn("código 7", str(exc))
                self.assertNotIn("secret private", str(exc))

    def test_error_marker_redacts_known_credentials_urls_and_paths(self):
        secret = "ghp_PRIVATE_TEST_VALUE"
        text = 'ERRO_ETAPA=Falha api_key="fake-key" token=fake-token password=privatepass Bearer privatebearer https://example.test/file?key=privatequery C:\\Users\\private\\voice.wav hf_PRIVATE_TEST_TOKEN ' + secret
        exc = self.subprocess_error(text)
        for value in ("fake-key", "fake-token", "privatepass", "privatebearer", "privatequery", "C:\\Users", "hf_PRIVATE", secret):
            self.assertNotIn(value, str(exc))
        self.assertIn("omitida", str(exc))

    def test_marker_size_and_controls_are_bounded(self):
        exc = self.subprocess_error("ERRO_ETAPA=\x1b[31mFalha\x00" + "x" * 20000)
        self.assertLessEqual(len(str(exc)), 1200)
        self.assertNotIn("\x1b", str(exc))
        self.assertNotIn("\x00", str(exc))

    def test_exit_zero_does_not_fail_from_marker_text_alone(self):
        self.assertIsNone(self.subprocess_error("ERRO_ETAPA=example text in a successful process", 0))

    def test_analysis_exports_only_merged_bounded_intervals(self):
        self.ambiguity()
        failure = runner.stage_failure(runner.PipelineError("Rosto/locutor ambíguo."), "analysis", "Confirmação dos rostos e locutores", self.config)
        self.assertIn("43.16–43.20s", str(failure))
        self.assertIn("44.20–44.28s", str(failure))
        self.assertEqual(len(failure.diagnostic["ambiguity"]), 4)
        self.assertEqual(failure.diagnostic["ambiguity_total_intervals"], 4)
        self.assertFalse(failure.diagnostic["ambiguity_truncated"])
        self.assertNotIn("private", json.dumps(failure.diagnostic))
        self.assertNotIn("boxes", json.dumps(failure.diagnostic))

    def test_analysis_diagnostic_rejects_invalid_windows_and_unknown_reason(self):
        runner.write_json(self.analysis, {"ambiguity": [{"start": float("nan"), "end": 1}, {"start": True, "end": 1}, {"start": -1, "end": 2}, {"start": 2, "end": 1}, {"start": 1, "end": 2, "reason": "token=private-secret"}]})
        diagnostic = runner.analysis_ambiguity_diagnostic(self.config)
        self.assertEqual(diagnostic["ambiguity"], [{"start": 1.0, "end": 2.0, "reason": "Ambiguidade facial detectada"}])

    def test_analysis_diagnostic_does_not_read_outside_or_oversized_file(self):
        outside = self.root / "outside.json"
        runner.write_json(outside, {"ambiguity": [{"start": 1, "end": 2}]})
        self.assertEqual(runner.analysis_ambiguity_diagnostic({**self.config, "output": str(outside)}), {})
        self.analysis.write_bytes(b"x" * (2 * 1024**2 + 1))
        self.assertEqual(runner.analysis_ambiguity_diagnostic(self.config), {})

    def test_many_intervals_are_truncated_without_inventing_complete_diagnostic(self):
        runner.write_json(self.analysis, {"ambiguity": [{"start": index * 2, "end": index * 2 + 0.04, "reason": "múltiplos rostos candidatos"} for index in range(80)]})
        failure = runner.stage_failure(runner.PipelineError("Rosto/locutor ambíguo."), "analysis", "Rostos", self.config)
        self.assertEqual(len(failure.diagnostic["ambiguity"]), 64)
        self.assertEqual(failure.diagnostic["ambiguity_total_intervals"], 80)
        self.assertTrue(failure.diagnostic["ambiguity_truncated"])
        self.assertLess(len(json.dumps(failure.diagnostic)), 20000)

    def test_failed_stage_has_diagnostic_but_no_success_checkpoint(self):
        self.ambiguity()
        cache = runner.StageCache(self.root, self.root)
        with patch.object(runner, "run_command", side_effect=runner.PipelineError("Rosto/locutor ambíguo.")), patch.object(runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(runner.PipelineError) as raised:
                cache.stage("analysis", "Confirmação dos rostos", self.config, [self.analysis])
        self.assertEqual(raised.exception.diagnostic["stage"], "analysis")
        self.assertEqual(len(raised.exception.diagnostic["ambiguity"]), 4)
        self.assertFalse((self.root / "checkpoints/analysis.json").exists())

    def test_global_failed_bundle_preserves_concrete_reason_and_small_schema(self):
        package = self.root / "package"
        video = package / "input/video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"immutable protocol fixture, not real media")
        manifest = {"schema_version": runner.SCHEMA_VERSION, "pipeline_version": runner.PIPELINE_VERSION, "job_id": "abcdef123456", "operation": "dub_video", "input_file": "input/video.mp4", "input_sha256": runner.sha256(video), "inputs": {"video": {"path": "input/video.mp4", "sha256": runner.sha256(video)}}, "target_languages": ["pl-PL"], "authorization": {"voice_clone_confirmed": True}, "options": {}}
        runner.write_json(package / "manifest.json", manifest)
        self.ambiguity()
        failure = runner.stage_failure(runner.PipelineError("Rosto/locutor ambíguo."), "analysis", "Confirmação dos rostos", self.config)
        output = self.root / "failure.zip"
        argv = ["runner.py", "--package-root", str(package), "--work-root", str(self.root / "work"), "--storage-root", str(self.root / "storage"), "--output", str(output)]
        with patch.object(runner.sys, "argv", argv), patch.object(runner, "process", side_effect=failure), patch("builtins.print"), self.assertRaises(runner.PipelineError):
            runner.main()
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("result.json"))
            record = result["outputs"][0]
            report = json.loads(archive.read(record["files"]["report"]))
            self.assertEqual(set(archive.namelist()), {"result.json", record["files"]["report"]})
        for item in (record, report):
            self.assertEqual(item["status"], "failed")
            self.assertEqual(item["diagnostic"]["stage"], "analysis")
            self.assertIn("43.16–43.20s", item["error"])
            self.assertEqual(len(item["diagnostic"]["ambiguity"]), 4)
            self.assertNotIn("private", json.dumps(item))


if __name__ == "__main__":
    unittest.main()
