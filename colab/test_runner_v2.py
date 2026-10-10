"""Protocol tests do not claim quality/model/GPU validation."""
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner
import dubstudio_translate as translator


class PackageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "input").mkdir()
        self.input = self.root / "input/video.mp4"
        self.input.write_bytes(b"protocol fixture, not media")
        self.manifest = {
            "schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": "abcdef123456",
            "operation": "dub_video", "input_file": "input/video.mp4", "input_sha256": runner.sha256(self.input),
            "target_languages": ["pl-PL", "en-US"], "authorization": {"voice_clone_confirmed": True},
            "inputs": {"video": {"path": "input/video.mp4", "sha256": runner.sha256(self.input)}}, "options": {},
        }
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        runner.write_json(self.root / "manifest.json", self.manifest)

    def test_valid_package_and_norwegian_alias(self):
        self.manifest["target_languages"].append("nb-NO")
        self.save()
        self.assertEqual(runner.validate_manifest(self.root)["job_id"], "abcdef123456")

    def test_wrong_checksum(self):
        self.input.write_bytes(b"changed media")
        with self.assertRaisesRegex(runner.PipelineError, "Checksum"):
            runner.validate_manifest(self.root)

    def test_no_generic_fallback(self):
        self.manifest["target_languages"] = ["uk-UA"]
        self.save()
        with self.assertRaisesRegex(runner.PipelineError, "Nenhuma voz genérica"):
            runner.validate_manifest(self.root)

    def test_authorization_required(self):
        self.manifest["authorization"] = {}
        self.save()
        with self.assertRaisesRegex(runner.PipelineError, "autorização"):
            runner.validate_manifest(self.root)

    def test_lipsync_operation_accepts_und_without_voice_fallback(self):
        self.manifest["operation"] = "lipsync_video"
        self.manifest["target_languages"] = ["und"]
        self.manifest["inputs"]["audio"] = dict(self.manifest["inputs"]["video"])
        self.save()
        self.assertEqual(runner.validate_manifest(self.root)["target_languages"], ["und"])

    def test_optional_lipsync_is_boolean_not_string(self):
        self.manifest["options"]["require_lipsync"] = "false"
        self.save()
        with self.assertRaisesRegex(runner.PipelineError, "explicitamente"):
            runner.validate_manifest(self.root)

    def test_lipsync_flag_follows_operation(self):
        for operation, flag, expected in (("dub_video", False, False), ("dub_video", True, True), ("lipsync_video", False, True), ("text_to_voice", True, False)):
            with self.subTest(operation=operation, flag=flag):
                self.assertEqual(runner.needs_lipsync({"operation": operation, "options": {"require_lipsync": flag}}), expected)
        self.assertTrue(runner.needs_lipsync({"operation": "dub_video"}))

    def test_unsafe_paths_rejected(self):
        for path in ("../escape", "C:/escape", "/escape", "input/../../escape", "..\\escape"):
            with self.subTest(path=path), self.assertRaises(runner.PipelineError):
                runner.safe_path(self.root, path)

    def test_unsafe_zip_rejected(self):
        archive = self.root / "bad.zip"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("../../escape.txt", "bad")
        with self.assertRaises(runner.PipelineError):
            runner.unpack_package(archive, self.root / "extracted")

    def test_review_keeps_source_timeline_and_speakers(self):
        source = [{"id": 1, "start": 0.3, "end": 1.2, "speaker": "SPEAKER_00", "text": "Olá"}]
        revised = [{**source[0], "translated_text": "Cześć"}]
        review = {"schema_version": "2.0", "job_id": self.manifest["job_id"], "input_sha256": self.manifest["input_sha256"], "translations": {lang: copy.deepcopy(revised) for lang in self.manifest["target_languages"]}}
        self.assertEqual(len(runner.validate_review(review, self.manifest, source)), 2)
        review["translations"]["pl-PL"][0]["speaker"] = "SPEAKER_01"
        with self.assertRaisesRegex(runner.PipelineError, "locutores"):
            runner.validate_review(review, self.manifest, source)

    def test_review_other_project_rejected(self):
        review = {"schema_version": "2.0", "job_id": "111111111111", "input_sha256": self.manifest["input_sha256"]}
        with self.assertRaisesRegex(runner.PipelineError, "outro projeto"):
            runner.validate_review(review, self.manifest, [])

    def test_result_checksums_and_paths(self):
        output = self.root / "outputs/pl-PL/report.json"
        runner.write_json(output, {"status": "failed"})
        relative = output.relative_to(self.root).as_posix()
        metadata = {"outputs": [{"language": "pl-PL", "status": "failed", "files": {"report": relative}, "checksums": {relative: runner.sha256(output)}}]}
        bundle = self.root / "result.zip"
        runner.bundle(self.root, bundle, "result.json", metadata)
        with zipfile.ZipFile(bundle) as archive:
            self.assertEqual(set(archive.namelist()), {"result.json", relative})
            self.assertEqual(json.loads(archive.read("result.json")), metadata)


class IntegrityTests(unittest.TestCase):
    def test_same_size_changed_checkpoint_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "source", Path(temporary) / "target"
            source.mkdir()
            target.mkdir()
            (source / "approval.json").write_text("abcd")
            (target / "approval.json").write_text("efgh")
            bootstrap.copy_missing(source, target, verify=True)
            self.assertEqual((target / "approval.json").read_text(), "abcd")

    def test_stage_cache_does_not_reuse_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "job"
            root.mkdir()
            cache = runner.StageCache(root, Path(temporary) / "saved")
            output = root / "voice.wav"
            config = {"stage": "test", "directory": str(root)}
            def fake_stage(*args, **kwargs):
                output.write_bytes(b"good")
            with patch.object(runner, "run_command", side_effect=fake_stage) as stage:
                cache.stage("voice", "Voice protocol fixture", config, [output])
                cache.stage("voice", "Voice protocol fixture", config, [output])
                self.assertEqual(stage.call_count, 1)
                output.write_bytes(b"fake")
                cache.stage("voice", "Voice protocol fixture", config, [output])
                self.assertEqual(stage.call_count, 2)

    def test_gpu_precheck_precedes_install(self):
        with patch.object(bootstrap.shutil, "which", return_value=None), patch.object(bootstrap, "execute") as execute:
            with self.assertRaisesRegex(RuntimeError, "GPU indisponível"):
                bootstrap.prepare(Path("unused"), Path("unused"))
            execute.assert_not_called()

    def test_no_lipsync_bootstrap_skips_both_face_repositories_and_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime, storage = Path(temporary) / "runtime", Path(temporary) / "storage"
            ready = runtime / "app-env/.ready.json"
            requirements = Path(bootstrap.__file__).with_name("requirements-app.txt")
            fingerprint = hashlib.sha256(requirements.read_bytes() + bootstrap.PIPELINE_VERSION.encode()).hexdigest()
            runner.write_json(ready, {"fingerprint": fingerprint})
            with patch.object(bootstrap, "gpu_precheck", return_value="GPU protocol fixture"), patch.object(bootstrap.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(100 * 1024**3, 0, 100 * 1024**3)), patch.object(bootstrap.shutil, "which", return_value="available"), patch.object(bootstrap, "execute") as execute, patch.object(bootstrap, "clone_at") as clone, patch.object(bootstrap, "download_model_weights") as weights, patch.object(bootstrap, "persist_cache"):
                bootstrap.prepare(runtime, storage, needs_lipsync=False)
            clone.assert_not_called()
            weights.assert_not_called()
            self.assertFalse((runtime / "musetalk-env").exists())
            self.assertEqual(execute.call_args_list[0].args[0], ["uv", "python", "install", "3.10"])

    def test_changed_worker_code_invalidates_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "job"
            root.mkdir()
            output = root / "voice.wav"
            config = {"stage": "test", "directory": str(root)}
            with patch.object(runner, "worker_fingerprint", return_value="code-v1"):
                cache = runner.StageCache(root, Path(temporary) / "saved")
            with patch.object(runner, "run_command", side_effect=lambda *args, **kwargs: output.write_bytes(b"good")) as stage:
                cache.stage("voice", "Voice code fixture", config, [output])
                with patch.object(runner, "worker_fingerprint", return_value="code-v2"):
                    updated = runner.StageCache(root, Path(temporary) / "saved")
                updated.stage("voice", "Voice code fixture", config, [output])
                self.assertEqual(stage.call_count, 2)

    def test_missing_reference_wav_requires_rebuild(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "job"
            root.mkdir()
            cache = runner.StageCache(root, Path(temporary) / "saved")
            refs, wave = root / "references.json", root / "reference-SPEAKER_00.wav"
            def make_references(*args, **kwargs):
                runner.write_json(refs, {"SPEAKER_00": str(wave)})
                wave.write_bytes(b"reference fixture")
            with patch.object(runner, "run_command", side_effect=make_references) as stage:
                cache.stage("references", "Reference fixture", {"stage": "test"}, [refs, wave])
                wave.unlink()
                cache.stage("references", "Reference fixture", {"stage": "test"}, [refs, wave])
                self.assertEqual(stage.call_count, 2)


class SpeechAndFaceTests(unittest.TestCase):
    def test_full_speech_tempo_does_not_clip(self):
        for ratio in (0.15, 0.5, 1.0, 1.8, 3.1, 8.0):
            factors = [float(part.split("=")[1]) for part in runner.tempo_filters(ratio).split(",")]
            self.assertTrue(all(0.5 <= value <= 2.0 for value in factors))
            self.assertAlmostEqual(math.prod(factors), ratio, places=6)

    def test_split_translation_keeps_all_words(self):
        text = "uma frase importante sem perder palavras " * 20
        self.assertEqual(" ".join(translator.split_text(text, 40)), " ".join(text.split()))

    def test_script_validation_accepts_short_polish_and_rejects_cyrillic(self):
        self.assertTrue(translator.is_sane_translation("Tak", "pl-PL"))
        self.assertFalse(translator.is_sane_translation("Привет всем", "pl-PL"))
        self.assertTrue(translator.is_sane_translation("Привет всем", "ru-RU"))

    def test_single_face_has_correct_speaker(self):
        track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
        transcript = [{"start": 0, "end": 1, "speaker": "SPEAKER_01"}]
        groups, ambiguity = runner.identity_segments([track], [[1] * 25], 25, 100, 100, transcript, "auto", "dub_video")
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["speaker"], "SPEAKER_01")

    def test_simultaneous_faces_not_silently_preserved(self):
        track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
        groups, ambiguity = runner.identity_segments([track, track], [[1] * 25, [1] * 25], 25, 100, 100, [], "auto", "dub_video")
        self.assertEqual(groups, [])
        self.assertEqual(len(ambiguity), 25)

    def test_multiple_voices_same_track_requires_review(self):
        track = {"track": {"frame": list(range(50)), "bbox": [[5, 5, 80, 80]] * 50}}
        transcript = [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}, {"start": 1, "end": 2, "speaker": "SPEAKER_01"}]
        groups, ambiguity = runner.identity_segments([track], [[1] * 50], 50, 100, 100, transcript, "auto", "dub_video")
        self.assertEqual(groups, [])
        self.assertTrue(ambiguity)

    def test_user_audio_can_select_detected_face(self):
        track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
        groups, ambiguity = runner.identity_segments([track], [[-1] * 25], 25, 100, 100, [], {"track_id": 0}, "lipsync_video")
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["speaker"], "USER_AUDIO")

    def test_normalized_bbox_can_resolve_dub_face_choice(self):
        first = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 45, 60]] * 25}}
        second = {"track": {"frame": list(range(25)), "bbox": [[50, 5, 90, 60]] * 25}}
        selection = {"bbox": [0.05, 0.05, 0.40, 0.55], "normalized": True, "frame_time": 0.0}
        groups, ambiguity = runner.identity_segments([first, second], [[-1] * 25, [1] * 25], 25, 100, 100, [{"start": 0, "end": 1, "speaker": "SPEAKER_00"}], selection, "dub_video")
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["track_id"], 0)
        self.assertIn("seleção do usuário", groups[0]["coordinate_source"])

    def test_wrong_bbox_does_not_invent_face(self):
        track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 45, 60]] * 25}}
        groups, ambiguity = runner.identity_segments([track], [[1] * 25], 25, 100, 100, [], {"bbox": [.7, .7, .1, .1], "normalized": True}, "lipsync_video")
        self.assertEqual(groups, [])
        self.assertTrue(ambiguity)

    def test_silent_reference_is_rejected_before_model_imports(self):
        silence = subprocess.CompletedProcess([], 0, b"\x00\x00" * 32000, b"")
        with patch.object(runner.subprocess, "run", return_value=silence), self.assertRaisesRegex(runner.PipelineError, "Referência sem fala"):
            runner.validate_voice_reference(Path("silent-reference.wav"))


class ProcessProtocolTests(unittest.TestCase):
    """Exercise orchestration with synthetic stage artifacts, never model quality."""
    save = PackageTests.save
    tearDown = PackageTests.tearDown

    def setUp(self):
        PackageTests.setUp(self)
        self.manifest["target_languages"] = ["pl-PL"]
        self.manifest["options"]["require_lipsync"] = False
        self.save()
        self.stage_configs = []

    def fake_stage(self, cache, key, label, config, required):
        self.stage_configs.append((key, config, list(required)))
        stage = config["stage"]
        directory = Path(config["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        segments = [{"id": 1, "start": 0.0, "end": 1.8, "speaker": "SPEAKER_00", "text": "Olá, mundo.", "translated_text": "Witaj, świecie."}]
        if stage == "separation":
            for path in required:
                path.write_bytes(b"separation protocol fixture")
        elif stage in {"asr", "diarization", "translation"}:
            if stage == "translation" and config["language"] == "en-US":
                raise runner.PipelineError("Translation protocol failure")
            runner.write_json(Path(config["output"]), {"language": "pt", "segments": segments})
        elif stage == "references":
            wave = directory / "reference-SPEAKER_00.wav"
            wave.write_bytes(b"reference protocol fixture")
            runner.write_json(Path(config["output"]), {"SPEAKER_00": str(wave)})
        elif stage == "synthesis":
            (directory / "dialogue.wav").write_bytes(b"voice protocol fixture")
            runner.write_json(directory / "voice-proof.json", {"voice_clone": {"required": True, "confirmed": True, "authorized": True}})
        elif stage == "mix":
            Path(config["output"]).write_bytes(b"mixed audio protocol fixture")
        elif stage == "export":
            Path(config["output"]).write_bytes(b"export protocol fixture, not playable")
            runner.write_json(Path(config["validation"]), {"video_duration": 2.0, "audio_duration": 2.0})
        else:
            raise AssertionError(f"Unexpected facial stage in no-lipsync protocol test: {stage}")
        cache.seconds[key], cache.used[key] = 0.001, False

    def execute_fixture(self):
        output = self.root / "result.zip"
        def fixture_command(command, *args, **kwargs):
            Path(command[-1]).write_bytes(b"MP3 protocol fixture")
        with patch.object(bootstrap, "gpu_precheck", return_value="GPU protocol fixture, not verified"), patch.object(runner, "media_info", return_value={"duration": 2.0}), patch.object(runner.StageCache, "stage", autospec=True, side_effect=self.fake_stage), patch.object(runner, "run_command", side_effect=fixture_command):
            result = runner.process(self.root, self.root / "work", self.root / "storage", output)
        return result, output

    def test_no_lipsync_dub_preserves_source_and_never_claims_lipsync(self):
        result, output = self.execute_fixture()
        record = result["outputs"][0]
        self.assertEqual(record["status"], "complete")
        with zipfile.ZipFile(output) as archive:
            report = json.loads(archive.read(record["files"]["report"]))
        self.assertEqual({key: report["lipsync"][key] for key in ("required", "confirmed", "applied")}, {"required": False, "confirmed": False, "applied": False})
        self.assertEqual(report["mode_label"], "Dublagem sem lip-sync")
        self.assertNotIn("analysis", [key for key, _, _ in self.stage_configs])
        self.assertFalse(any(key.startswith("lipsync") for key, _, _ in self.stage_configs))
        exported = next(config for key, config, _ in self.stage_configs if key.startswith("export"))
        self.assertEqual(exported["video"], exported["source"])
        references = next(required for key, _, required in self.stage_configs if key == "references")
        self.assertTrue(any(path.suffix == ".wav" for path in references))

    def test_one_failed_translation_does_not_block_other_language(self):
        self.manifest["target_languages"].append("en-US")
        self.save()
        result, output = self.execute_fixture()
        self.assertEqual([(item["language"], item["status"]) for item in result["outputs"]], [("pl-PL", "complete"), ("en-US", "failed")])
        self.assertIn("Translation protocol failure", result["outputs"][1]["error"])


class RealMediaValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = shutil.which("ffmpeg")
        candidate = Path("C:/Users/edeni/Downloads/ffmpeg-8.1-essentials_build/bin/ffmpeg.exe")
        cls.ffmpeg = Path(binary) if binary else candidate
        if not cls.ffmpeg.is_file():
            raise unittest.SkipTest("FFmpeg unavailable for real media-stream validation")

    def test_real_streams_accept_full_audio_and_reject_truncated_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete, shortened = root / "complete.mp4", root / "short-audio.mp4"
            for output, audio_duration in ((complete, 2.0), (shortened, 0.3)):
                subprocess.run([str(self.ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x284:r=25:d=2", "-f", "lavfi", "-i", f"sine=frequency=220:duration={audio_duration}", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(output)], check=True, capture_output=True)
            with patch.dict(os.environ, {"PATH": str(self.ffmpeg.parent) + os.pathsep + os.environ.get("PATH", "")}):
                proof = runner.verify_final_media(complete, 2.0)
                self.assertLessEqual(proof["timing_tolerance_seconds"], 1.0)
                with self.assertRaisesRegex(runner.PipelineError, "Resultado incompleto"):
                    runner.verify_final_media(shortened, 2.0)


class NotebookTests(unittest.TestCase):
    def test_every_code_cell_compiles(self):
        notebook = Path(__file__).parent.parent / "DubStudio_Colab.ipynb"
        payload = json.loads(notebook.read_text(encoding="utf-8"))
        for index, cell in enumerate(payload["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook cell {index}", "exec")
        joined = json.dumps(payload)
        self.assertNotIn("--clear", joined)
        self.assertIn("nvidia-smi", "".join(payload["cells"][1]["source"]))
        self.assertIn("cache_on_drive", joined)
        self.assertIn("Salvar no Drive (autorizar)", joined)


if __name__ == "__main__":
    unittest.main()
