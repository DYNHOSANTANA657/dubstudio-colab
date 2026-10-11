"""Approved source snapshots are immutable, media-bound inputs on resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner


class ReviewResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        (self.package / "input").mkdir(parents=True)
        self.video = self.package / "input/video.mp4"
        self.video.write_bytes(b"Protocol fixture; real duration is supplied by independent media probing")
        identity = {"schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": "abcdef123456",
                    "input_sha256": runner.sha256(self.video)}
        self.manifest = {**identity, "operation": "dub_video", "input_file": "input/video.mp4",
                         "target_languages": ["pl-PL", "en-US"],
                         "authorization": {"voice_clone_confirmed": True},
                         "inputs": {"video": {"path": "input/video.mp4", "sha256": identity["input_sha256"]}},
                         "options": {"source_language": "auto", "require_lipsync": False, "review_mode": "before_voice"}}
        self.snapshot = {**identity, "source_language": "pt", "segments": [
            {"id": 1, "start": 0.0, "end": 0.8, "speaker": "SPEAKER_00", "text": "Olá."},
            {"id": 2, "start": 1.0, "end": 2.8, "speaker": "SPEAKER_01", "text": "Boa noite."},
        ]}
        self.review = {**identity, "source_language": "pt", "translations": {
            language: [{**segment, "translated_text": "Cześć." if language == "pl-PL" else "Hello."}
                       for segment in self.snapshot["segments"]] for language in self.manifest["target_languages"]}}
        self.save_bundle()

    def tearDown(self):
        self.temporary.cleanup()

    def save_bundle(self):
        runner.write_json(self.package / "review-source.json", self.snapshot)
        source_hash = runner.sha256(self.package / "review-source.json")
        self.review["source_snapshot_sha256"] = source_hash
        runner.write_json(self.package / "review-approved.json", self.review)
        self.manifest["review_snapshot"] = {
            "path": "review-source.json", "sha256": source_hash,
            "approved_path": "review-approved.json", "approved_sha256": runner.sha256(self.package / "review-approved.json"),
        }
        runner.write_json(self.package / "manifest.json", self.manifest)

    def load(self, duration=3.0):
        return runner.load_approved_snapshot(self.package, runner.validate_manifest(self.package), duration)

    def test_valid_snapshot_preserves_source_and_all_languages(self):
        loaded = self.load()
        self.assertEqual(loaded["source"], self.snapshot)
        self.assertEqual(loaded["translations"], self.review["translations"])
        self.assertEqual(loaded["source_sha256"], self.manifest["review_snapshot"]["sha256"])

    def test_normal_package_does_not_require_snapshot(self):
        (self.package / "review-approved.json").unlink()
        self.manifest.pop("review_snapshot")
        self.assertIsNone(runner.load_approved_snapshot(self.package, self.manifest, 3.0))

    def test_legacy_approval_requires_explicit_reexport(self):
        self.manifest.pop("review_snapshot")
        with self.assertRaisesRegex(runner.PipelineError, "Reexporte.*snapshot"):
            runner.load_approved_snapshot(self.package, self.manifest, 3.0)

    def test_missing_or_tampered_files_rejected(self):
        for name in ("review-source.json", "review-approved.json"):
            with self.subTest(name=name):
                path = self.package / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(runner.PipelineError, "Checksum"):
                    self.load()
                path.unlink()
                with self.assertRaisesRegex(runner.PipelineError, "Checksum"):
                    self.load()
                path.write_bytes(original)

    def test_identity_cannot_change_even_when_file_hashes_are_refreshed(self):
        for payload in (self.snapshot, self.review):
            for key, value in (("job_id", "111111111111"), ("input_sha256", "0" * 64),
                               ("pipeline_version", "99.0.0"), ("schema_version", "1.0")):
                with self.subTest(payload="source" if payload is self.snapshot else "approved", key=key):
                    original = payload[key]
                    payload[key] = value
                    self.save_bundle()
                    with self.assertRaisesRegex(runner.PipelineError, "outro projeto, mídia ou pipeline"):
                        self.load()
                    payload[key] = original
        self.save_bundle()

    def test_other_media_rejected_before_review_is_used(self):
        self.video.write_bytes(b"a different video")
        with self.assertRaisesRegex(runner.PipelineError, "Checksum"):
            self.load()

    def test_approval_must_reference_exact_source_checksum(self):
        self.review["source_snapshot_sha256"] = "0" * 64
        runner.write_json(self.package / "review-approved.json", self.review)
        self.manifest["review_snapshot"]["approved_sha256"] = runner.sha256(self.package / "review-approved.json")
        with self.assertRaisesRegex(runner.PipelineError, "vinculada"):
            runner.load_approved_snapshot(self.package, self.manifest, 3.0)

    def test_snapshot_paths_are_not_user_controlled(self):
        for key in ("path", "approved_path"):
            with self.subTest(key=key):
                modified = copy.deepcopy(self.manifest)
                modified["review_snapshot"][key] = "../other.json"
                with self.assertRaisesRegex(runner.PipelineError, "contrato"):
                    runner.load_approved_snapshot(self.package, modified, 3.0)

    def test_source_language_must_match_approved_and_requested_language(self):
        self.review["source_language"] = "en"
        self.save_bundle()
        with self.assertRaisesRegex(runner.PipelineError, "idioma original|Idioma original"):
            self.load()
        self.review["source_language"] = "pt"
        self.manifest["options"]["source_language"] = "es-ES"
        self.save_bundle()
        with self.assertRaisesRegex(runner.PipelineError, "idioma original configurado"):
            self.load()

    def test_all_and_only_requested_languages_must_be_present(self):
        original = copy.deepcopy(self.review["translations"])
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                self.review["translations"] = copy.deepcopy(original)
                if mode == "missing":
                    self.review["translations"].pop("en-US")
                else:
                    self.review["translations"]["es-ES"] = copy.deepcopy(original["en-US"])
                self.save_bundle()
                with self.assertRaisesRegex(runner.PipelineError, "idiomas solicitados"):
                    self.load()

    def test_source_segment_count_order_and_ids_are_validated(self):
        original = copy.deepcopy(self.snapshot["segments"])
        variants = [[], list(reversed(original)), [original[0], copy.deepcopy(original[0])],
                    [{**original[0], "id": True}], [{**original[0], "id": 1.5}]]
        for segments in variants:
            with self.subTest(segments=segments):
                self.snapshot["segments"] = segments
                self.save_bundle()
                with self.assertRaises(runner.PipelineError):
                    self.load()

    def test_source_times_reject_nonfinite_boolean_and_out_of_media(self):
        original = copy.deepcopy(self.snapshot["segments"])
        for key, value in (("start", float("nan")), ("end", float("inf")), ("start", True),
                           ("end", "2.0"), ("start", -0.1), ("end", 4.01), ("end", 0.0)):
            with self.subTest(key=key, value=value):
                self.snapshot["segments"] = copy.deepcopy(original)
                self.snapshot["segments"][0][key] = value
                self.save_bundle()
                with self.assertRaises(runner.PipelineError):
                    self.load()

    def test_real_duration_must_be_confirmed(self):
        for duration in (float("nan"), float("inf"), 0.0):
            with self.subTest(duration=duration), self.assertRaisesRegex(runner.PipelineError, "duração real"):
                self.load(duration)

    def test_source_speaker_and_text_are_not_optional(self):
        for key, value in (("speaker", "../../escape"), ("speaker", None), ("text", ""), ("text", 1)):
            with self.subTest(key=key):
                old = self.snapshot["segments"][0][key]
                self.snapshot["segments"][0][key] = value
                self.save_bundle()
                with self.assertRaises(runner.PipelineError):
                    self.load()
                self.snapshot["segments"][0][key] = old

    def test_approved_ids_times_speakers_and_original_text_cannot_change(self):
        original = copy.deepcopy(self.review["translations"]["pl-PL"])
        for key, value in (("id", "1"), ("start", 0.02), ("start", 0.001), ("end", 0.801),
                           ("speaker", "SPEAKER_99"), ("text", "Outro texto."),
                           ("start", float("nan")), ("end", float("inf")), ("start", True)):
            with self.subTest(key=key, value=value):
                self.review["translations"]["pl-PL"] = copy.deepcopy(original)
                self.review["translations"]["pl-PL"][0][key] = value
                self.save_bundle()
                with self.assertRaises(runner.PipelineError):
                    self.load()

    def test_approved_translation_is_complete_nonempty_text(self):
        original = copy.deepcopy(self.review["translations"]["pl-PL"])
        for value in ("", None, 3):
            with self.subTest(value=value):
                self.review["translations"]["pl-PL"] = copy.deepcopy(original)
                self.review["translations"]["pl-PL"][0]["translated_text"] = value
                self.save_bundle()
                with self.assertRaisesRegex(runner.PipelineError, "traduzida vazia"):
                    self.load()
        self.review["translations"]["pl-PL"] = original[:1]
        self.save_bundle()
        with self.assertRaisesRegex(runner.PipelineError, "todos os segmentos"):
            self.load()

    def test_code_fingerprint_change_never_retranscribes_approved_snapshot(self):
        work = self.root / "work"
        stale = work / self.manifest["job_id"] / "shared/diarization.json"
        altered_asr = copy.deepcopy(self.snapshot["segments"])
        altered_asr[1]["start"] += 0.02  # The real fresh-ASR failure was exactly this drift.
        runner.write_json(stale, {"language": "pt", "segments": altered_asr})
        seen = []

        class StopBeforeGpuModels(Exception):
            pass

        def fake_stage(cache, key, label, config, required):
            seen.append(key)
            if key == "separation":
                for file in required:
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(b"fake audio for protocol test")
            elif key == "references":
                frozen = runner.read_json(Path(config["segments"]))
                self.assertEqual(frozen["segments"], self.snapshot["segments"])
                self.assertEqual(frozen["approved_source_snapshot_sha256"], self.manifest["review_snapshot"]["sha256"])
                self.assertTrue(cache.used["approved_source_snapshot"])
                self.assertEqual(cache.worker_code, "a-new-fixed-worker-fingerprint")
                raise StopBeforeGpuModels()
            else:
                self.fail(f"Unexpected nondeterministic stage after approval: {key}")

        with patch.object(bootstrap, "gpu_precheck", return_value="Mock T4, not live validation"), \
             patch.object(runner, "media_info", return_value={"duration": 3.0, "streams": [{"codec_type": "video", "duration": "3.0"}, {"codec_type": "audio", "duration": "3.0"}]}), \
             patch.object(runner, "worker_fingerprint", return_value="a-new-fixed-worker-fingerprint"), \
             patch.object(runner.StageCache, "stage", fake_stage):
            with self.assertRaises(StopBeforeGpuModels):
                runner.process(self.package, work, self.root / "storage", self.root / "result.zip")
        self.assertEqual(seen, ["separation", "references"])


if __name__ == "__main__":
    unittest.main()
