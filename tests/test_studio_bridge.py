"""Integrity and actual A/V duration acceptance for offline Studio v2 results."""

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from languages import languages_payload, supports_original_voice
from pipeline import Job, PipelineError, probe_video
from studio_bridge import (
    PIPELINE_VERSION, SCHEMA_VERSION, export_approved_review, import_review,
    import_studio_result, prepare_studio_package,
)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class StudioBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        cls.source = Path(cls.shared.name) / "source.mp4"
        subprocess.run([
            shutil.which("ffmpeg"), "-y", "-f", "lavfi", "-i", "color=c=blue:s=160x90:r=25",
            "-f", "lavfi", "-i", "sine=frequency=550:sample_rate=48000", "-t", "3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.source),
        ], check=True, capture_output=True)
        cls.truncated = Path(cls.shared.name) / "truncated.mp4"
        subprocess.run([
            shutil.which("ffmpeg"), "-y", "-i", str(cls.source), "-filter_complex",
            "[0:v]trim=duration=0.4,setpts=PTS-STARTPTS[v]", "-map", "[v]", "-map", "0:a:0",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.truncated),
        ], check=True, capture_output=True)
        cls.wav = Path(cls.shared.name) / "voice.wav"
        cls.mp3 = Path(cls.shared.name) / "voice.mp3"
        for target in (cls.wav, cls.mp3):
            subprocess.run([shutil.which("ffmpeg"), "-y", "-i", str(cls.source), "-vn", str(target)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        source = self.root / "original.mp4"
        shutil.copy2(self.source, source)
        self.job = Job(id="abc123abc123", directory=self.root, input_path=source, options={
            "operation": "dub_video", "target_languages": ["pl-PL", "en-US"],
            "voice_clone_consent": True, "require_lipsync": True,
        }, metadata=probe_video(source))
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))

    def tearDown(self):
        self.folder.cleanup()

    def report(self, language):
        return {
            "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
            "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"],
            "target_language": language, "summary": {"duration": 3.0},
            "operation": self.manifest["operation"], "input_checksums": {role: item["sha256"] for role, item in self.manifest["inputs"].items()},
            "voice_clone": {"required": True, "confirmed": True, "authorized": True},
            "lipsync": {"required": True, "confirmed": True, "applied": True, "ambiguity": [], "confirmed_segments": [
                {"start": 0.0, "end": 1.0, "coordinate_source": "TalkNet"},
                {"start": 2.0, "end": 3.0, "coordinate_source": "TalkNet"},
            ]},
            "preserved_no_speaker_visual_segments": [{"start": 1.0, "end": 2.0}],
            "no_lipsync_segments": [],
        }

    def make_result(self, *, media=None, edit=None, extra=None, failed_language=None, checksum_error=False, segments_edit=None, audio_media=None):
        result = {
            "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
            "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "outputs": [],
            "operation": self.manifest["operation"], "input_checksums": {role: item["sha256"] for role, item in self.manifest["inputs"].items()},
        }
        blobs = {}
        for language in self.manifest["target_languages"]:
            if language == failed_language:
                result["outputs"].append({"language": language, "status": "failed", "error": "GPU memory unavailable"})
                continue
            report = self.report(language)
            if edit:
                edit(report)
            base = f"outputs/{language}"
            segments = {"target_language": language, "segments": [
                {"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"},
            ]}
            operation = self.manifest["operation"]
            files = {"report": f"{base}/report.json"}
            blobs[files["report"]] = json.dumps(report).encode()
            if operation == "text_to_voice":
                for role, source in (("wav", self.wav), ("mp3", self.mp3)):
                    files[role] = f"{base}/voice.{role}"
                    blobs[files[role]] = (audio_media or {}).get(role, source).read_bytes()
                segments["segments"][0].update(end=report["summary"]["duration"], text=self.job.options["text"], translated_text=self.job.options["text"])
                if segments_edit:
                    segments_edit(segments)
                files["segments"] = f"{base}/voice-segments.json"
                blobs[files["segments"]] = json.dumps(segments).encode()
            else:
                files["video"] = f"{base}/video.mp4"
                blobs[files["video"]] = (media or self.source).read_bytes()
                if operation == "dub_video":
                    if segments_edit:
                        segments_edit(segments)
                    files.update({"segments": f"{base}/segments.json", "original_srt": f"{base}/original.srt", "translation_srt": f"{base}/translation.srt"})
                    blobs[files["segments"]] = json.dumps(segments).encode()
                    for role in ("original_srt", "translation_srt"):
                        blobs[files[role]] = "1\n00:00:00,000 --> 00:00:01,000\nTexto\n".encode()
            checksums = {path: hashlib.sha256(blobs[path]).hexdigest() for path in files.values()}
            if checksum_error:
                checksums[files.get("video", files["report"])] = "0" * 64
            result["outputs"].append({"language": language, "status": "complete", "files": files, "checksums": checksums})
        output = self.root / "result.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("result.json", json.dumps(result))
            for path, content in blobs.items():
                archive.writestr(path, content)
            if extra:
                for path, content in extra:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        archive.writestr(path, content)
        return output

    def test_package_contains_v2_identity_inputs_and_only_public_code(self):
        with zipfile.ZipFile(self.root / "pacote-colab.zip") as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["schema_version"], "2.0")
            self.assertEqual(manifest["target_languages"], ["pl-PL", "en-US"])
            self.assertEqual(manifest["inputs"]["video"]["sha256"], hashlib.sha256(archive.read(manifest["input_file"])).hexdigest())
            self.assertFalse(any("runtime/" in name or "config.json" in name for name in archive.namelist()))

    def test_drive_cache_preference_is_explicit_and_survives_review_export(self):
        self.assertIs(self.manifest["options"]["cache_on_drive"], True)
        for preference in (True, False):
            with self.subTest(preference=preference):
                self.job.options["cache_on_drive"] = preference
                package = prepare_studio_package(self.job)
                self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
                with zipfile.ZipFile(package) as archive:
                    self.assertIs(json.loads(archive.read("manifest.json"))["options"]["cache_on_drive"], preference)
                review = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "job_id": self.job.id,
                          "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
                          "translations": {language: [{"id": 1, "start": 0.0, "end": 1.0,
                                         "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"}]
                                           for language in self.manifest["target_languages"]}}
                review_zip = self.root / "review-cache.zip"
                with zipfile.ZipFile(review_zip, "w") as archive:
                    archive.writestr("review.json", json.dumps(review))
                import_review(self.job, review_zip)
                with zipfile.ZipFile(export_approved_review(self.job, review)) as archive:
                    self.assertIs(json.loads(archive.read("manifest.json"))["options"]["cache_on_drive"], preference)
        self.job.options["cache_on_drive"] = "false"
        with self.assertRaisesRegex(PipelineError, "preferência de cache"):
            prepare_studio_package(self.job)

    def test_import_accepts_multiple_confirmed_scenes_and_preserves_non_speaker_frames(self):
        result = import_studio_result(self.job, self.make_result())
        self.assertEqual(result["success_count"], 2)
        self.assertTrue(self.job.output_path.is_file())
        for output in result["outputs"]:
            self.assertTrue((self.root / output["files"]["video"]).is_file())
            self.assertEqual(output["validation"]["media"]["video"]["video_duration"], 3.0)
        first = self.job.output_path
        import_studio_result(self.job, self.make_result())
        self.assertTrue(first.exists(), "previous validated media must survive re-import")
        self.assertNotEqual(first, self.job.output_path)

    def test_partial_result_records_failure_but_only_publishes_successful_language(self):
        result = import_studio_result(self.job, self.make_result(failed_language="en-US"))
        self.assertEqual((result["success_count"], result["failed_count"]), (1, 1))
        self.assertEqual(result["outputs"][1]["files"], {})

    def test_tampered_output_never_publishes_any_language(self):
        with self.assertRaisesRegex(PipelineError, "integridade SHA-256"):
            import_studio_result(self.job, self.make_result(checksum_error=True))
        self.assertIsNone(self.job.output_path)
        self.assertFalse((self.root / "studio-results").exists())

    def test_video_duration_is_measured_independently_from_long_audio(self):
        with self.assertRaisesRegex(PipelineError, "faixa de vídeo"):
            import_studio_result(self.job, self.make_result(media=self.truncated))
        self.assertFalse((self.root / "studio-results").exists())

    def test_generic_voice_or_ambiguous_lipsync_are_rejected(self):
        for edit, expected in [
            (lambda report: report["voice_clone"].update(confirmed=False), "timbre original"),
            (lambda report: report["lipsync"].update(ambiguity=[{"start": 1}]), "ambíguos"),
            (lambda report: report["lipsync"].update(confirmed_segments=[]), "trechos faciais"),
        ]:
            with self.subTest(expected=expected), self.assertRaisesRegex(PipelineError, expected):
                import_studio_result(self.job, self.make_result(edit=edit))

    def test_archive_rejects_traversal_windows_paths_and_duplicate_names(self):
        for name in ("../outside.mp4", "outputs/pl-PL/../../outside.mp4", "C:/outside.mp4", "outputs\\pl-PL\\video.mp4", "outputs/pl-PL/CON.json", "RESULT.JSON"):
            with self.subTest(name=name), self.assertRaises(PipelineError):
                import_studio_result(self.job, self.make_result(extra=[(name, b"x")]))

    def test_archive_rejects_symlink_and_compression_bomb_before_extraction(self):
        symlink = zipfile.ZipInfo("outputs/pl-PL/link")
        symlink.create_system = 3
        symlink.external_attr = 0o120777 << 16
        archive_path = self.make_result()
        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr(symlink, "../../../outside")
        with self.assertRaisesRegex(PipelineError, "vínculo"):
            import_studio_result(self.job, archive_path)
        archive_path = self.make_result()
        with zipfile.ZipFile(archive_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("outputs/pl-PL/bomb.json", b"0" * (8 * 1024**2))
        with self.assertRaisesRegex(PipelineError, "descompactação"):
            import_studio_result(self.job, archive_path)
        self.assertFalse((self.root / "studio-results").exists())

    def test_source_identity_is_required_in_every_language_report(self):
        with self.assertRaisesRegex(PipelineError, "outro vídeo"):
            import_studio_result(self.job, self.make_result(edit=lambda report: report.update(input_sha256="1" * 64)))
        self.assertIsNone(self.job.output_path)

    def test_voiceover_needs_complete_scene_analysis_to_preserve_all_frames(self):
        def voiceover(report):
            report["lipsync"].update(confirmed_segments=[], confirmed=False, applied=False, no_visible_speaking_face=True, scene_analysis_complete=True)
            report["preserved_no_speaker_visual_segments"] = [{"start": 0.0, "end": 3.0}]
        result = import_studio_result(self.job, self.make_result(edit=voiceover))
        self.assertEqual(result["success_count"], 2)
        self.assertTrue(all(not output["validation"]["lipsync_confirmed"] for output in result["outputs"]))

    def test_lipsync_requires_applied_and_consistent_scene_windows(self):
        edits = [
            lambda report: report["lipsync"].update(applied=False),
            lambda report: report["lipsync"].pop("applied"),
            lambda report: report["lipsync"]["confirmed_segments"][1].update(end=3.9),
            lambda report: report["lipsync"]["confirmed_segments"][0].update(start=float("nan")),
            lambda report: report.update(preserved_no_speaker_visual_segments=[{"start": -20, "end": 300}]),
            lambda report: report.update(preserved_no_speaker_visual_segments=[{"start": 0.5, "end": 2}]),
            lambda report: report.update(preserved_no_speaker_visual_segments=[]),
            lambda report: report["lipsync"].update(no_visible_speaking_face=True),
        ]
        for index, edit in enumerate(edits):
            with self.subTest(case=index), self.assertRaises(PipelineError):
                import_studio_result(self.job, self.make_result(edit=edit))
        self.assertIsNone(self.job.output_path)
        self.assertFalse((self.root / "studio-results").exists())

    def test_result_and_report_bind_operation_and_all_input_checksums(self):
        for edit in (lambda report: report.update(operation="lipsync_video"),
                     lambda report: report.update(input_checksums={}),
                     lambda report: report["input_checksums"].update(video="0" * 64)):
            with self.subTest(edit=edit), self.assertRaises(PipelineError):
                import_studio_result(self.job, self.make_result(edit=edit))
        for key, value in (("operation", "lipsync_video"), ("input_checksums", {})):
            original = self.make_result()
            with zipfile.ZipFile(original) as archive:
                blobs = {name: archive.read(name) for name in archive.namelist()}
            header = json.loads(blobs["result.json"])
            header[key] = value
            blobs["result.json"] = json.dumps(header).encode()
            changed = self.root / "changed-result.zip"
            with zipfile.ZipFile(changed, "w") as archive:
                for name, data in blobs.items():
                    archive.writestr(name, data)
            with self.subTest(header=key), self.assertRaises(PipelineError):
                import_studio_result(self.job, changed)

    def lipsync_job(self, audio):
        self.job.options.update(operation="lipsync_video", target_languages=["und"], audio_path=str(audio))
        prepare_studio_package(self.job)
        self.manifest = self.job.metadata["studio"]["manifest"]

    def test_audio_lipsync_cannot_claim_success_without_rendered_faces(self):
        audio = self.root / "uploaded.wav"
        shutil.copy2(self.wav, audio)
        self.lipsync_job(audio)
        def absent_faces(report):
            report["lipsync"].update(confirmed_segments=[], confirmed=False, applied=False,
                                     no_visible_speaking_face=True, scene_analysis_complete=True)
            report["preserved_no_speaker_visual_segments"] = [{"start": 0.0, "end": 3.0}]
        with self.assertRaisesRegex(PipelineError, "trechos faciais"):
            import_studio_result(self.job, self.make_result(edit=absent_faces))
        with self.assertRaisesRegex(PipelineError, "mídias de entrada"):
            import_studio_result(self.job, self.make_result(edit=lambda report: report["input_checksums"].update(audio="0" * 64)))

    def test_short_uploaded_audio_can_be_padded_but_its_last_voice_cannot_disappear(self):
        short = self.root / "uploaded-short.wav"
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.wav), "-t", "1", str(short)], check=True, capture_output=True)
        self.lipsync_job(short)
        padded = self.root / "padded.mp4"
        missing_tail = self.root / "missing-tail.mp4"
        for target, audio_filter in ((padded, "apad"), (missing_tail, "atrim=duration=0.2,apad")):
            subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.source), "-i", str(short),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-af", audio_filter, "-t", "3", str(target)], check=True, capture_output=True)
        accepted = import_studio_result(self.job, self.make_result(media=padded))
        self.assertEqual(accepted["success_count"], 1)
        existing = self.job.output_path
        with self.assertRaisesRegex(PipelineError, "janela correspondente"):
            import_studio_result(self.job, self.make_result(media=missing_tail))
        self.assertEqual(self.job.output_path, existing)
        self.assertTrue(existing.is_file())

    def test_audio_lipsync_refuses_to_trim_uploaded_audio_larger_than_video(self):
        larger = self.root / "uploaded-too-long.wav"
        subprocess.run([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.wav), "-af", "apad", "-t", "3.1", str(larger)], check=True, capture_output=True)
        self.lipsync_job(larger)
        with self.assertRaisesRegex(PipelineError, "não pode ser cortado"):
            import_studio_result(self.job, self.make_result())

    def test_dubbing_without_lipsync_requires_explicit_opt_out_and_honest_report(self):
        def without_lipsync(report):
            report["lipsync"] = {"required": False, "confirmed": False, "reason": "Desativado pelo usuário"}
        with self.assertRaisesRegex(PipelineError, "lip-sync obrigatório"):
            import_studio_result(self.job, self.make_result(edit=without_lipsync))
        self.job.options["require_lipsync"] = False
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
        result = import_studio_result(self.job, self.make_result(edit=without_lipsync))
        self.assertEqual(result["success_count"], 2)
        for output in result["outputs"]:
            self.assertFalse(output["validation"]["lipsync_requested"])
            self.assertFalse(output["validation"]["lipsync_confirmed"])
            self.assertTrue(output["validation"]["authorized_clone"])
        with self.assertRaisesRegex(PipelineError, "etapa não solicitada"):
            import_studio_result(self.job, self.make_result())

    def test_lipsync_defaults_to_true_and_rejects_implicit_opt_out(self):
        del self.job.options["require_lipsync"]
        prepare_studio_package(self.job)
        self.assertTrue(self.job.metadata["studio"]["manifest"]["options"]["require_lipsync"])
        for invalid in (None, 0, "false", ""):
            self.job.options["require_lipsync"] = invalid
            with self.subTest(value=invalid), self.assertRaisesRegex(PipelineError, "verdadeira ou falsa"):
                prepare_studio_package(self.job)
        audio = self.root / "audio.wav"
        shutil.copy2(self.wav, audio)
        self.job.options.update(operation="lipsync_video", audio_path=str(audio), require_lipsync=False)
        prepare_studio_package(self.job)
        self.assertTrue(self.job.metadata["studio"]["manifest"]["options"]["require_lipsync"])

    def test_review_only_allows_translated_text_to_change(self):
        review = {
            "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "job_id": self.job.id,
            "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
            "translations": {language: [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"}] for language in self.manifest["target_languages"]},
        }
        path = self.root / "review.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("review.json", json.dumps(review))
        imported = import_review(self.job, path)
        approved = copy.deepcopy(imported)
        approved["translations"]["pl-PL"][0]["translated_text"] = "Dzień dobry"
        exported = export_approved_review(self.job, approved)
        with zipfile.ZipFile(exported) as archive:
            self.assertEqual(json.loads(archive.read("review-approved.json"))["translations"]["pl-PL"][0]["translated_text"], "Dzień dobry")
            resume_manifest = json.loads(archive.read("manifest.json"))
            contract = resume_manifest.pop("review_snapshot")
            self.assertEqual(resume_manifest, self.manifest)
            self.assertEqual(contract["sha256"], hashlib.sha256(archive.read("review-source.json")).hexdigest())
            self.assertEqual(contract["approved_sha256"], hashlib.sha256(archive.read("review-approved.json")).hexdigest())
            source = json.loads(archive.read("review-source.json"))
            self.assertEqual(source["segments"], [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá"}])
            self.assertEqual(archive.read("worker/dubstudio_runner_v2.py"), (Path(__file__).parent.parent / "colab/dubstudio_runner_v2.py").read_bytes())
            self.assertEqual(hashlib.sha256(archive.read(self.manifest["input_file"])).hexdigest(), self.manifest["input_sha256"])
            self.assertIn("worker/dubstudio_colab_worker.py", archive.namelist())
            resumed = self.root / "resumed-package"
            archive.extractall(resumed)
        from colab import dubstudio_runner_v2 as runner
        verified = runner.load_approved_snapshot(resumed, runner.validate_manifest(resumed), 3.0)
        self.assertEqual(verified["translations"]["pl-PL"][0]["translated_text"], "Dzień dobry")
        self.assertEqual(verified["source"]["segments"], source["segments"])
        approved["translations"]["pl-PL"][0]["end"] = 1.5
        with self.assertRaisesRegex(PipelineError, "mesma transcrição|somente a tradução"):
            export_approved_review(self.job, approved)

    def test_review_import_rejects_mismatched_language_timelines(self):
        review = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                  "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
                  "translations": {lang: [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"}]
                                   for lang in self.manifest["target_languages"]}}
        review["translations"]["en-US"][0]["start"] = 0.02
        archive = self.root / "different-timelines.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("review.json", json.dumps(review))
        with self.assertRaisesRegex(PipelineError, "mesma transcrição"):
            import_review(self.job, archive)

    def test_review_import_rejects_other_pipeline(self):
        review = {"schema_version": SCHEMA_VERSION, "pipeline_version": "2.9.0",
                  "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
                  "translations": {}}
        archive = self.root / "old-pipeline.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("review.json", json.dumps(review))
        with self.assertRaisesRegex(PipelineError, "pipeline incompatível"):
            import_review(self.job, archive)

    def test_review_import_rejects_invalid_source_times_identity_and_languages(self):
        base = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
                "translations": {lang: [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"}]
                                 for lang in self.manifest["target_languages"]}}
        modifications = [("start", float("nan")), ("end", float("inf")), ("start", True),
                         ("end", 4.01), ("id", False), ("speaker", "../escape")]
        for key, value in modifications:
            with self.subTest(key=key, value=value):
                review = copy.deepcopy(base)
                for items in review["translations"].values():
                    items[0][key] = value
                archive = self.root / "invalid-review.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("review.json", json.dumps(review))
                with self.assertRaises(PipelineError):
                    import_review(self.job, archive)

        for mode in ("missing", "extra", "different_media"):
            with self.subTest(mode=mode):
                review = copy.deepcopy(base)
                if mode == "missing":
                    review["translations"].pop("en-US")
                elif mode == "extra":
                    review["translations"]["es-ES"] = copy.deepcopy(review["translations"]["en-US"])
                else:
                    review["input_sha256"] = "0" * 64
                archive = self.root / "invalid-review.zip"
                with zipfile.ZipFile(archive, "w") as output:
                    output.writestr("review.json", json.dumps(review))
                with self.assertRaises(PipelineError):
                    import_review(self.job, archive)

    def approve_fixture(self, text="Dzień dobry"):
        review = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                  "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "source_language": "pt",
                  "translations": {lang: [{"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"}]
                                   for lang in self.manifest["target_languages"]}}
        path = self.root / "review-before-result.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("review.json", json.dumps(review))
        import_review(self.job, path)
        for items in review["translations"].values():
            items[0]["translated_text"] = text
        export_approved_review(self.job, review)
        return review

    def test_import_rejects_previous_translation_after_latest_approval(self):
        old_result = self.make_result()
        import_studio_result(self.job, old_result)
        previous_output = self.job.output_path
        previous_metadata = copy.deepcopy(self.job.metadata["studio_result"])
        self.approve_fixture()
        with self.assertRaisesRegex(PipelineError, "revisão aprovada atual"):
            import_studio_result(self.job, old_result)
        self.assertEqual(self.job.output_path, previous_output)
        self.assertEqual(self.job.metadata["studio_result"], previous_metadata)

    def test_import_accepts_current_approval_with_speed_metadata(self):
        approved = self.approve_fixture()
        def current_segments(payload):
            payload["segments"] = copy.deepcopy(approved["translations"][payload["target_language"]])
            payload["segments"][0]["speed"] = 1.234
        imported = import_studio_result(self.job, self.make_result(segments_edit=current_segments))
        self.assertEqual(imported["success_count"], 2)

    def test_import_rejects_changed_approved_timeline_speaker_ids_and_text(self):
        approved = self.approve_fixture()
        for key, value in (("start", 0.001), ("end", 1.001), ("speaker", "SPEAKER_99"),
                           ("id", "1"), ("id", True), ("text", "Outro texto.")):
            with self.subTest(key=key, value=value):
                def altered(payload):
                    payload["segments"] = copy.deepcopy(approved["translations"][payload["target_language"]])
                    payload["segments"][0][key] = value
                with self.assertRaisesRegex(PipelineError, "revisão aprovada atual"):
                    import_studio_result(self.job, self.make_result(segments_edit=altered))

    def test_import_rejects_tampered_local_approved_source_snapshot(self):
        self.approve_fixture()
        snapshot = self.root / "review-source.json"
        snapshot.write_text(snapshot.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(PipelineError, "snapshot original aprovado foi alterado"):
            import_studio_result(self.job, self.make_result())

    def test_import_does_not_ignore_missing_current_approval(self):
        self.approve_fixture()
        (self.root / "review-approved.json").unlink()
        with self.assertRaisesRegex(PipelineError, "revisão aprovada atual não foi encontrada"):
            import_studio_result(self.job, self.make_result())

    def test_tts_requires_reference_and_accepts_both_audio_formats(self):
        reference = self.root / "reference.wav"
        shutil.copy2(self.wav, reference)
        self.job.input_path = reference
        self.job.options.update(operation="text_to_voice", text="Olá mundo.", reference_path=str(reference))
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
        self.assertFalse(self.manifest["options"]["require_lipsync"])
        result = import_studio_result(self.job, self.make_result(edit=lambda report: report.update(lipsync={"required": False, "confirmed": False})))
        self.assertEqual(result["success_count"], 2)
        self.assertEqual(self.job.output_path.suffix, ".wav")

    def test_tts_timeline_uses_final_audio_duration_not_reference_duration(self):
        reference = self.root / "reference.wav"
        shutil.copy2(self.wav, reference)
        self.job.input_path = reference
        self.job.options.update(operation="text_to_voice", text="Olá. Boa noite.", reference_path=str(reference))
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
        audio = {}
        for role in ("wav", "mp3"):
            path = self.root / f"long-voice.{role}"
            subprocess.run([shutil.which("ffmpeg"), "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "6", str(path)], check=True, capture_output=True)
            audio[role] = path
        def report(payload):
            payload.update(lipsync={"required": False, "confirmed": False})
            payload["summary"]["duration"] = 6.0
        def segments(payload):
            payload["segments"] = [
                {"id": 1, "start": 0.0, "end": 2.0, "speaker": "SPEAKER_00", "text": "Olá.", "translated_text": "Olá."},
                {"id": 2, "start": 2.18, "end": 6.0, "speaker": "SPEAKER_00", "text": "Boa noite.", "translated_text": "Boa noite."},
            ]
        result = import_studio_result(self.job, self.make_result(edit=report, segments_edit=segments, audio_media=audio))
        self.assertEqual(result["success_count"], 2)
        self.assertAlmostEqual(result["outputs"][0]["validation"]["media"]["wav"]["audio_duration"], 6.0, places=2)

    def test_tts_rejects_zero_or_out_of_audio_segment_times(self):
        reference = self.root / "reference.wav"
        shutil.copy2(self.wav, reference)
        self.job.input_path = reference
        self.job.options.update(operation="text_to_voice", text="Olá mundo.", reference_path=str(reference))
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
        for end in (0.0, 3.2, True):
            with self.subTest(end=end):
                def segments(payload):
                    payload["segments"][0]["end"] = end
                with self.assertRaisesRegex(PipelineError, "tempos da revisão|tempos da voz gerada|valor inválido"):
                    import_studio_result(self.job, self.make_result(edit=lambda report: report.update(lipsync={"required": False, "confirmed": False}), segments_edit=segments))

    def test_tts_rejects_a_video_renamed_as_audio(self):
        reference = self.root / "reference.wav"
        shutil.copy2(self.wav, reference)
        self.job.input_path = reference
        self.job.options.update(operation="text_to_voice", text="Olá mundo.", reference_path=str(reference))
        prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / "studio-manifest.json").read_text(encoding="utf-8"))
        self.wav = self.source
        with self.assertRaisesRegex(PipelineError, "formato real"):
            import_studio_result(self.job, self.make_result(edit=lambda report: report.update(lipsync={"required": False, "confirmed": False})))
        self.assertIsNone(self.job.output_path)

    def test_unsupported_original_voice_is_rejected_before_packaging(self):
        self.job.options["target_languages"] = ["uk-UA"]
        with self.assertRaisesRegex(PipelineError, "timbre original"):
            prepare_studio_package(self.job)


class LanguageCapabilityTests(unittest.TestCase):
    def test_model_support_is_distinct_from_end_to_end_validation_and_accent(self):
        languages = {item["code"]: item for item in languages_payload()}
        self.assertTrue(languages["pl-PL"]["supported"])
        self.assertFalse(languages["uk-UA"]["supported"])
        self.assertTrue(supports_original_voice("nb-NO"))
        self.assertEqual(languages["pl-PL"]["validation_status"], "pending_gpu_validation")
        self.assertFalse(languages["pt-BR"]["regional_accent_guaranteed"])


if __name__ == "__main__":
    unittest.main()
