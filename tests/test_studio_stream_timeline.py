"""Real CPU media regressions for input/proxy/import duration consistency."""

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import studio_bridge as bridge
from pipeline import Job, PipelineError, probe_video


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class StudioStreamTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix="st-")
        cls.media = Path(cls.shared.name)
        cls.mp4 = cls.media / "padded.mp4"
        cls.high = cls.media / "high.mp4"
        cls.mkv = cls.media / "padded.mkv"
        cls.webm = cls.media / "padded.webm"
        cls.tiny = cls.media / "tiny.mkv"
        cls.audio = cls.media / "audio.wav"
        cls.long_audio = cls.media / "long.wav"
        for target, dimensions in ((cls.mp4, "160x90"), (cls.high, "2048x1152")):
            cls.make_media(target, f"color=c=blue:s={dimensions}:r=5:d=1.2", "sine=frequency=700:sample_rate=48000:duration=2.4", ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac"])
        subprocess.run(["ffmpeg", "-y", "-i", str(cls.mp4), "-c", "copy", str(cls.mkv)], check=True, capture_output=True)
        cls.make_media(cls.webm, "color=c=blue:s=160x90:r=5:d=1.2", "sine=frequency=700:sample_rate=48000:duration=2.4", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus"])
        cls.make_media(cls.tiny, "color=c=blue:s=160x90:r=120:d=0.008333333", "sine=frequency=700:sample_rate=48000:duration=1.2", ["-c:v", "libx264", "-c:a", "aac"])
        for target, duration in ((cls.audio, 1.2), (cls.long_audio, 1.5)):
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", f"sine=frequency=900:sample_rate=48000:duration={duration}", str(target)], check=True, capture_output=True)

    @classmethod
    def make_media(cls, target, video, audio, codecs):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", audio,
            *codecs, "-pix_fmt", "yuv420p", str(target)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="sj-")
        self.root = Path(self.folder.name)

    def tearDown(self):
        self.folder.cleanup()

    def make_job(self, source, audio=None, operation="lipsync_video"):
        target = self.root / f"original{source.suffix}"
        shutil.copy2(source, target)
        job = Job(id="abc123abc123", directory=self.root, input_path=target,
            options={"operation": operation, "timing_mode": "source_video", "target_languages": ["und" if operation == "lipsync_video" else "es-ES"],
                     "voice_clone_consent": True, "require_lipsync": operation == "lipsync_video"}, metadata=probe_video(target))
        if operation == "lipsync_video":
            submitted = self.root / "submitted.wav"
            shutil.copy2(audio or self.audio, submitted)
            job.metadata["studio_inputs"] = {"audio": {"path": submitted.name}}
        return job

    def manifest(self, job):
        bridge.prepare_studio_package(job)
        return json.loads((job.directory / "studio-manifest.json").read_text(encoding="utf-8"))

    def test_mkv_input_manifest_uses_images_not_audio_container(self):
        job = self.make_job(self.mkv)
        manifest = self.manifest(job)
        self.assertAlmostEqual(manifest["input_metadata"]["duration"], 1.2, places=5)
        self.assertGreater(manifest["input_metadata"]["container_duration"], 2.4)
        self.assertEqual(manifest["input_metadata"]["duration_basis"], "v:0")

    def test_webm_input_manifest_supports_absent_video_stream_duration(self):
        job = self.make_job(self.webm)
        manifest = self.manifest(job)
        self.assertAlmostEqual(manifest["input_metadata"]["duration"], 1.2, places=5)
        self.assertGreater(manifest["input_metadata"]["container_duration"], 2.4)

    def test_uploaded_audio_longer_than_real_mkv_images_blocks_before_export(self):
        job = self.make_job(self.mkv, self.long_audio)
        with self.assertRaisesRegex(PipelineError, "áudio enviado excede"):
            bridge.prepare_studio_package(job)
        self.assertFalse((self.root / bridge.PACKAGE_NAME).exists())

    def test_tiny_images_do_not_accept_long_audio_even_with_long_container(self):
        job = self.make_job(self.tiny)
        with self.assertRaisesRegex(PipelineError, "áudio enviado excede"):
            bridge.prepare_studio_package(job)
        self.assertFalse((self.root / bridge.PACKAGE_NAME).exists())

    def test_proxy_and_supplied_audio_roundtrip_preserves_true_video_timeline(self):
        job = self.make_job(self.high)
        manifest = self.manifest(job)
        self.assertTrue(manifest["proxy_created"])
        metadata = manifest["input_metadata"]
        self.assertAlmostEqual(metadata["duration"], 1.2)
        self.assertAlmostEqual(metadata["packaged_video_duration"], 1.2)
        self.assertGreater(metadata["container_duration"], 2.3)
        final = self.root / "final.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", str(self.root / "proxy-1080p.mp4"), "-i", str(self.root / "submitted.wav"),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-t", "1.2", str(final)], check=True, capture_output=True)
        mp3 = self.root / "voice.mp3"
        subprocess.run(["ffmpeg", "-y", "-i", str(self.root / "submitted.wav"), "-c:a", "libmp3lame", str(mp3)], check=True, capture_output=True)
        binding = {"schema_version": bridge.SCHEMA_VERSION, "pipeline_version": bridge.PIPELINE_VERSION,
            "job_id": job.id, "input_sha256": manifest["input_sha256"], "operation": "lipsync_video",
            "input_checksums": {role: value["sha256"] for role, value in manifest["inputs"].items()}}
        report = {**binding, "target_language": "und", "summary": {"duration": 1.2},
            "voice_clone": {"required": False, "confirmed": False, "authorized": True},
            "lipsync": {"required": True, "confirmed": True, "applied": True, "ambiguity": [],
                "confirmed_segments": [{"start": 0.0, "end": 1.2, "coordinate_source": "external_audio",
                    "face_confirmation": "detected", "speaker_confirmation": "external_audio"}]},
            "preserved_no_speaker_visual_segments": []}
        files = {"video": "outputs/und/video.mp4", "wav": "outputs/und/mix.wav", "mp3": "outputs/und/voice.mp3", "report": "outputs/und/report.json"}
        blobs = {files["video"]: final.read_bytes(), files["wav"]: (self.root / "submitted.wav").read_bytes(),
            files["mp3"]: mp3.read_bytes(), files["report"]: json.dumps(report).encode()}
        result = {**binding, "outputs": [{"language": "und", "status": "complete", "files": files,
            "checksums": {path: hashlib.sha256(blob).hexdigest() for path, blob in blobs.items()}}]}
        result_zip = self.root / "result.zip"
        with zipfile.ZipFile(result_zip, "w") as archive:
            archive.writestr("result.json", json.dumps(result))
            for path, blob in blobs.items():
                archive.writestr(path, blob)
        imported = bridge.import_studio_result(job, result_zip)
        self.assertEqual(imported["success_count"], 1)
        actual = imported["outputs"][0]["validation"]["media"]["video"]
        self.assertAlmostEqual(actual["video_duration"], 1.2)
        self.assertAlmostEqual(actual["audio_duration"], 1.2)
        poisoned = copy.deepcopy(manifest)
        poisoned["input_metadata"]["duration"] = 999
        self.assertAlmostEqual(bridge._validate_media(final, "video", job, poisoned, report)["video_duration"], 1.2)

    def test_tts_reference_duration_is_physical_audio_not_long_video(self):
        reference = self.root / "reference.mp4"
        self.make_media(reference, "color=c=blue:s=160x90:r=5:d=2.4", "sine=frequency=900:sample_rate=48000:duration=1.2", ["-c:v", "libx264", "-c:a", "aac"])
        job = Job(id="abc123abc123", directory=self.root, input_path=reference,
            options={"operation": "text_to_voice", "timing_mode": "source_video", "target_languages": ["es-ES"], "voice_clone_consent": True,
                     "text": "Prueba de voz."}, metadata={})
        manifest = self.manifest(job)
        self.assertAlmostEqual(manifest["input_metadata"]["duration"], 1.2)
        self.assertEqual(manifest["input_metadata"]["duration_basis"], "a:0")
        self.assertGreater(manifest["input_metadata"]["container_duration"], 2.3)


if __name__ == "__main__":
    unittest.main()
