"""Full local orchestration with synthetic media and mocked AI stages only.

FFmpeg retiming, mix, export, PCM durations, subtitles and result ZIP are real.
Synthetic actor labels/boxes and voice samples are not face/timbre/lip quality
evidence; this suite never launches a GPU or installs/downloads models.
"""
import array
import copy
import json
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest.mock import patch

import dubstudio_audio_master as audio_master
import dubstudio_bootstrap as bootstrap
import dubstudio_colab_worker as worker
import dubstudio_runner_v2 as runner


def pcm(path, frames, rate=48000):
    path.parent.mkdir(parents=True, exist_ok=True)
    values = array.array("h", [6000, 3000, -3000, -6000])
    raw = (values.tobytes() * ((frames * 2 + 3) // 4))[:frames * 4]
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(raw)


def source():
    return [{"id": 1, "start": 0.0, "end": 3.0, "speaker": "A", "text": "Primeira pessoa."},
            {"id": 2, "start": 3.0, "end": 5.0, "speaker": "B", "text": "Segunda pessoa."},
            {"id": 3, "start": 5.0, "end": 6.0, "speaker": "A", "text": "A primeira volta."}]


def faces():
    return {"status": "complete", "ambiguity": [], "scene_analysis_complete": True,
            "confirmed_face_segments": [
                {"speaker": speaker, "start_frame": first, "end_frame": last, "fps": 25.0,
                 "boxes": [[10, 10, 80, 90]] * (last - first), "track_id": index,
                 "face_confirmation": "detected", "speaker_confirmation": "active_speaker",
                 "coordinate_source": "synthetic test geometry, not real detection"}
                for index, (speaker, first, last) in enumerate((("A", 0, 75), ("B", 75, 125), ("A", 125, 150)))]}


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe required")
class AudioMasterOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".test-audio-master-process-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.package = self.root / "package"
        self.video = self.package / "input/video.mp4"
        self.video.parent.mkdir(parents=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", "testsrc2=size=100x100:rate=30:duration=6", "-an", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(self.video)], check=True, capture_output=True)
        self.reference = self.package / "input/reference.wav"
        pcm(self.reference, 48000)
        self.work, self.storage = self.root / "work", self.root / "storage"
        self.calls = []
        self.failure_stage = None

    def manifest(self, lipsync=True, languages=None, operation="dub_video"):
        video_hash, reference_hash = runner.sha256(self.video), runner.sha256(self.reference)
        inputs = {"video": {"path": "input/video.mp4", "sha256": video_hash}}
        if operation == "text_to_voice":
            inputs = {"reference": {"path": "input/reference.wav", "sha256": reference_hash}}
        manifest = {"schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": "123abc456def",
                    "operation": operation, "input_file": "input/reference.wav" if operation == "text_to_voice" else "input/video.mp4",
                    "input_sha256": reference_hash if operation == "text_to_voice" else video_hash,
                    "inputs": inputs, "target_languages": languages or ["pl-PL", "es-ES"],
                    "authorization": {"voice_clone_confirmed": True},
                    "options": {"timing_mode": "audio_master", "require_lipsync": lipsync,
                                "preserve_ambience": True, "review_mode": "auto", "source_language": "pt",
                                **({"text": "Texto natural autorizado."} if operation == "text_to_voice" else {})}}
        runner.write_json(self.package / "manifest.json", manifest)
        return manifest

    def ai_stage(self, name, config, outputs):
        directory = Path(config["directory"])
        directory.mkdir(parents=True, exist_ok=True)
        if name in {"separation", "asr", "diarization", "references", "analysis"}:
            if name == "separation":
                pcm(directory / "vocals.wav", 48000 * 6)
                pcm(directory / "background.wav", 48000 * 6)
            elif name in {"asr", "diarization"}:
                runner.write_json(Path(config["output"]), {"segments": source(), "language": "pt"})
            elif name == "references":
                references = {}
                for speaker in ("A", "B"):
                    target = directory / f"reference-{speaker}.wav"
                    pcm(target, 48000)
                    references[speaker] = str(target)
                runner.write_json(Path(config["output"]), references)
            else:
                runner.write_json(Path(config["output"]), faces())
                (directory / "talknet-normalized.avi").write_bytes(b"AI analysis protocol fixture only")
        elif name == "synthesis":
            segments = runner.read_json(Path(config["segments"]))["segments"]
            lengths = ([336000, 48000, 96000] if config["language"] == "pl-PL" else [192000, 96000, 96000])[:len(segments)]
            dialogue = directory / "dialogue.wav"
            pcm(dialogue, sum(lengths))
            natural_segments = runner.tts_segments_for_frames(segments, lengths, pause_frames=0)
            runner.write_json(directory / "voice-segments.json", {"segments": natural_segments})
            proof = {"voice_engine": "synthetic PCM fixture; no AI execution", "segments": segments,
                     "voice_clone": {"required": True, "confirmed": True, "authorized": True},
                     "references": {key: runner.sha256(Path(value)) for key, value in config["references"].items()},
                     "timing_adjustments": [{"segment": item["id"], "tempo_factor": 1.0} for item in segments]}
            runner.write_json(directory / "voice-proof.json", proof)
            if config.get("timing_mode") == "audio_master":
                timeline = audio_master.build_timeline(segments, lengths)
                timeline.update(target_language=config["language"], input_sha256=config["identity"]["input_sha256"],
                                source_video_sha256=config["source_video_sha256"],
                                audio_master={"path": "dialogue.wav", "sha256": runner.sha256(dialogue),
                                              "sample_rate": 48000, "sample_count": sum(lengths)})
                runner.write_json(directory / "audio-master.json", timeline)
        elif name == "translation":
            segments = [{**item, "translated_text": f"{config['language']} approved test speech {item['id']}"} for item in source()]
            runner.write_json(Path(config["output"]), {"source_language": "pt", "target_language": config["language"],
                                                        "segments": segments, "translation_engine": "synthetic translation fixture"})
        elif name == "lipsync":
            # Only the AI render is mocked; the retimed base is a real video.
            shutil.copy2(Path(config["video"]), Path(config["output"]))
            proof = runner.read_json(Path(config["analysis"]))
            runner.write_json(Path(config["proof"]), proof)
        else:
            runner.stage_main(name, config)
        for path in outputs:
            if not Path(path).is_file():
                raise AssertionError(f"stage {name} expected output in wrong language/clock: {path}")

    def process(self, lipsync=True, languages=None, operation="dub_video"):
        self.manifest(lipsync, languages, operation)
        owner = self

        class CPUCache:
            def __init__(self, *args):
                self.used, self.seconds, self.gpu_activity = {}, {}, {}

            def stage(self, key, label, config, outputs):
                owner.calls.append((key, copy.deepcopy(config)))
                if key == owner.failure_stage:
                    raise runner.PipelineError("synthetic safe failure with checkpoints preserved")
                owner.ai_stage(config["stage"], config, outputs)
                self.used[key], self.seconds[key] = False, 0.0

            def persist(self):
                pass

        with patch.object(bootstrap, "gpu_precheck", return_value="synthetic GPU precheck; no GPU launched"), \
             patch.object(runner, "StageCache", CPUCache), patch.object(worker, "write_progress"):
            return runner.process(self.package, self.work, self.storage, self.root / "result.zip")

    def test_parallel_language_settings_are_isolated_and_lipsync_follows_retiming(self):
        result = self.process()
        self.assertEqual([item["status"] for item in result["outputs"]], ["complete", "complete"])
        self.assertIn("analysis", [key for key, _ in self.calls])
        for language, duration in (("pl-PL", 10), ("es-ES", 8)):
            keys = [key for key, _ in self.calls]
            self.assertLess(keys.index(f"retime-{language}"), keys.index(f"mix-{language}"))
            self.assertLess(keys.index(f"mix-{language}"), keys.index(f"lipsync-{language}"))
            self.assertLess(keys.index(f"lipsync-{language}"), keys.index(f"export-{language}"))
            configs = {key: config for key, config in self.calls}
            retime, mix = configs[f"retime-{language}"], configs[f"mix-{language}"]
            self.assertEqual(Path(retime["timeline"]).parent.name, language)
            self.assertEqual(Path(retime["background"]).parent.name, "shared")
            self.assertEqual(mix["duration"], duration)
            target = self.work / "123abc456def/outputs" / language
            with wave.open(str(target / "dialogue.wav"), "rb") as voice, wave.open(str(target / "mix.wav"), "rb") as mixed:
                self.assertEqual(voice.getnframes(), duration * 48000)
                self.assertEqual(mixed.getnframes(), voice.getnframes())
            self.assertEqual(worker.video_frame_count(target / "video.mp4"), duration * 30)
            self.assertIn(f"00:00:{duration - 2:02d},000 --> 00:00:{duration:02d},000",
                          (target / "translation.srt").read_text(encoding="utf-8"))
        with zipfile.ZipFile(self.root / "result.zip") as archive:
            result_metadata = json.loads(archive.read("result.json"))
            for record in result_metadata["outputs"]:
                self.assertIn("timeline", record["files"])
                self.assertIn("master_audio", record["files"])
                for path, checksum in record["checksums"].items():
                    self.assertEqual(runner.hashlib.sha256(archive.read(path)).hexdigest(), checksum)

    def test_no_lipsync_still_analyzes_person_and_retimes_video_without_facial_render(self):
        result = self.process(lipsync=False, languages=["pl-PL"])
        self.assertEqual(result["outputs"][0]["status"], "complete")
        keys = [key for key, _ in self.calls]
        self.assertIn("analysis", keys)
        self.assertIn("retime-pl-PL", keys)
        self.assertNotIn("lipsync-pl-PL", keys)
        report = runner.read_json(self.work / "123abc456def/outputs/pl-PL/report.json")
        self.assertFalse(report["lipsync"]["applied"])
        self.assertEqual(report["timing_mode"], "audio_master")
        self.assertEqual(source(), runner.read_json(self.work / "123abc456def/shared/diarization.json")["segments"])

    def test_one_failed_language_preserves_checkpoints_and_other_language_continues(self):
        self.failure_stage = "retime-pl-PL"
        result = self.process()
        self.assertEqual([item["status"] for item in result["outputs"]], ["failed", "complete"])
        self.assertTrue((self.work / "123abc456def/outputs/pl-PL/dialogue.wav").is_file())
        self.assertNotIn("video", result["outputs"][0]["files"])
        self.assertIn("video", result["outputs"][1]["files"])

    def test_text_to_voice_keeps_natural_samples_and_never_calls_video_stages(self):
        result = self.process(languages=["es-ES"], operation="text_to_voice")
        self.assertEqual(result["outputs"][0]["status"], "complete")
        keys = [key for key, _ in self.calls]
        self.assertFalse(any(key == "analysis" or key.startswith(("retime-", "mix-", "lipsync-", "export-")) for key in keys))
        with wave.open(str(self.work / "123abc456def/outputs/es-ES/dialogue.wav"), "rb") as stream:
            self.assertEqual(stream.getnframes(), 192000)


if __name__ == "__main__":
    unittest.main()
