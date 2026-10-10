"""Real WAV/FFmpeg timing tests; synthesis models and timbre are not validated here."""
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

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner
import dubstudio_translate as translator


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VoiceOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "whole-utterance.wav"
        self.sine(self.raw, 1.0)

    def tearDown(self):
        self.temporary.cleanup()

    def sine(self, path, duration):
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        f"sine=frequency=880:sample_rate=48000:duration={duration}", "-ac", "2", "-c:a", "pcm_s16le", str(path)],
                       check=True, capture_output=True)

    def samples(self, path):
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes(), audio.getframerate(), audio.readframes(audio.getnframes())

    def test_rounding_overrun_retries_entire_utterance_without_trimming(self):
        original = runner.sha256(self.raw)
        commands = []
        run = runner.run_command
        def execute(command, *args, **kwargs):
            commands.append(command)
            run(command, *args, **kwargs)
        output = self.root / "fitted.wav"
        available = 47040  # 0.98s, a small overrun rather than a shortened input.
        with patch.object(runner, "run_command", side_effect=execute):
            factor = runner.fit_voice_to_frames(self.raw, output, 1.0, available)
        frames, rate, pcm = self.samples(output)
        self.assertGreaterEqual(len(commands), 2)
        self.assertGreater(factor, 1.0)
        self.assertLessEqual(factor, 1.05)
        self.assertLessEqual(frames, available)
        self.assertEqual(rate, 48000)
        self.assertNotEqual(pcm[-4096:], b"\x00" * min(4096, len(pcm)))
        self.assertEqual(runner.sha256(self.raw), original)
        for command in commands:
            self.assertEqual(command[command.index("-i") + 1], str(self.raw))
            self.assertNotIn("-t", command)
            self.assertFalse(any("trim" in argument for argument in command))

    def test_persistent_overrun_fails_instead_of_cropping_samples(self):
        output = self.root / "not-fitted.wav"
        def never_fits(command, *args, **kwargs):
            shutil.copy2(self.raw, Path(command[-1]))
        with patch.object(runner, "run_command", side_effect=never_fits) as encode:
            with self.assertRaisesRegex(runner.PipelineError, "não cortar seu final"):
                runner.fit_voice_to_frames(self.raw, output, 1.0, 47520)
        self.assertEqual(encode.call_count, 3)
        self.assertEqual(self.samples(output)[0], self.samples(self.raw)[0])

    def test_unexpected_large_rounding_correction_fails_safely(self):
        output = self.root / "large-overrun.wav"
        with patch.object(runner, "run_command", side_effect=lambda command, *args, **kwargs: shutil.copy2(self.raw, command[-1])):
            with self.assertRaisesRegex(runner.PipelineError, "não cortar seu final"):
                runner.fit_voice_to_frames(self.raw, output, 1.0, 36000)
        self.assertEqual(self.samples(output)[0], 48000)

    def test_empty_timeline_window_never_encodes(self):
        with patch.object(runner, "run_command") as encode:
            with self.assertRaisesRegex(runner.PipelineError, "Não há espaço"):
                runner.fit_voice_to_frames(self.raw, self.root / "out.wav", 1.0, 0)
        encode.assert_not_called()

    def test_tts_timestamps_count_actual_samples_and_pauses_without_mutating_input(self):
        second = self.root / "second.wav"
        self.sine(second, 0.7)
        segments = [
            {"id": 1, "start": 0.0, "end": 0.0, "speaker": "SPEAKER_00", "text": "Olá.", "translated_text": "Olá."},
            {"id": 2, "start": 0.0, "end": 0.0, "speaker": "SPEAKER_00", "text": "Boa noite.", "translated_text": "Boa noite."},
        ]
        original = copy.deepcopy(segments)
        frame_counts = [self.samples(path)[0] for path in (self.raw, second)]
        rendered = runner.tts_segments_for_frames(segments, frame_counts)
        self.assertEqual(segments, original)
        self.assertEqual([(item["start"], item["end"]) for item in rendered], [(0.0, 1.0), (1.18, 1.88)])
        rendered[0]["text"] = "output mutation"
        self.assertEqual(segments, original)

    def test_tts_missing_or_empty_utterance_cannot_claim_complete_timing(self):
        source = [{"id": 1, "text": "Olá", "translated_text": "Olá"}]
        for counts in ([], [0], [True], [1.5]):
            with self.subTest(counts=counts), self.assertRaises(runner.PipelineError):
                runner.tts_segments_for_frames(source, counts)

    def test_tts_process_exports_output_timeline_and_keeps_stage_input_immutable(self):
        package = self.root / "package"
        reference = package / "input/reference.wav"
        reference.parent.mkdir(parents=True)
        shutil.copy2(self.raw, reference)
        manifest = {"schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": "abcde1234567",
                    "operation": "text_to_voice", "input_file": "input/reference.wav", "input_sha256": runner.sha256(reference),
                    "target_languages": ["pt-BR"], "authorization": {"voice_clone_confirmed": True},
                    "inputs": {"reference": {"path": "input/reference.wav", "sha256": runner.sha256(reference)}},
                    "options": {"text": "Olá. Boa noite.", "source_language": "pt", "require_lipsync": False}}
        runner.write_json(package / "manifest.json", manifest)
        second = self.root / "second.wav"
        self.sine(second, 0.7)
        staged_inputs = []
        def actual_media_stage(cache, key, label, config, required):
            self.assertEqual(config["stage"], "synthesis")
            directory = Path(config["directory"])
            source = Path(config["segments"])
            before = source.read_bytes()
            input_segments = runner.read_json(source)["segments"]
            chunks = [self.samples(path) for path in (self.raw, second)]
            with wave.open(str(directory / "dialogue.wav"), "wb") as output:
                output.setnchannels(2)
                output.setsampwidth(2)
                output.setframerate(48000)
                output.writeframes(chunks[0][2] + b"\x00" * (8640 * 4) + chunks[1][2])
            emitted = runner.tts_segments_for_frames(input_segments, [item[0] for item in chunks])
            runner.write_json(directory / "voice-segments.json", {"target_language": config["language"], "segments": emitted})
            runner.write_json(directory / "voice-proof.json", {"voice_clone": {"required": True, "confirmed": True, "authorized": True},
                                                               "test_fixture": "Synthetic tones do not verify cloned speech"})
            self.assertIn(directory / "voice-segments.json", required)
            self.assertEqual(source.read_bytes(), before)
            staged_inputs.append(source)
            cache.seconds[key], cache.used[key] = 0.001, False
        output = self.root / "result.zip"
        with patch.object(bootstrap, "gpu_precheck", return_value="Mock GPU, protocol only"), \
             patch.object(translator, "split_text", return_value=["Olá.", "Boa noite."]), \
             patch.object(runner.StageCache, "stage", actual_media_stage):
            result = runner.process(package, self.root / "work", self.root / "storage", output)
        record = result["outputs"][0]
        self.assertEqual(record["status"], "complete")
        self.assertTrue(record["files"]["segments"].endswith("voice-segments.json"))
        with zipfile.ZipFile(output) as archive:
            emitted = json.loads(archive.read(record["files"]["segments"]))
            report = json.loads(archive.read(record["files"]["report"]))
        self.assertAlmostEqual(emitted["segments"][-1]["end"], report["summary"]["duration"], places=3)
        for path in staged_inputs:
            self.assertTrue(all(item["start"] == item["end"] == 0 for item in runner.read_json(path)["segments"]))


if __name__ == "__main__":
    unittest.main()
