"""Real CPU audio-clock tests; synthetic signals do not certify ML speech quality."""
import array
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker


FILTER = (
    "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.95[bg];"
    "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.0[voice];"
    "[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0,"
    "loudnorm=I=-14:TP=-1.5:LRA=11[out]"
)


class MixOutputCommandTests(unittest.TestCase):
    def test_explicit_output_clock_does_not_change_filter_or_trim_speech(self):
        with patch.object(worker, "run") as run:
            output = worker.mix_audio(Path("background.wav"), Path("dialogue.wav"),
                                      2.137, Path("synthetic-job"), Path("mix.log"))
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("-filter_complex") + 1], FILTER)
        self.assertEqual(command[command.index("-t") + 1], "2.137")
        self.assertEqual(command[-7:], ["-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output)])
        self.assertEqual(output, Path("synthetic-job/dub-mix.wav"))
        self.assertNotIn("atrim", FILTER)
        self.assertNotIn("silenceremove", FILTER)
        self.assertNotIn("-shortest", command)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class MixOutputFFmpegTests(unittest.TestCase):
    rate = 48000

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                prefix=".test-mix-clock-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def signal(self, name, seconds, voice=True, loud=False):
        """A voiced-like harmonic marker stays active through the final PCM frame."""
        count = round(seconds * self.rate)
        pcm = array.array("h")
        for index in range(count):
            t = index / self.rate
            if voice:
                envelope = 0.72 + 0.28 * math.sin(2 * math.pi * 7.0 * t) ** 2
                frequency = 1080 if index >= count - round(0.18 * self.rate) else 240
                value = envelope * (math.sin(2 * math.pi * frequency * t)
                                    + 0.22 * math.sin(2 * math.pi * 2 * frequency * t))
                value *= 0.58 if loud else 0.16
                left, right = value, value * 0.86
            else:
                level = 0.20 if loud else 0.012
                left = level * math.sin(2 * math.pi * 90 * t)
                right = level * math.sin(2 * math.pi * 120 * t)
            pcm.extend((round(left * 32767), round(right * 32767)))
        if sys.byteorder != "little":
            pcm.byteswap()
        path = self.root / name
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(self.rate)
            handle.writeframes(pcm.tobytes())
        return path

    def metadata(self, path):
        data = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=codec_name,sample_rate,channels,duration,duration_ts,time_base",
            "-of", "json", str(path)
        ], text=True)
        return json.loads(data)["streams"][0]

    def decode_48k(self, path):
        data = subprocess.check_output([
            "ffmpeg", "-loglevel", "error", "-i", str(path), "-map", "0:a:0",
            "-ar", str(self.rate), "-ac", "2", "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1"
        ])
        samples = array.array("h")
        samples.frombytes(data)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples

    def old_output(self, background, dialogue, seconds):
        path = self.root / "legacy-192k.wav"
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(background), "-i", str(dialogue),
            "-filter_complex", FILTER, "-map", "[out]", "-t", str(seconds), str(path)
        ], check=True)
        return path

    def mix(self, background, dialogue, seconds):
        return worker.mix_audio(background, dialogue, seconds, self.root, self.root / "stage.log")

    def assert_clock(self, path, seconds):
        metadata = self.metadata(path)
        self.assertEqual(metadata["sample_rate"], "48000")
        self.assertEqual(metadata["channels"], 2)
        self.assertEqual(metadata["codec_name"], "pcm_s16le")
        self.assertLessEqual(abs(float(metadata["duration"]) - seconds), 0.001)
        self.assertLessEqual(abs(metadata["duration_ts"] - round(seconds * self.rate)), 1)

    def correlation(self, first, second):
        self.assertEqual(len(first), len(second))
        mean_a, mean_b = sum(first) / len(first), sum(second) / len(second)
        variance_a = sum((value - mean_a) ** 2 for value in first)
        variance_b = sum((value - mean_b) ** 2 for value in second)
        covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(first, second))
        return covariance / math.sqrt(variance_a * variance_b)

    def loudness(self, path):
        result = subprocess.run([
            "ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
            "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"
        ], capture_output=True, text=True, check=True)
        return json.loads(re.findall(r"\{[^{}]+\}", result.stderr, flags=re.DOTALL)[-1])

    def test_stereo_48k_inputs_produce_exact_stereo_pcm_48k_output(self):
        for seconds in (2.0, 2.137):
            with self.subTest(seconds=seconds):
                background = self.signal("background.wav", seconds, voice=False)
                dialogue = self.signal("dialogue.wav", seconds)
                self.assert_clock(self.mix(background, dialogue, seconds), seconds)

    def test_short_background_never_cuts_the_longer_dialogue(self):
        seconds = 2.137
        background = self.signal("background.wav", 0.73, voice=False)
        dialogue = self.signal("dialogue.wav", seconds)
        output = self.mix(background, dialogue, seconds)
        self.assert_clock(output, seconds)
        samples = self.decode_48k(output)
        self.assertEqual(len(samples), round(seconds * self.rate) * 2)
        tail = samples[-round(0.02 * self.rate) * 2:]
        self.assertGreater(math.sqrt(sum(value * value for value in tail) / len(tail)), 100)

    def test_legacy_and_new_duration_and_waveform_are_preserved(self):
        seconds = 2.137
        background = self.signal("background.wav", seconds, voice=False)
        dialogue = self.signal("dialogue.wav", seconds)
        old = self.old_output(background, dialogue, seconds)
        new = self.mix(background, dialogue, seconds)
        self.assertEqual(self.metadata(old)["sample_rate"], "192000")
        self.assert_clock(new, seconds)
        self.assertLessEqual(abs(float(self.metadata(old)["duration"]) - seconds), 0.001)
        old_pcm, new_pcm = self.decode_48k(old), self.decode_48k(new)
        self.assertEqual(len(new_pcm), round(seconds * self.rate) * 2)
        self.assertGreater(self.correlation(old_pcm, new_pcm), 0.99999)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(old_pcm, new_pcm)), 2)
        old_payload = self.metadata(old)["duration_ts"] * 2 * 2
        new_payload = self.metadata(new)["duration_ts"] * 2 * 2
        self.assertEqual(old_payload, new_payload * 4)

    def test_last_synthetic_utterance_marker_is_complete_without_padding(self):
        seconds = 2.137
        background = self.signal("background.wav", seconds, voice=False)
        dialogue = self.signal("dialogue.wav", seconds)
        new = self.mix(background, dialogue, seconds)
        old = self.old_output(background, dialogue, seconds)
        old_pcm, new_pcm = self.decode_48k(old), self.decode_48k(new)
        tail_count = round(0.15 * self.rate) * 2
        self.assertGreater(self.correlation(old_pcm[-tail_count:], new_pcm[-tail_count:]), 0.99999)
        # Spectral projection verifies the known final marker, not intelligibility.
        tail = new_pcm[-tail_count::2]
        start = round(seconds * self.rate) - len(tail)
        cosine = sum(value * math.cos(2 * math.pi * 1080 * (start + index) / self.rate)
                     for index, value in enumerate(tail))
        sine = sum(value * math.sin(2 * math.pi * 1080 * (start + index) / self.rate)
                   for index, value in enumerate(tail))
        self.assertGreater(2 * math.hypot(cosine, sine) / len(tail), 100)
        self.assert_clock(new, seconds)

    def test_loud_input_is_not_clipped_and_matches_legacy_loudness(self):
        seconds = 3.2
        background = self.signal("background.wav", seconds, voice=False, loud=True)
        dialogue = self.signal("dialogue.wav", seconds, loud=True)
        new = self.mix(background, dialogue, seconds)
        old = self.old_output(background, dialogue, seconds)
        samples = self.decode_48k(new)
        self.assertLess(max(abs(value) for value in samples), 32767)
        old_metrics, new_metrics = self.loudness(old), self.loudness(new)
        for metric in ("input_i", "input_tp"):
            self.assertTrue(math.isfinite(float(new_metrics[metric])))
            self.assertLessEqual(abs(float(new_metrics[metric]) - float(old_metrics[metric])), 0.03)
        # FFmpeg's short-clip LRA histogram is quantized in 0.1 LU steps.
        # Here PCM differs by at most one LSB, yet the native-rate meters can
        # straddle one histogram bin. This is not a media-duration tolerance.
        self.assertTrue(math.isfinite(float(new_metrics["input_lra"])))
        self.assertLessEqual(abs(float(new_metrics["input_lra"]) - float(old_metrics["input_lra"])), 0.11)
        self.assertGreater(self.correlation(self.decode_48k(old), samples), 0.99999)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(self.decode_48k(old), samples)), 2)
        self.assertLessEqual(float(new_metrics["input_tp"]), -1.40)
        self.assert_clock(new, seconds)


if __name__ == "__main__":
    unittest.main()
