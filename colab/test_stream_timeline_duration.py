"""CPU-only frame/packet duration tests; no GPU/model quality claims."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_runner_v2 as runner
import dubstudio_colab_worker as worker


class StreamTimelineUnitTests(unittest.TestCase):
    def probe_result(self, collection, values):
        return subprocess.CompletedProcess([], 0, json.dumps({collection: values}), "")

    def test_absent_stream_duration_never_uses_container_without_local_frames(self):
        for stream in ({"codec_type": "video"}, {"codec_type": "video", "duration": "N/A"}):
            with self.subTest(stream=stream), self.assertRaises(runner.PipelineError):
                runner.input_stream_duration({"duration": 500, "streams": [stream]}, "video")
        with self.assertRaises(runner.PipelineError):
            runner.input_stream_duration({"duration": 500}, "video")

    def test_na_stream_uses_selected_real_frame_span_not_format_duration(self):
        frames = [{"best_effort_timestamp_time": "10.0", "pkt_duration_time": "0.2"},
                  {"best_effort_timestamp_time": "10.2", "pkt_duration_time": "0.2"}]
        with patch.object(runner.subprocess, "run", return_value=self.probe_result("frames", frames)) as probe:
            value = runner.input_stream_duration({"duration": 500, "_media_path": "local.mkv",
                "streams": [{"codec_type": "video", "duration": "N/A"}]}, "video")
        self.assertAlmostEqual(value, .4)
        self.assertIn("v:0", probe.call_args.args[0])
        self.assertIn("-show_frames", probe.call_args.args[0])

    def test_last_frame_requires_own_positive_frame_or_matching_packet_duration(self):
        frames = [{"best_effort_timestamp_time": "0", "pkt_duration_time": ".1"},
                  {"best_effort_timestamp_time": ".1"}]
        packets = [{"pts_time": ".1", "duration_time": ".3"}]
        with patch.object(runner.subprocess, "run", side_effect=[self.probe_result("frames", frames), self.probe_result("packets", packets)]):
            self.assertAlmostEqual(runner.stream_timeline_duration(Path("local.webm"), "video"), .4)

    def test_frame_without_provable_positive_duration_blocks_even_with_time_deltas(self):
        frames = [{"best_effort_timestamp_time": "0", "pkt_duration_time": ".1"},
                  {"best_effort_timestamp_time": ".1", "pkt_duration_time": "0"}]
        packets = [{"pts_time": ".1", "duration_time": "0"}]
        with patch.object(runner.subprocess, "run", side_effect=[self.probe_result("frames", frames), self.probe_result("packets", packets)]):
            with self.assertRaisesRegex(runner.PipelineError, "duração positiva verificável"):
                runner.stream_timeline_duration(Path("unknown-last.webm"), "video")

    def test_nonfinite_frame_timestamp_fails_clearly(self):
        frames = [{"best_effort_timestamp_time": "nan", "pkt_duration_time": ".1"}]
        with patch.object(runner.subprocess, "run", return_value=self.probe_result("frames", frames)):
            with self.assertRaisesRegex(runner.PipelineError, "timestamp finito"):
                runner.stream_timeline_duration(Path("broken.mkv"), "video")

    def test_tiny_positive_frame_is_not_rounded_to_zero(self):
        frames = [{"best_effort_timestamp_time": "5.0", "duration_time": "0.000004"}]
        with patch.object(runner.subprocess, "run", return_value=self.probe_result("frames", frames)):
            duration = runner.stream_timeline_duration(Path("tiny.mkv"), "video")
        self.assertGreater(duration, 0)
        self.assertAlmostEqual(duration, .000004, places=11)
        with self.assertRaises(runner.PipelineError):
            runner.validate_uploaded_audio_duration(.00001, duration)

    def test_audio_samples_are_physical_duration_not_video_container(self):
        frames = [{"best_effort_timestamp_time": "0", "nb_samples": "8000"},
                  {"best_effort_timestamp_time": ".5", "nb_samples": "8000"}]
        with patch.object(runner.subprocess, "run", return_value=self.probe_result("frames", frames)) as probe:
            value = runner.input_stream_duration({"duration": 500, "_media_path": "voice.mkv",
                "streams": [{"codec_type": "audio", "sample_rate": "16000"}]}, "audio")
        self.assertEqual(value, 1)
        self.assertIn("a:0", probe.call_args.args[0])

    def test_explicit_corrupt_duration_cannot_trigger_fallback(self):
        for value in (0, -1, "nan", "inf", True):
            with self.subTest(value=value), patch.object(runner.subprocess, "run") as probe:
                with self.assertRaises(runner.PipelineError):
                    runner.input_stream_duration({"duration": 500, "_media_path": "local.mp4",
                        "streams": [{"codec_type": "video", "duration": value}]}, "video")
                probe.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class RealStreamTimelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="tm-")
        cls.root = Path(cls.folder.name)
        cls.source = cls.root / "padded.mp4"
        cls.mkv = cls.root / "padded.mkv"
        cls.webm = cls.root / "padded.webm"
        cls.tiny = cls.root / "tiny.mkv"
        cls.make_media(cls.source, "color=c=blue:s=160x90:r=5:d=1.2", "sine=frequency=700:sample_rate=48000:duration=2.4", ["-c:v", "libx264", "-c:a", "aac"])
        subprocess.run([shutil.which("ffmpeg"), "-y", "-i", str(cls.source), "-c", "copy", str(cls.mkv)], check=True, capture_output=True)
        cls.make_media(cls.webm, "color=c=blue:s=160x90:r=5:d=1.2", "sine=frequency=700:sample_rate=48000:duration=2.4", ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-c:a", "libopus"])
        cls.make_media(cls.tiny, "color=c=blue:s=160x90:r=120:d=0.008333333", "sine=frequency=700:sample_rate=48000:duration=1.2", ["-c:v", "libx264", "-c:a", "aac"])

    @classmethod
    def make_media(cls, target, video, audio, codecs):
        subprocess.run([shutil.which("ffmpeg"), "-y", "-f", "lavfi", "-i", video,
            "-f", "lavfi", "-i", audio, *codecs, "-pix_fmt", "yuv420p", str(target)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_mkv_without_stream_duration_matches_video_images_in_both_workers(self):
        raw = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(self.mkv)], check=True, capture_output=True, text=True).stdout)
        self.assertNotIn("duration", next(item for item in raw["streams"] if item["codec_type"] == "video"))
        info = runner.media_info(self.mkv)
        self.assertGreater(info["container_duration"], 2.4)
        self.assertAlmostEqual(runner.input_stream_duration(info, "video"), 1.2, places=5)
        self.assertAlmostEqual(worker.video_stream_duration(self.mkv), 1.2, places=5)
        with self.assertRaises(runner.PipelineError):
            runner.validate_uploaded_audio_duration(1.5, info["duration"])

    def test_webm_without_stream_duration_uses_last_image_not_long_opus(self):
        info = runner.media_info(self.webm)
        self.assertGreater(info["container_duration"], 2.4)
        self.assertAlmostEqual(info["duration"], 1.2, places=5)
        self.assertAlmostEqual(worker.ffprobe(self.webm)["duration"], 1.2, places=5)
        with self.assertRaises(runner.PipelineError):
            runner.validate_uploaded_audio_duration(1.5, info["duration"])

    def test_one_tiny_frame_cannot_hide_inside_long_audio(self):
        info = runner.media_info(self.tiny)
        self.assertGreater(info["container_duration"], 1)
        self.assertGreater(info["duration"], 0)
        self.assertLess(info["duration"], .01)
        with self.assertRaises(runner.PipelineError):
            runner.validate_uploaded_audio_duration(.02, info["duration"])

    def test_long_audio_is_measured_independently_and_fails_final_coverage(self):
        info = runner.media_info(self.mkv)
        self.assertGreater(runner.input_stream_duration(info, "audio"), 2.3)
        with self.assertRaisesRegex(runner.PipelineError, "Resultado incompleto"):
            runner.verify_final_media(self.mkv, 1.2)


if __name__ == "__main__":
    unittest.main()
