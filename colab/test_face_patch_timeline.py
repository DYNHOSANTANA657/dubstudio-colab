"""CPU regressions for frame-preserving MuseTalk composition, not visual quality."""
import ast
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker
from test_perioral_composition import face68


def synthetic_timeline_landmarks(source, boxes, start=0, width=120, height=80):
    """Artificial test anatomy ONLY, not DWPose inference or visual approval.

    Testsrc2 has no person. Explicit landmarks let these CPU tests exercise the
    real mask/clock guards without inventing production evidence or a fallback.
    """
    binding = worker.perioral_landmark_binding(source, boxes, width, height, start)
    frames = []
    for index, (x1, y1, x2, y2) in enumerate(boxes):
        scale = min((x2-x1)/160, (y2-y1)/130)
        center_x, center_y = (x1+x2)/2, y1+(y2-y1)*.55
        points = [[center_x+(x-120)*scale, center_y+(y-150)*scale, confidence]
                  for x, y, confidence in face68()]
        frames.append({"frame_index": binding["start_frame"]+index, "points": points})
    return {**binding, "device": "cpu", "model_sha256": "1"*64, "frames": frames}


class MuseTalkIntegerAudioClockTests(unittest.TestCase):
    pinned = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
    source = ("import math\nclass AudioProcessor:\n"
              "    def get_whisper_chunk(self, librosa_length, fps=25):\n"
              "        sr = 16000\n        audio_fps = 50\n        fps = int(fps)\n"
              "        num_frames = math.floor((librosa_length / sr) * fps)\n"
              "        actual_length = math.floor((librosa_length / sr) * audio_fps)\n"
              "        return num_frames, actual_length\n")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                    prefix=".test-pcm-clock-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.file = self.root / "musetalk/utils/audio_processor.py"
        self.file.parent.mkdir(parents=True)
        self.file.write_bytes(self.source.encode())

    def apply(self, helper=None, commit=None):
        head = subprocess.CompletedProcess([], 0, (commit or self.pinned) + "\n", "")
        with patch.object(subprocess, "run", return_value=head):
            return (helper or worker.patch_musetalk_audio_frame_clock)(self.root)

    def counter(self):
        namespace = {}
        exec(compile(self.file.read_bytes(), str(self.file), "exec"), namespace)
        return namespace["AudioProcessor"]().get_whisper_chunk

    def test_exact_runtime_232_frames_are_not_lost_to_float_floor(self):
        self.assertEqual(self.counter()(148480), (231, 463))
        result = self.apply()
        self.assertEqual(self.counter()(148480), (232, 464))
        self.assertTrue(result["changed"])
        self.assertEqual(result["commit"], self.pinned)
        self.assertEqual(result["revision"], "musetalk-pcm-integer-frame-clock-v1")

    def test_all_29989_canonical_lengths_use_the_exact_pcm_clock(self):
        self.apply()
        count = self.counter()
        dropped = 0
        for frames in range(12, 30001):
            samples = frames * 640
            dropped += math.floor((samples / 16000) * 25) != frames
            self.assertEqual(count(samples), (frames, frames * 2))
        self.assertEqual(dropped, 1849)

    def test_partial_pcm_frame_is_floored_not_padded_or_extended(self):
        self.apply()
        count = self.counter()
        for samples in (1, 639, 640, 641, 148479, 148480, 148481):
            self.assertEqual(count(samples), (samples * 25 // 16000, samples * 50 // 16000))

    def test_idempotence_preserves_bytes_and_reports_verified_hashes(self):
        before = self.file.read_bytes()
        first = self.apply()
        after = self.file.read_bytes()
        second = self.apply()
        self.assertNotEqual(before, after)
        self.assertEqual(self.file.read_bytes(), after)
        self.assertEqual(first["source_sha256_before"], hashlib.sha256(before).hexdigest())
        self.assertEqual(first["source_sha256_after"], hashlib.sha256(after).hexdigest())
        self.assertEqual(second["source_sha256_before"], second["source_sha256_after"])
        self.assertFalse(second["changed"])

    def test_unknown_commit_never_edits_runtime_code(self):
        before = self.file.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "revisão de código fixada"):
            self.apply(commit="unknown-commit")
        self.assertEqual(self.file.read_bytes(), before)

    def test_unknown_clock_contracts_never_write_partial_patch(self):
        changed = (
            self.source.replace("sr = 16000", "sr = 24000"),
            self.source.replace("fps = int(fps)", "fps = 30"),
            self.source.replace("math.floor((librosa_length / sr) * fps)", "math.ceil((librosa_length / sr) * fps)"),
            self.source.replace("math.floor((librosa_length / sr) * fps)", "(librosa_length * fps) // sr"),
            self.source.replace("        return", "        num_frames = 100\n        return"),
            self.source.replace("def get_whisper_chunk", "def unknown_processor"),
        )
        for source in changed:
            with self.subTest(source=source):
                before = source.encode()
                self.file.write_bytes(before)
                with self.assertRaisesRegex(RuntimeError, "fonte incompatível"):
                    self.apply()
                self.assertEqual(self.file.read_bytes(), before)

    def test_crlf_and_unrelated_code_are_preserved(self):
        self.file.write_bytes((self.source + "# preserve original code\n").replace("\n", "\r\n").encode())
        self.apply()
        after = self.file.read_bytes()
        self.assertIn(b"# preserve original code\r\n", after)
        self.assertNotIn(b"\n", after.replace(b"\r\n", b""))
        self.assertEqual(self.counter()(148480), (232, 464))

    def test_atomic_replace_failure_leaves_original_code_intact(self):
        before = self.file.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("synthetic filesystem failure")):
            with self.assertRaises(RuntimeError):
                self.apply()
        self.assertEqual(self.file.read_bytes(), before)

    def test_patch_function_is_extractable_using_only_stdlib(self):
        tree = ast.parse(Path(worker.__file__).read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "patch_musetalk_audio_frame_clock")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "isolated-clock-helper", "exec"), namespace)
        result = self.apply(helper=namespace[function.name])
        self.assertTrue(result["changed"])
        self.assertEqual(self.counter()(148480), (232, 464))


class FacePatchGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                    prefix=".test-face-guard-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("render.mp4", "base.mp4", "original.mp4"):
            (self.root/name).write_bytes(b"synthetic media identity fixture")
        self.media = {"streams": [{"codec_type": "video", "width": 120, "height": 80,
                                   "r_frame_rate": "25/1", "avg_frame_rate": "25/1"}]}

    def compose(self):
        boxes = [(20, 10, 60, 50)]*382
        return worker.compose_face_patch(self.root / "render.mp4", self.root / "base.mp4",
                                         self.root / "original.mp4", 4.32, 15.28, 2,
                                         self.root, self.root / "log", boxes, 25, (120, 80),
                                         landmark_payload=synthetic_timeline_landmarks(self.root/"original.mp4", boxes, 4.32))

    def synthetic_mask_fixture(self, path, boxes, width, height, fps, log, **options):
        """Mock encoder bytes only; real binding/anatomy/raster guards still run."""
        self.assertEqual(fps,25)
        binding = worker.perioral_landmark_binding(options["reference_video"],boxes,width,height,options["start_seconds"])
        payload = options["landmark_payload"]
        frames = worker.validate_perioral_landmarks(payload,binding,boxes)
        path.write_bytes(b"".join(worker.perioral_mask_frame(width,height,box,frame["points"])
                                 for box,frame in zip(boxes,frames)))
        worker.write_json(path.with_suffix(".perioral.json"),{
            **binding,"mask_sha256":worker.sha256(path),"device":"cpu","model_sha256":"1"*64,
            "landmarks_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
            "eyes_forehead_alpha":0,"feather":"inward-only","composition_completed":False,
            "human_visual_review":"pending"})

    def synthetic_composition_fixture(self, command, **options):
        Path(command[-1]).write_bytes(b"synthetic output identity fixture; FFmpeg graph checked separately")

    def test_short_render_fails_before_composition_instead_of_extending_last_frame(self):
        for actual in (15.24, float("nan")):
            with self.subTest(duration=actual), \
                 patch.object(worker, "video_stream_duration", side_effect=[20, 20, actual]), \
                 patch.object(worker, "ffprobe", return_value=self.media), \
                 patch.object(worker, "video_frame_count", return_value=600), \
                 patch.object(worker, "run") as run:
                with self.assertRaisesRegex(worker.WorkerError, "composição bloqueada"):
                    self.compose()
                run.assert_not_called()

    def test_preceding_composition_cannot_silently_lose_a_frame(self):
        with patch.object(worker, "video_stream_duration", side_effect=[20, 19.966667, 15.28]), \
             patch.object(worker, "ffprobe", return_value=self.media), \
             patch.object(worker, "video_frame_count", side_effect=[600, 599]), \
             patch.object(worker, "run") as run:
            with self.assertRaisesRegex(worker.WorkerError, "composição bloqueada"):
                self.compose()
            run.assert_not_called()

    def test_output_frame_count_is_independently_verified(self):
        with patch.object(worker, "video_stream_duration", side_effect=[20, 20, 15.28]), \
             patch.object(worker, "ffprobe", return_value=self.media), \
             patch.object(worker, "video_frame_count", side_effect=[600, 600, 382, 599]), \
             patch.object(worker, "write_spatial_face_mask", side_effect=self.synthetic_mask_fixture), \
             patch.object(worker, "run"), \
             patch.object(worker, "assert_video_stream_coverage") as coverage:
            with self.assertRaisesRegex(worker.WorkerError, "alterou a contagem"):
                self.compose()
            coverage.assert_not_called()

    def test_overlay_uses_source_clock_and_exclusive_confirmed_bounds(self):
        with patch.object(worker, "video_stream_duration", side_effect=[20, 20, 15.28]), \
             patch.object(worker, "ffprobe", return_value=self.media), \
             patch.object(worker, "video_frame_count", side_effect=[600, 600, 382, 600]), \
             patch.object(worker, "write_spatial_face_mask", side_effect=self.synthetic_mask_fixture), \
             patch.object(worker, "run", side_effect=self.synthetic_composition_fixture) as run, \
             patch.object(worker, "assert_video_stream_coverage"):
            self.compose()
        command = run.call_args.args[0]
        graph = command[command.index("-filter_complex") + 1]
        self.assertIn("[1:v]setpts=PTS-STARTPTS", graph)
        self.assertIn("setpts=PTS-STARTPTS+4.320000/TB", graph)
        self.assertIn("[face][mask]alphamerge", graph)
        self.assertIn("gte(t,4.320000000)*lt(t,19.600000000)", graph)
        self.assertNotIn("concat", graph)
        self.assertNotIn("fps=", graph)
        self.assertEqual(command[command.index("-vsync") + 1], "0")

    def test_missing_spatial_boxes_blocks_full_frame_overlay(self):
        with patch.object(worker, "video_stream_duration", side_effect=[20, 20, 15.28]), \
             patch.object(worker, "ffprobe", return_value=self.media), \
             patch.object(worker, "video_frame_count", return_value=600), patch.object(worker, "run") as run:
            with self.assertRaisesRegex(worker.WorkerError, "overlay completo"):
                worker.compose_face_patch(self.root / "render.mp4", self.root / "base.mp4",
                                          self.root / "original.mp4", 4.32, 15.28, 2, self.root, self.root / "log")
            run.assert_not_called()

    def test_invalid_spatial_box_or_coverage_never_enters_composition(self):
        bad = [(20, 10, 121, 50), (-1, 10, 60, 50), (20, 10, 20, 50),
               (20, 10, float("nan"), 50), (20.5, 10, 60, 50), (20, True, 60, 50), (20, 10, 30, 20)]
        for box in bad:
            with self.subTest(box=box), self.assertRaises(worker.WorkerError):
                worker.spatial_face_boxes([box] * 25, 25, 1, (120, 80), (120, 80))
        for boxes, fps in [([], 25), ([(20, 10, 60, 50)] * 24, 25),
                           ([(20, 10, 60, 50)] * 25, float("nan"))]:
            with self.subTest(count=len(boxes), fps=fps), self.assertRaises(worker.WorkerError):
                worker.spatial_face_boxes(boxes, fps, 1, (120, 80), (120, 80))

    def test_mask_is_zero_outside_box_and_feathers_only_inward(self):
        mask = worker.spatial_face_mask_frame(120, 80, (20, 10, 60, 50))
        self.assertEqual(len(mask), 120 * 80)
        for y in range(80):
            for x in range(120):
                if not (20 <= x < 60 and 10 <= y < 50):
                    self.assertEqual(mask[y*120+x], 0)
        self.assertEqual(mask[10*120+20], 0)
        self.assertGreater(mask[12*120+22], 0)
        self.assertLess(mask[12*120+22], 255)
        self.assertEqual(mask[30*120+40], 255)

    def test_scaled_boxes_are_bound_to_original_resolution(self):
        scaled = worker.spatial_face_boxes([(20, 10, 60, 50)] * 25, 25, 1, (120,80), (240,160))
        self.assertEqual(scaled, [(40,20,120,100)]*25)

    def test_noncanonical_mask_rate_is_rejected_before_creating_files_or_processes(self):
        for rate in (24, 30, 30000/1001, float("nan"), True, None, "invalid"):
            with self.subTest(rate=rate), patch.object(worker.subprocess, "Popen") as process:
                with self.assertRaisesRegex(worker.WorkerError, "canônico de 25 fps"):
                    worker.write_spatial_face_mask(self.root / "mask.mkv", [(20,10,60,50)],
                                                   120, 80, rate, self.root / "log")
                process.assert_not_called()

    def test_noncanonical_box_rate_is_not_resampled_or_relabelled(self):
        for rate in (24, 30, 30000/1001, True):
            with self.subTest(rate=rate), self.assertRaises(worker.WorkerError):
                worker.spatial_face_boxes([(20,10,60,50)]*30, rate, 30/rate, (120,80), (120,80))

    def test_render_must_match_canonical_average_and_nominal_frame_rates(self):
        for nominal, average in (("30/1", "30/1"), ("25/1", "30/1"), ("30/1", "25/1"),
                                 ("30000/1001", "30000/1001"), ("0/0", "25/1"),
                                 (None, "25/1"), ("25/1", "invalid")):
            render_media = {"streams": [{"codec_type": "video", "r_frame_rate": nominal,
                                         "avg_frame_rate": average}]}
            with self.subTest(nominal=nominal, average=average), \
                 patch.object(worker, "video_stream_duration", side_effect=[20,20,15.28]), \
                 patch.object(worker, "video_frame_count", return_value=600), \
                 patch.object(worker, "ffprobe", side_effect=[self.media, render_media]), \
                 patch.object(worker, "write_spatial_face_mask") as mask, patch.object(worker, "run") as run:
                with self.assertRaisesRegex(worker.WorkerError, "contrato canônico de 25 fps"):
                    self.compose()
                mask.assert_not_called()
                run.assert_not_called()

    def test_render_frame_count_must_match_every_tracked_box(self):
        for render_frames in (0, 381, 383):
            with self.subTest(render_frames=render_frames), \
                 patch.object(worker, "video_stream_duration", side_effect=[20,20,15.28]), \
                 patch.object(worker, "video_frame_count", side_effect=[600,600,render_frames]), \
                 patch.object(worker, "ffprobe", return_value=self.media), \
                 patch.object(worker, "write_spatial_face_mask") as mask, patch.object(worker, "run") as run:
                with self.assertRaisesRegex(worker.WorkerError, "contrato canônico de 25 fps"):
                    self.compose()
                mask.assert_not_called()
                run.assert_not_called()

    def test_canonical_one_or_two_frame_padding_is_trimmed_not_claimed(self):
        for extra_frames in (1,2):
            with self.subTest(extra_frames=extra_frames), \
                 patch.object(worker, "video_stream_duration", side_effect=[20,20,(382+extra_frames)/25]), \
                 patch.object(worker, "video_frame_count", side_effect=[600,600,382+extra_frames,600]), \
                 patch.object(worker, "ffprobe", return_value=self.media), \
                 patch.object(worker, "write_spatial_face_mask", side_effect=self.synthetic_mask_fixture) as mask, \
                 patch.object(worker, "run", side_effect=self.synthetic_composition_fixture) as run, \
                 patch.object(worker, "assert_video_stream_coverage"):
                self.compose()
                graph = run.call_args.args[0][run.call_args.args[0].index("-filter_complex")+1]
                self.assertIn("[0:v]trim=duration=15.280000", graph)
                self.assertIn("lt(t,19.600000000)", graph)
                self.assertEqual(len(mask.call_args.args[1]),382)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class RealFacePatchTimelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parent, prefix=".test-face-patch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "original.mp4"
        self.log = self.root / "composition.log"

    def execute(self, arguments):
        return subprocess.run(arguments, capture_output=True, check=True, timeout=60).stdout

    def make_original(self, vfr=False):
        filters = ["select='not(eq(mod(n,7),1))'"] if vfr else []
        command = ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                   "testsrc2=size=120x80:rate=30:duration=20"]
        if filters:
            command.extend(["-vf", ",".join(filters), "-fps_mode:v", "vfr"])
        command.extend(["-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16",
                        "-pix_fmt", "yuv420p", str(self.original)])
        self.execute(command)

    def render(self, start, duration, index):
        output = self.root / f"render-{index}.mp4"
        self.execute(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.original),
                      "-ss", str(start), "-t", str(duration), "-vf",
                      "setpts=PTS-STARTPTS,fps=25:start_time=0,drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill",
                      "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", str(output)])
        return output

    def timestamps(self, path):
        result = self.execute(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                               "-of", "json", str(path)])
        return [float(item["best_effort_timestamp_time"]) for item in json.loads(result)["frames"]]

    def gray_frames(self, path):
        return self.execute(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
                             "-f", "rawvideo", "-pix_fmt", "gray", "-fps_mode:v", "passthrough", "-"])

    def check_windows(self):
        first_boxes = [(20, 10, 60, 50)] * 97
        second_boxes = [(20 if frame < 180 else 60, 10, 60 if frame < 180 else 100, 50)
                        for frame in range(382)]
        first = worker.compose_face_patch(self.render(0, 3.88, 1), self.original, self.original,
                                          0, 3.88, 1, self.root, self.log, first_boxes, 25, (120,80),
                                          landmark_payload=synthetic_timeline_landmarks(self.original,first_boxes))
        final = worker.compose_face_patch(self.render(4.32, 15.28, 2), first, self.original,
                                          4.32, 15.28, 2, self.root, self.log, second_boxes, 25, (120,80),
                                          landmark_payload=synthetic_timeline_landmarks(self.original,second_boxes,4.32))
        source_pts, output_pts = self.timestamps(self.original), self.timestamps(final)
        self.assertEqual(len(output_pts), len(source_pts), "No original frame may disappear or be padded")
        self.assertEqual(output_pts, source_pts, "The original timeline must remain the composition clock")
        source_frames, output_frames = self.gray_frames(self.original), self.gray_frames(final)
        size = 120 * 80
        self.assertEqual(len(output_frames), len(source_frames))
        for index, timestamp in enumerate(source_pts):
            output = output_frames[index * size:(index + 1) * size]
            original = source_frames[index * size:(index + 1) * size]
            in_face = timestamp < 3.88 or 4.32 <= timestamp < 19.6
            if in_face:
                boxes = first_boxes if timestamp < 3.88 else second_boxes
                local_time = timestamp if timestamp < 3.88 else timestamp - 4.32
                selected = min(len(boxes)-1, int(local_time*25 + 1e-5))
                x1,y1,x2,y2 = boxes[selected]
                # The fixture now patches the mouth, not the entire face box.
                # Keep the white-patch assertion on its opaque anatomical core.
                mouth_x,mouth_y = (x1+x2)/2,y1+(y2-y1)*.55
                center = [output[int(mouth_y)*120+int(mouth_x)]]
                self.assertGreater(sum(center)/len(center), 235,
                                   f"Confirmed perioral mouth {index} at {timestamp}s must receive the patch")
                outside = [(output[y*120+x], original[y*120+x]) for y in range(80) for x in range(120)
                           if x < x1-3 or x >= x2+3 or y < y1-3 or y >= y2+3]
                mse = sum((a-b)**2 for a,b in outside)/len(outside)
                self.assertLess(mse, 35, f"Nonface pixels at {timestamp}s must remain source, not full render")
            else:
                mse = sum((a - b) ** 2 for a, b in zip(output, original)) / size
                self.assertLess(mse, 35, f"Frame {index} outside the confirmed interval must remain original")
        self.assertAlmostEqual(worker.video_stream_duration(final), worker.video_stream_duration(self.original), places=5)

    def test_non_aligned_25fps_windows_preserve_all_600_cfr_frames(self):
        self.make_original()
        self.assertEqual(worker.video_frame_count(self.original), 600)
        self.check_windows()

    def test_original_vfr_timestamps_and_unconfirmed_images_are_preserved(self):
        self.make_original(vfr=True)
        pts = self.timestamps(self.original)
        self.assertGreater(len(set(round(b - a, 6) for a, b in zip(pts, pts[1:]))), 1)
        self.check_windows()

    def test_actual_30fps_render_is_rejected_for_25fps_boxes_before_composition(self):
        self.make_original()
        render = self.root / "mismatched-30fps.mp4"
        self.execute(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                      "color=white:size=120x80:rate=30:duration=1.2", "-an", "-c:v", "libx264", str(render)])
        with patch.object(worker, "write_spatial_face_mask") as mask:
            with self.assertRaisesRegex(worker.WorkerError, "contrato canônico de 25 fps"):
                worker.compose_face_patch(render, self.original, self.original, .48, 1.2, 1,
                                          self.root, self.log, [(20,10,60,50)]*30, 25, (120,80))
        mask.assert_not_called()
        self.assertFalse((self.root / "lipsync-segmented-perioral-v1-01.mp4").exists())

    def test_actual_rational_rate_contract_is_rejected_before_matroska_rounding(self):
        self.make_original()
        render = self.root / "rational-render.mp4"
        rate = 30000/1001
        self.execute(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                      "color=white:size=120x80:rate=30000/1001:duration=1.001", "-an",
                      "-c:v", "libx264", str(render)])
        with patch.object(worker, "write_spatial_face_mask") as mask:
            with self.assertRaises(worker.WorkerError):
                worker.compose_face_patch(render, self.original, self.original, .5, 30/rate, 1,
                                          self.root, self.log, [(20,10,60,50)]*30, rate, (120,80))
        mask.assert_not_called()
        self.assertFalse((self.root / "lipsync-segmented-perioral-v1-01.mp4").exists())

    def test_actual_one_or_two_frame_canonical_padding_never_leaks_outside_window(self):
        self.make_original()
        source_pts = self.timestamps(self.original)
        original_gray = self.gray_frames(self.original)
        size = 120*80
        for extra_frames in (1,2):
            render = self.root / f"padded-render-{extra_frames}.mp4"
            self.execute(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                          f"color=white:size=120x80:rate=25:duration={(30+extra_frames)/25}",
                          "-an", "-c:v", "libx264", str(render)])
            with self.subTest(extra_frames=extra_frames):
                final = worker.compose_face_patch(render, self.original, self.original, .48, 1.2,
                                                  extra_frames, self.root, self.log,
                                                  [(20,10,60,50)]*30, 25, (120,80),
                                                  landmark_payload=synthetic_timeline_landmarks(self.original,[(20,10,60,50)]*30,.48))
                self.assertEqual(self.timestamps(final),source_pts)
                final_gray = self.gray_frames(final)
                for frame_index,timestamp in enumerate(source_pts):
                    if .48 <= timestamp < 1.68:
                        continue
                    frame = final_gray[frame_index*size:(frame_index+1)*size]
                    original = original_gray[frame_index*size:(frame_index+1)*size]
                    mse = sum((a-b)**2 for a,b in zip(frame,original))/size
                    self.assertLess(mse,35, f"Padding cannot leak at {timestamp}s")


if __name__ == "__main__":
    unittest.main()
