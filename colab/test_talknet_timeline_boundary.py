"""Source-timeline regressions; fixtures do not run MuseTalk or assert quality."""

import contextlib
import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker
from test_face_patch_timeline import synthetic_timeline_landmarks


def face_segment(start, end):
    return {
        "start_frame": start, "end_frame": end, "fps": 25.0,
        "boxes": [[10 + index, 10, 90 + index, 90] for index in range(end - start)],
        "speaker": "USER_AUDIO", "coordinate_source": "S3FD detected track",
        "face_confirmation": "detected", "speaker_confirmation": "external_audio",
    }


class TalkNetTimelineBoundaryTests(unittest.TestCase):
    def test_real_case_clips_only_two_tail_boxes_and_preserves_identity(self):
        raw = [face_segment(0, 200), face_segment(462, 502)]
        original = copy.deepcopy(raw)
        bounded, evidence = worker.bound_confirmed_talknet_segments(raw, 20.0, 20.12)
        self.assertEqual(raw, original)
        self.assertEqual(bounded[1]["end_frame"], 500)
        self.assertEqual(bounded[1]["boxes"], original[1]["boxes"][:38])
        self.assertEqual(bounded[1]["end_frame"] / bounded[1]["fps"], 20.0)
        for field in ("speaker", "coordinate_source", "face_confirmation", "speaker_confirmation"):
            self.assertEqual(bounded[1][field], original[1][field])
        self.assertEqual(evidence["clipped_segments"][0]["removed_normalized_frames"], 2)
        self.assertEqual(evidence["source_video_duration"], 20.0)
        self.assertAlmostEqual(evidence["normalization_delta_seconds"], 0.12)

    def test_fractional_last_source_frame_is_not_extended_or_invented(self):
        bounded, _ = worker.bound_confirmed_talknet_segments([face_segment(462, 502)], 20.03, 20.12)
        self.assertEqual(bounded[0]["end_frame"], 500)
        self.assertEqual(len(bounded[0]["boxes"]), 38)

    def test_exact_source_timeline_and_interior_windows_are_unchanged(self):
        raw = [face_segment(25, 100), face_segment(462, 500)]
        bounded, evidence = worker.bound_confirmed_talknet_segments(raw, 20, 20)
        self.assertEqual(bounded, raw)
        self.assertEqual(evidence["clipped_segments"], [])

    def test_significant_normalization_drift_and_invalid_durations_fail(self):
        for source, normalized in ((20, 20.2), (20, 19.8), (0, 20), (float("nan"), 20),
                                   (20, float("inf")), (-1, -1)):
            with self.subTest(source=source, normalized=normalized), self.assertRaises(worker.WorkerError):
                worker.bound_confirmed_talknet_segments([face_segment(0, 50)], source, normalized)

    def test_empty_or_short_window_after_clipping_is_not_silently_omitted(self):
        for segment, minimum in ((face_segment(500, 502), 1), (face_segment(483, 503), 20)):
            with self.subTest(start=segment["start_frame"]), self.assertRaisesRegex(worker.WorkerError, "vazio ou curto"):
                worker.bound_confirmed_talknet_segments([segment], 20, 20.12, minimum)

    def test_overlap_order_wrong_frame_rate_and_coordinates_fail(self):
        malformed = face_segment(0, 25)
        malformed["boxes"].pop()
        fractional = face_segment(0, 25)
        fractional["end_frame"] = 25.5
        bad_fps = face_segment(0, 25)
        bad_fps["fps"] = float("nan")
        cases = ([face_segment(25, 75), face_segment(50, 100)],
                 [face_segment(100, 125), face_segment(0, 25)], [malformed], [fractional], [bad_fps], [])
        for segments in cases:
            with self.subTest(segments=segments), self.assertRaises(worker.WorkerError):
                worker.bound_confirmed_talknet_segments(segments, 20, 20.12)

    def test_boxes_beyond_normalized_media_are_rejected_not_clamped(self):
        with self.assertRaisesRegex(worker.WorkerError, "duração normalizada"):
            worker.bound_confirmed_talknet_segments([face_segment(450, 501)], 20, 20)


class BoundedRenderingContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                     prefix=".test-source-timeline-")
        self.root = Path(self.temporary.name)
        self.video, self.audio = self.root / "video.mp4", self.root / "audio.wav"
        self.video.write_bytes(b"source video protocol fixture")
        self.audio.write_bytes(b"external audio protocol fixture")
        self.talknet, self.muse = self.root / "TalkNet", self.root / "MuseTalk"
        self.normalized = self.talknet / "demo/video/pyavi/video.avi"
        self.normalized.parent.mkdir(parents=True)
        self.normalized.write_bytes(b"503 frames at 25 fps protocol fixture")
        for relative in (
            "musetalkV15/musetalk.json", "musetalkV15/unet.pth", "sd-vae/config.json",
            "sd-vae/diffusion_pytorch_model.bin", "whisper/config.json", "whisper/pytorch_model.bin",
            "dwpose/dw-ll_ucoco_384.pth", "face-parse-bisent/79999_iter.pth",
        ):
            file = self.muse / "models" / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"unused weight protocol fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def test_preparation_composition_and_runner_proof_use_the_same_bound(self):
        analysis = {"confirmed_face_segments": [face_segment(0, 200), face_segment(462, 502)], "ambiguity": []}
        commands, durations, compositions, coordinate_lengths = [], {}, [], []
        source_duration = 20.0

        def command(arguments, *args, **kwargs):
            commands.append(arguments)
            output = Path(arguments[-1])
            output.write_bytes(b"prepared media protocol fixture")
            durations[output] = float(arguments[arguments.index("-t") + 1])

        def duration(file):
            return 20.12 if file == self.normalized else durations.get(file, source_duration)

        def coordinates(boxes, *args):
            coordinate_lengths.append(len(boxes))
            return boxes, "unchanged detected coordinates protocol fixture"

        def infer(arguments, *args, **kwargs):
            result_dir = Path(arguments[arguments.index("--result_dir") + 1])
            result_dir.mkdir()
            (result_dir / "new-render.mp4").write_bytes(b"render protocol fixture")
            return subprocess.CompletedProcess(arguments, 0, "Results saved", "")

        def compose(rendered, base, reference, start, length, index, directory, log, **spatial):
            self.assertEqual(spatial["box_fps"], 25)
            self.assertEqual(spatial["box_size"], (360, 640))
            self.assertEqual(len(spatial["face_boxes"]), round(length * 25))
            compositions.append((start, length))
            # Explicitly artificial encoder/output fixture, not real inference
            # or approval: the new runner summary still checks physical hashes.
            boxes = spatial["face_boxes"]
            payload = synthetic_timeline_landmarks(reference,boxes,start,360,640)
            binding = worker.perioral_landmark_binding(reference,boxes,360,640,start)
            frames = worker.validate_perioral_landmarks(payload,binding,boxes)
            mask = directory/f"musetalk-perioral-source-mask-v1-{index:02d}.mkv"
            mask.write_bytes(b"".join(worker.perioral_mask_frame(360,640,box,frame["points"])
                                     for box,frame in zip(boxes,frames)))
            output = directory/f"lipsync-segmented-perioral-v1-{index:02d}.mp4"
            output.write_bytes(b"synthetic composed timeline identity fixture")
            durations[output] = source_duration
            worker.write_json(mask.with_suffix(".perioral.json"),{
                **binding,"mask_sha256":worker.sha256(mask),"output_sha256":worker.sha256(output),
                "render_sha256":worker.sha256(rendered),"output_frame_count":round(source_duration*25),
                "device":"cpu","model_sha256":"1"*64,
                "landmarks_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                "eyes_forehead_alpha":0,"feather":"inward-only","composition_completed":True,
                "human_visual_review":"pending"})
            return output

        media = {"streams": [{"codec_type": "video", "width": 360, "height": 640}]}
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(worker.os.environ, {"DUBSTUDIO_TALKNET_ROOT": str(self.talknet),
                                                               "DUBSTUDIO_MUSETALK_ROOT": str(self.muse)}))
            stack.enter_context(patch.object(worker, "patch_musetalk_audio_frame_clock", return_value={"revision": "mock-pcm-clock"}))
            stack.enter_context(patch.object(worker, "video_stream_duration", side_effect=duration))
            stack.enter_context(patch.object(worker, "ffprobe", return_value=media))
            stack.enter_context(patch.object(worker, "run", side_effect=command))
            stack.enter_context(patch.object(worker, "video_frame_count", side_effect=lambda file: round(durations[file] * 25)))
            stack.enter_context(patch.object(worker, "resample_face_boxes", side_effect=coordinates))
            stack.enter_context(patch.object(worker, "run_streaming", side_effect=infer))
            stack.enter_context(patch.object(worker, "write_progress"))
            stack.enter_context(patch.object(worker, "assert_video_stream_coverage"))
            stack.enter_context(patch.object(worker, "compose_face_patch", side_effect=compose))
            worker.render_confirmed_talknet_segments(self.video, self.audio, self.root, analysis, self.root / "test.log", True)
        self.assertEqual(coordinate_lengths, [200, 38])
        self.assertEqual(compositions, [(0.0, 8.0), (18.48, 1.52)])
        self.assertEqual([float(command[command.index("-t") + 1]) for command in commands], [8, 8, 1.52, 1.52])
        exported = [{"start": face["start_frame"] / face["fps"], "end": face["end_frame"] / face["fps"]}
                    for face in analysis["confirmed_face_segments"]]
        self.assertEqual(exported, analysis["lipsync_confirmed_segments"])
        self.assertEqual(exported[-1]["end"], 20.0)
        self.assertEqual(analysis["spatial_composition"]["policy"],worker.PERIORAL_MASK_CONTRACT)
        self.assertEqual(analysis["spatial_composition"]["segments"],2)
        self.assertEqual(analysis["spatial_composition"]["human_visual_review"],"pending")
        self.assertEqual(analysis["preserved_no_speaker_visual_segments"], [{
            "start": 8.0, "end": 18.48,
            "reason": "TalkNet não confirmou rosto falante; imagem original preservada",
        }])

    def test_fractional_tail_is_preserved_in_composition_and_not_claimed_as_lipsync(self):
        media = {"streams": [{"codec_type": "video", "width": 360, "height": 640,
                              "r_frame_rate": "25/1", "avg_frame_rate": "25/1"}]}
        boxes = [(30, 40, 120, 180)]*38
        synthetic = synthetic_timeline_landmarks(self.video,boxes,18.48,360,640)
        (self.root/"render.mp4").write_bytes(b"synthetic render identity fixture")

        def mask_fixture(path, tracked, width, height, fps, log, **options):
            # Only encoder/container bytes are mocked. Source/clock/anatomy and
            # alpha raster guards use explicitly artificial test landmarks.
            binding = worker.perioral_landmark_binding(options["reference_video"],tracked,width,height,options["start_seconds"])
            payload = options["landmark_payload"]
            frames = worker.validate_perioral_landmarks(payload,binding,tracked)
            path.write_bytes(b"".join(worker.perioral_mask_frame(width,height,box,frame["points"])
                                     for box,frame in zip(tracked,frames)))
            worker.write_json(path.with_suffix(".perioral.json"),{
                **binding,"mask_sha256":worker.sha256(path),"device":"cpu","model_sha256":"1"*64,
                "landmarks_sha256":hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),
                "eyes_forehead_alpha":0,"feather":"inward-only","composition_completed":False,
                "human_visual_review":"pending"})

        def output_fixture(arguments,**options):
            Path(arguments[-1]).write_bytes(b"synthetic output identity fixture")

        with patch.object(worker, "video_stream_duration", side_effect=[20.03,20.03,1.52]), \
             patch.object(worker, "ffprobe", return_value=media), \
             patch.object(worker, "source_frame_rate", return_value=30), \
             patch.object(worker, "video_frame_count", side_effect=[601,601,38,601]), \
             patch.object(worker, "write_spatial_face_mask", side_effect=mask_fixture), \
             patch.object(worker, "run", side_effect=output_fixture) as command, \
             patch.object(worker, "assert_video_stream_coverage"):
            worker.compose_face_patch(self.root / "render.mp4", self.video, self.video, 18.48, 1.52,
                                      1, self.root, self.root / "test.log", boxes, 25, (360,640),landmark_payload=synthetic)
        filters = command.call_args.args[0][command.call_args.args[0].index("-filter_complex") + 1]
        self.assertIn("[1:v]setpts=PTS-STARTPTS,setsar=1[base]", filters)
        self.assertIn("trim=duration=1.520000", filters)
        self.assertIn("setpts=PTS-STARTPTS+18.480000/TB", filters)
        self.assertIn("enable='gte(t,18.480000000)*lt(t,20.000000000)'", filters)
        self.assertNotIn("concat=", filters)

    def test_composition_rejects_old_overshoot_before_running_ffmpeg(self):
        media = {"streams": [{"codec_type": "video", "width": 360, "height": 640}]}
        with patch.object(worker, "video_stream_duration", return_value=20), \
             patch.object(worker, "ffprobe", return_value=media), patch.object(worker, "run") as command:
            with self.assertRaisesRegex(worker.WorkerError, "não cabe"):
                worker.compose_face_patch(self.root / "render.mp4", self.video, self.video, 18.48, 1.6,
                                          1, self.root, self.root / "test.log")
        command.assert_not_called()

    def test_ambiguity_is_not_hidden_by_tail_clamping(self):
        analysis = {"confirmed_face_segments": [face_segment(462, 502)], "ambiguity": [{"start": 19.98, "end": 20.08}]}
        with patch.object(worker, "run_streaming") as infer, self.assertRaisesRegex(worker.WorkerError, "ambiguidades"):
            worker.render_confirmed_talknet_segments(self.video, self.audio, self.root, analysis, self.root / "test.log", True)
        infer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
