"""Offline contracts for explicit short facial windows; not visual GPU approval."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_runner_v2 as runner
import dubstudio_colab_worker as worker


def track(start=0, end=18, box=None):
    return {"track": {"frame": list(range(start, end)), "bbox": [box or [10, 10, 60, 70]] * (end - start)}}


class ShortSelectedTrackTests(unittest.TestCase):
    def associate(self, count=18, scores=None, selection=None, operation="dub_video", transcript=None):
        return runner.identity_segments([track(0, count)], [scores if scores is not None else [2.0] * count], count, 100, 100,
                                        transcript if transcript is not None else [{"start": 0, "end": count / 25, "speaker": "MAIN"}],
                                        {"track_id": 0} if selection is None else selection, operation)

    def segment(self, **kwargs):
        groups, ambiguous = self.associate(**kwargs)
        self.assertFalse(ambiguous)
        self.assertEqual(len(groups), 1)
        return groups[0]

    def test_selected_eighteen_frames_has_recalculable_positive_proof(self):
        segment = self.segment()
        proof = segment["explicit_selected_short_track"]
        self.assertEqual(proof["frame_count"], 18)
        self.assertEqual(proof["positive_score_count"], 18)
        self.assertEqual(proof["positive_score_ratio"], 1.0)
        self.assertEqual(proof["peak_score"], 2.0)
        self.assertEqual(proof["selection_source"], "track_id")
        self.assertEqual(worker.confirmed_face_minimum_frames(segment), 12)
        bounded, _ = worker.bound_confirmed_talknet_segments([segment], .72, .72)
        self.assertEqual(bounded, [segment])

    def test_explicit_bbox_binds_to_the_detected_short_track(self):
        selection = {"bbox": [.1, .1, .5, .6], "normalized": True, "frame_time": .12}
        segment = self.segment(selection=selection)
        self.assertEqual(segment["explicit_selected_short_track"]["selection_source"], "bbox")
        self.assertEqual(worker.confirmed_face_minimum_frames(segment), 12)

    def test_selected_twelve_frames_is_the_absolute_floor(self):
        segment = self.segment(count=12)
        self.assertEqual(worker.confirmed_face_minimum_frames(segment), 12)
        for count in (1, 4, 11):
            with self.subTest(count=count):
                groups, ambiguous = self.associate(count=count)
                self.assertFalse(groups)
                self.assertTrue(ambiguous)

    def test_automatic_nineteen_frames_stays_blocked(self):
        for operation in ("dub_video", "lipsync_video"):
            groups, ambiguous = self.associate(count=19, selection="auto", operation=operation)
            self.assertFalse(groups)
            self.assertTrue(ambiguous)

    def test_positive_eighty_percent_and_peak_point_eight_are_required(self):
        accepted = [0.8] * 12 + [-0.1] * 3
        segment = self.segment(count=15, scores=accepted)
        self.assertEqual(segment["explicit_selected_short_track"]["positive_score_ratio"], .8)
        self.assertEqual(worker.confirmed_face_minimum_frames(segment), 12)
        for scores in ([0.8] * 11 + [-0.1] * 4, [0.799] * 15):
            groups, ambiguous = self.associate(count=15, scores=scores)
            self.assertFalse(groups)
            self.assertTrue(ambiguous)

    def test_nonfinite_and_missing_scores_cannot_approve_dubbing(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            scores = [2.0] * 18
            scores[4] = value
            groups, ambiguous = self.associate(scores=scores)
            self.assertFalse(groups)
            self.assertTrue(ambiguous)
        groups, ambiguous = self.associate(scores=[])
        self.assertFalse(groups)
        self.assertTrue(ambiguous)

    def test_external_audio_selection_proves_geometry_not_original_speech(self):
        for scores in ([-2.0] * 18, []):
            segment = self.segment(scores=scores, operation="lipsync_video", transcript=[])
            proof = segment["explicit_selected_short_track"]
            self.assertEqual(segment["speaker_confirmation"], "external_audio")
            self.assertFalse(proof["score_evidence_required"])
            self.assertEqual(proof["scores"], [])
            self.assertIsNone(proof["peak_score"])
            self.assertEqual(worker.confirmed_face_minimum_frames(segment), 12)

    def test_manual_bbox_does_not_invent_or_guess_a_track(self):
        groups, ambiguous = self.associate(selection={"bbox": [.8, .8, .1, .1], "normalized": True})
        self.assertFalse(groups)
        self.assertTrue(ambiguous)
        groups, ambiguous = runner.identity_segments([track(), track()], [[2.0] * 18, [2.0] * 18], 18, 100, 100, [],
                                                    {"bbox": [.1, .1, .5, .6], "normalized": True}, "dub_video")
        self.assertFalse(groups)
        self.assertTrue(ambiguous)
        groups, _ = self.associate(selection={"track_id": 99})
        self.assertFalse(groups)

    def test_missing_frame_cannot_be_filled_to_meet_the_floor(self):
        raw = track(0, 18)
        del raw["track"]["frame"][9]
        del raw["track"]["bbox"][9]
        groups, ambiguous = runner.identity_segments([raw], [[2.0] * 17], 18, 100, 100, [], {"track_id": 0}, "dub_video")
        self.assertFalse(groups)
        self.assertTrue(ambiguous)

    def test_multiple_voices_still_require_review_after_selection(self):
        groups, ambiguous = self.associate(transcript=[{"start": 0, "end": .32, "speaker": "ONE"},
                                                     {"start": .32, "end": .72, "speaker": "TWO"}])
        self.assertFalse(groups)
        self.assertIn("mais de um locutor", ambiguous[0]["reason"])

    def test_ordinary_twenty_frame_windows_do_not_receive_an_exception(self):
        segment = self.segment(count=20)
        self.assertNotIn("explicit_selected_short_track", segment)
        self.assertEqual(worker.confirmed_face_minimum_frames(segment), 20)

    def test_worker_refuses_short_window_without_manual_proof(self):
        segment = self.segment()
        del segment["explicit_selected_short_track"]
        with self.assertRaises(worker.WorkerError):
            worker.bound_confirmed_talknet_segments([segment], .72, .72)

    def test_worker_recomputes_score_statistics_instead_of_trusting_summary(self):
        source = self.segment()
        modifications = [("positive_score_count", 17), ("positive_score_ratio", .8), ("peak_score", 9.0),
                         ("finite_score_count", 19), ("scores", [2.0] * 17), ("scores", [-2.0] * 18),
                         ("scores", [float("inf")] * 18), ("scores", [True] * 18)]
        for key, value in modifications:
            segment = copy.deepcopy(source)
            segment["explicit_selected_short_track"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(worker.WorkerError):
                worker.confirmed_face_minimum_frames(segment)

    def test_worker_rejects_forged_selection_bounds_digest_and_confirmation(self):
        source = self.segment()
        modifications = [("schema_version", True), ("policy", "automatic"), ("frame_count", 19),
                         ("start_frame", 1), ("end_frame", 17), ("track_id", 9), ("fps", 30),
                         ("minimum_frames", 1), ("boxes_sha256", "0" * 64), ("selection_source", "automatic"),
                         ("operation", "text_to_voice")]
        for key, value in modifications:
            segment = copy.deepcopy(source)
            segment["explicit_selected_short_track"][key] = value
            with self.subTest(key=key), self.assertRaises(worker.WorkerError):
                worker.confirmed_face_minimum_frames(segment)
        for key, value in (("face_confirmation", "detected"), ("speaker_confirmation", "active_speaker"),
                           ("selection_source", None), ("boxes", [[10, 10, 61, 70]] * 18)):
            segment = copy.deepcopy(source)
            segment[key] = value
            with self.subTest(segment_key=key), self.assertRaises(worker.WorkerError):
                worker.confirmed_face_minimum_frames(segment)

    def test_worker_cannot_change_dub_proof_into_external_audio(self):
        segment = self.segment()
        segment["explicit_selected_short_track"]["operation"] = "lipsync_video"
        with self.assertRaises(worker.WorkerError):
            worker.confirmed_face_minimum_frames(segment)

    def test_short_proof_is_not_silently_clipped_or_stretched(self):
        segment = self.segment()
        for source_duration in (.44, .68):
            with self.subTest(source_duration=source_duration), self.assertRaises(worker.WorkerError):
                worker.bound_confirmed_talknet_segments([segment], source_duration, .72)
        segment["end_frame"] = 20
        segment["boxes"] += [[10, 10, 60, 70]] * 2
        with self.assertRaises(worker.WorkerError):
            worker.confirmed_face_minimum_frames(segment)

    def test_render_guard_accepts_only_valid_selected_short_contract(self):
        segment = self.segment()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, audio = root / "source.mp4", root / "audio.wav"
            video.write_bytes(b"synthetic source; short-track protocol fixture only")
            audio.write_bytes(b"synthetic audio; no synthesis or GPU validation")
            normalized = root / "TalkNet/demo/source/pyavi/video.avi"
            normalized.parent.mkdir(parents=True)
            normalized.write_bytes(b"normalization protocol fixture")
            muse = root / "MuseTalk"
            for relative in ("models/musetalkV15/musetalk.json", "models/musetalkV15/unet.pth", "models/sd-vae/config.json",
                             "models/sd-vae/diffusion_pytorch_model.bin", "models/whisper/config.json", "models/whisper/pytorch_model.bin",
                             "models/dwpose/dw-ll_ucoco_384.pth", "models/face-parse-bisent/79999_iter.pth"):
                path = muse / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"weight protocol fixture; no model execution")
            def fake_run(command, **kwargs):
                Path(command[-1]).write_bytes(b"media protocol fixture; no rendered output")
            def fake_compose(*args, **kwargs):
                # This test isolates the short-track routing contract. The
                # independent perioral suite exercises actual geometry and
                # FFmpeg. Still create bound, checksummed physical fixtures so
                # the report guard is exercised rather than bypassed.
                mask = root / "musetalk-perioral-source-mask-v1-01.mkv"
                composed = root / "lipsync-segmented-perioral-v1-01.mp4"
                mask.write_bytes(b"synthetic perioral mask protocol fixture")
                composed.write_bytes(b"synthetic composition protocol fixture")
                proof = {"contract": worker.PERIORAL_MASK_CONTRACT,
                         "source_sha256": worker.sha256(video),
                         "composition_completed": True, "eyes_forehead_alpha": 0,
                         "feather": "inward-only", "human_visual_review": "pending",
                         "frame_count": 18, "mask_sha256": worker.sha256(mask),
                         "output_sha256": worker.sha256(composed),
                         "model_sha256": worker.sha256(muse / "models/dwpose/dw-ll_ucoco_384.pth"),
                         "landmarks_sha256": worker.sha256(normalized)}
                mask.with_suffix(".perioral.json").write_text(json.dumps(proof), encoding="utf-8")
                return video
            environment = {"DUBSTUDIO_TALKNET_ROOT": str(root / "TalkNet"), "DUBSTUDIO_MUSETALK_ROOT": str(muse)}
            with patch.dict(os.environ, environment), patch.object(worker, "ffprobe", return_value={"streams": [{"codec_type": "video", "width": 100, "height": 100}]}), \
                 patch.object(worker, "patch_musetalk_audio_frame_clock", return_value={"revision": "mock-pcm-clock"}), \
                 patch.object(worker, "video_stream_duration", return_value=.72), patch.object(worker, "video_frame_count", return_value=18), \
                 patch.object(worker, "run", side_effect=fake_run), patch.object(worker, "run_streaming") as model, \
                 patch.object(worker, "select_musetalk_output", return_value=root / "rendered.mp4"), \
                 patch.object(worker, "assert_video_stream_coverage"), patch.object(worker, "compose_face_patch", side_effect=fake_compose), \
                 patch.object(worker, "write_progress"):
                analysis = {"confirmed_face_segments": [segment], "ambiguity": []}
                self.assertEqual(worker.render_confirmed_talknet_segments(video, audio, root, analysis, root / "stage.log", False), video)
                model.assert_called_once()
                self.assertEqual(analysis["confirmed_face_segments"][0]["explicit_selected_short_track"], segment["explicit_selected_short_track"])
                forged = copy.deepcopy(segment)
                forged["explicit_selected_short_track"]["positive_score_ratio"] = .1
                with self.assertRaises(worker.WorkerError):
                    worker.render_confirmed_talknet_segments(video, audio, root, {"confirmed_face_segments": [forged]}, root / "stage.log", False)
                self.assertEqual(model.call_count, 1)


if __name__ == "__main__":
    unittest.main()
