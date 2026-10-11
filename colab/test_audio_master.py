"""CPU/FFmpeg checks of real natural-audio montage, not perceptual model quality."""

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import dubstudio_audio_master as master
import dubstudio_colab_worker as worker


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


class AudioMasterWorkerTests(unittest.TestCase):
    def test_other_mode_does_not_silently_select_audio_master(self):
        with self.assertRaises(master.AudioMasterError):
            master.validate_timeline({"timing_mode": "source", "policy": master.TIMING_POLICY})

    def test_visual_proof_requires_physical_timeline_and_original_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "base.mp4"
            video.write_bytes(b"base video")
            timeline = master.build_timeline([{"id": 1, "start": 0, "end": 1, "speaker": "A", "text": "ola"}], [48000])
            timeline.update(source_video={"fps_num": 30, "fps_den": 1, "frame_count": 30},
                            video_base={"fps_num": 30, "fps_den": 1, "frame_count": 30, "sha256": worker.sha256(video)},
                            visual_mapping=[{"source_segment_id": 1, "speaker": "A", "source_start_frame": 0, "source_end_frame": 30,
                                             "output_start_frame": 0, "output_end_frame": 30, "method": "trim",
                                             "source_face_track_id": 0, "face_confirmation": "detected", "speaker_confirmation": "active_speaker"}])
            with self.assertRaisesRegex(worker.WorkerError, "mudou"):
                worker.validate_audio_master_visual_binding(video, {"timeline": timeline})

    def test_variable_frame_rate_is_not_relabelled_at_average_speed(self):
        with self.assertRaisesRegex(master.AudioMasterError, "FPS variável"):
            master._rate({"streams": [{"codec_type": "video", "avg_frame_rate": "2997/100", "r_frame_rate": "30/1"}]})

    def test_canonical_patch_cannot_cross_native_switch_to_the_next_person(self):
        source = [{"id": 1, "start": 0, "end": 1, "speaker": "A", "text": "a"},
                  {"id": 2, "start": 1, "end": 2, "speaker": "B", "text": "b"}]
        timeline = master.build_timeline(source, [49600, 46400])
        analysis = {"confirmed_face_segments": [
            {"start_frame": start, "end_frame": end, "fps": 25, "speaker": speaker, "track_id": index,
             "boxes": [[20, 20, 220, 220]] * (end - start), "face_confirmation": "detected", "speaker_confirmation": "active_speaker"}
            for index, (start, end, speaker) in enumerate([(0, 25, "A"), (25, 50, "B")])]}
        value, groups = master.enrich_video_mapping(timeline, {"fps": 30, "frame_count": 60, "width": 240, "height": 240}, analysis)
        mapped = master.mapped_analysis(value, groups, analysis)
        mapped["mapped_source_segments"] = mapped["confirmed_face_segments"]
        canonical = {**mapped["confirmed_face_segments"][0], "start_frame": 0, "end_frame": 26, "fps": 25}
        display, proof = worker.audio_master_native_display(canonical, mapped)
        self.assertAlmostEqual(display[1], 31 / 30)
        self.assertEqual(proof["end_frame"], 31)
        self.assertEqual(proof["fps"], 30)
        self.assertLess(proof["end_frame"] / proof["fps"], canonical["end_frame"] / canonical["fps"])
        # B begins at canonical 1.04, so its first native sample is preserved,
        # not filled with A's preceding mouth or declared positively scored.
        canonical_b = {**mapped["confirmed_face_segments"][1], "start_frame": 26, "end_frame": 50, "fps": 25}
        _, proof_b = worker.audio_master_native_display(canonical_b, mapped)
        records = worker.audio_master_boundary_preservation(mapped, [proof, proof_b])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["speaker"], "B")
        self.assertEqual(records[0]["kind"], "canonical_neutral_boundary")
        self.assertIs(records[0]["machine_confirmed"], False)
        self.assertAlmostEqual(records[0]["end"] - records[0]["start"], 1 / 30)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/FFprobe necessários")
class AudioMasterRealMediaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / "source.mkv"
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=red:s=160x128:r=30:d=3",
                    "-f", "lavfi", "-i", "color=green:s=160x128:r=30:d=2",
                    "-f", "lavfi", "-i", "color=blue:s=160x128:r=30:d=1",
                    "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(cls.source)])
        cls.audio = cls.root / "dialogue.wav"
        with wave.open(str(cls.audio), "wb") as handle:
            handle.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
            # Different nonzero sample values prove the entire natural clips
            # survive unchanged, including the final one-sample tail.
            handle.writeframes((100).to_bytes(2, "little", signed=True) * 2 * 336000)
            handle.writeframes((200).to_bytes(2, "little", signed=True) * 2 * 48000)
            handle.writeframes((300).to_bytes(2, "little", signed=True) * 2 * 96000)
        cls.segments = [{"id": 1, "start": 0, "end": 3, "speaker": "A", "text": "primeira fala"},
                        {"id": 2, "start": 3, "end": 5, "speaker": "B", "text": "segunda fala"},
                        {"id": 3, "start": 5, "end": 6, "speaker": "A", "text": "ultima fala"}]
        cls.timeline = master.build_timeline(cls.segments, [336000, 48000, 96000])
        cls.timeline.update(source_video_sha256=worker.sha256(cls.source),
                            audio_master={"path": "dialogue.wav", "sha256": worker.sha256(cls.audio),
                                          "sample_rate": 48000, "sample_count": 480000})
        cls.timeline_path = cls.root / "audio-master.json"
        write_json(cls.timeline_path, cls.timeline)
        cls.analysis = {"status": "complete", "confirmed_face_segments": [
            {"start_frame": start, "end_frame": end, "fps": 25.0, "speaker": speaker, "track_id": track,
             "boxes": [[20, 20, 140, 118]] * (end - start), "face_confirmation": "detected",
             "speaker_confirmation": "active_speaker", "coordinate_source": "source-verified"}
            for start, end, speaker, track in [(0, 75, "A", 0), (75, 125, "B", 1), (125, 150, "A", 0)]]}
        cls.analysis_path = cls.root / "source-analysis.json"
        write_json(cls.analysis_path, cls.analysis)
        cls.output = cls.root / "base-video.mp4"
        cls.result = master.retime_video({"video": str(cls.source), "timeline": str(cls.timeline_path),
            "analysis": str(cls.analysis_path), "output": str(cls.output), "mapped_analysis": str(cls.root / "mapped-analysis.json")}, worker)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_three_turn_real_video_is_ten_seconds_not_original_six(self):
        self.assertAlmostEqual(worker.video_stream_duration(self.output), 10)
        self.assertEqual(worker.video_frame_count(self.output), 300)
        self.assertAlmostEqual(worker.source_frame_rate(self.output), 30)

    def test_master_voice_bytes_and_source_file_are_unchanged(self):
        self.assertEqual(worker.sha256(self.audio), self.timeline["audio_master"]["sha256"])
        self.assertEqual(worker.sha256(self.source), self.timeline["source_video_sha256"])
        self.assertEqual(json.loads(self.analysis_path.read_text(encoding="utf-8")), self.analysis)

    def test_video_contains_the_correct_turn_not_the_original_next_person(self):
        for second, channel in [(0, 0), (3.5, 0), (6.9, 0), (7.1, 1), (7.9, 1), (8.1, 2), (9.9, 2)]:
            with self.subTest(second=second):
                result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(second),
                    "-i", str(self.output), "-frames:v", "1", "-vf", "scale=1:1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertEqual(max(range(3), key=lambda index: result.stdout[index]), channel)

    def test_mapped_canonical_clock_and_boxes_bind_to_real_source_fps(self):
        analysis = self.result["mapped_analysis"]
        self.assertEqual(analysis["timeline_sha256"], worker.sha256(self.timeline_path))
        self.assertEqual(worker.video_frame_count(Path(analysis["normalized_video"])), 250)
        self.assertEqual([item["speaker"] for item in analysis["confirmed_face_segments"]], ["A", "B", "A"])
        self.assertEqual([(item["start_frame"], item["end_frame"]) for item in analysis["confirmed_face_segments"]], [(0, 175), (175, 200), (200, 250)])
        worker.validate_audio_master_visual_binding(self.output, analysis)

    def test_changed_source_or_audio_blocks_before_any_new_render(self):
        bad = copy.deepcopy(self.timeline)
        bad["source_video_sha256"] = "f" * 64
        path = self.root / "wrong-timeline.json"
        write_json(path, bad)
        with self.assertRaises(master.AudioMasterError), patch.object(worker, "run") as run:
            master.retime_video({"video": str(self.source), "timeline": str(path), "analysis": str(self.analysis_path)}, worker)
        run.assert_not_called()

    def test_natural_audio_export_finishes_with_master_not_original_video(self):
        outdir = self.root / "export"
        outdir.mkdir(exist_ok=True)
        timeline = self.result["timeline"]
        exported = worker.finalize(self.output, self.audio, self.source, outdir, 10,
                                   outdir / "stage.log", timing_mode="audio_master", timeline=timeline)
        self.assertEqual(worker.video_frame_count(exported), 300)
        self.assertAlmostEqual(worker.video_stream_duration(exported), 10)

    def test_truncated_mix_cannot_hide_last_speech_during_export(self):
        short = self.root / "short-mix.wav"
        with wave.open(str(short), "wb") as handle:
            handle.setparams((2, 2, 48000, 0, "NONE", "not compressed"))
            handle.writeframes(b"\x00" * (480000 - 1) * 4)
        outdir = self.root / "export-short"
        outdir.mkdir(exist_ok=True)
        with self.assertRaisesRegex(worker.WorkerError, "perdeu amostras"):
            worker.finalize(self.output, short, self.source, outdir, 10, outdir / "stage.log",
                            timing_mode="audio_master", timeline=self.result["timeline"])
        self.assertFalse((outdir / "dublagem-final.mp4").exists())

    def test_confirmed_faceless_narration_can_follow_natural_voice_without_claiming_lip(self):
        directory = self.root / "narration"
        directory.mkdir(exist_ok=True)
        # Same immutable source and complete audio, independent output edit.
        timeline_file, analysis_file = directory / "audio-master.json", directory / "analysis.json"
        write_json(timeline_file, self.timeline)
        write_json(analysis_file, {"status": "complete", "confirmed_face_segments": [],
                                  "no_visible_speaking_face": True, "scene_analysis_complete": True})
        result = master.retime_video({"video": str(self.source), "timeline": str(timeline_file),
            "audio": str(self.audio), "analysis": str(analysis_file), "output": str(directory / "base.mp4"),
            "require_faces": False}, worker)
        self.assertEqual(worker.video_frame_count(Path(result["output"])), 300)
        proof = result["mapped_analysis"]
        self.assertTrue(proof["no_visible_speaking_face"])
        self.assertTrue(proof["scene_analysis_complete"])
        self.assertFalse(proof["safe_for_single_face_lipsync"])
        self.assertEqual(proof["confirmed_face_segments"], [])
        self.assertEqual(proof["preserved_no_speaker_visual_segments"][0]["end"], 10)

    def test_face_detection_absence_without_complete_scene_analysis_is_not_narration_proof(self):
        directory = self.root / "unconfirmed-narration"
        directory.mkdir(exist_ok=True)
        timeline_file, analysis_file = directory / "audio-master.json", directory / "analysis.json"
        write_json(timeline_file, self.timeline)
        write_json(analysis_file, {"confirmed_face_segments": [], "no_visible_speaking_face": True})
        with self.assertRaisesRegex(master.AudioMasterError, "explicitamente confirmada"):
            master.retime_video({"video": str(self.source), "timeline": str(timeline_file), "audio": str(self.audio),
                "analysis": str(analysis_file), "require_faces": False}, worker)
        self.assertFalse((directory / "base-video.mp4").exists())

    def test_background_is_cut_and_repeated_without_changing_voice(self):
        background = self.root / "background.wav"
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "sine=frequency=440:sample_rate=48000:duration=6", "-ac", "2", "-c:a", "pcm_s16le", str(background)])
        output = self.root / "background-retimed.wav"
        master.retime_background(background, output, self.result["timeline"], worker, self.root / "stage.log")
        with wave.open(str(output), "rb") as handle:
            self.assertEqual(handle.getnframes(), 480000)
        self.assertEqual(worker.sha256(self.audio), self.timeline["audio_master"]["sha256"])

    def test_canonical_face_coordinates_follow_exact_decoded_frames_not_guessed_time(self):
        source = self.root / "frame-clock.mkv"
        normalized = self.root / "frame-clock-25.mkv"
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "nullsrc=s=160x128:r=30:d=2", "-vf", "geq=lum='N*3':cb='128':cr='128'",
                    "-an", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(source)])
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-vf", "setpts=PTS-STARTPTS,fps=25:round=near:start_time=0", "-an",
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(normalized)])
        def frames(path):
            result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                "-an", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True)
            self.assertEqual(result.returncode, 0)
            size = 160 * 128
            return [result.stdout[offset:offset + size] for offset in range(0, len(result.stdout), size)]
        original, canonical = frames(source), frames(normalized)
        self.assertEqual(len(original), 60)
        self.assertEqual(len(canonical), 50)
        for index, pixels in enumerate(canonical):
            with self.subTest(frame=index):
                selected = master.canonical_source_frame(index, Fraction(30), 60)
                self.assertEqual(pixels, original[selected], f"Coordinates would be from wrong frame {selected}")

    def test_real_perioral_composition_preserves_next_person_at_native_noncanonical_cut(self):
        # GPU/DWPose are mocked with explicitly synthetic landmarks. The real
        # compositor, alpha mask, native clock and FFmpeg export are executed.
        from test_perioral_composition import evidence
        directory = self.root / "native-boundary"
        directory.mkdir(exist_ok=True)
        source, render = directory / "base.mkv", directory / "render.mkv"
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=red:s=240x240:r=30:d=2", "-vf",
                    "drawbox=x=0:y=0:w=iw:h=ih:color=blue:t=fill:enable='gte(n,31)'",
                    "-frames:v", "32", "-an", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(source)])
        worker.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=white:s=240x240:r=25:d=1.04", "-frames:v", "26", "-an", "-c:v", "ffv1",
                    "-pix_fmt", "yuv420p", str(render)])
        boxes = [(20, 20, 220, 220)] * 26
        output = worker.compose_face_patch(render, source, source, 0, 26 / 25, 1, directory,
                                          directory / "stage.log", face_boxes=boxes, box_fps=25,
                                          box_size=(240, 240), landmark_payload=evidence(source, boxes),
                                          display_window=(0.0, 31 / 30))
        self.assertEqual(worker.video_frame_count(output), 32)
        for frame, expected in [(30, "white"), (31, "blue")]:
            result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(output),
                "-vf", f"select=eq(n\\,{frame}),crop=2:2:120:150,scale=1:1", "-frames:v", "1", "-f",
                "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            red, green, blue = result.stdout
            if expected == "white":
                self.assertGreater(min(red, green, blue), 220)
            else:
                self.assertGreater(blue, 200)
                self.assertLess(max(red, green), 50)
        proof = json.loads((directory / "musetalk-perioral-source-mask-v1-01.perioral.json").read_text(encoding="utf-8"))
        self.assertEqual(proof["display_window_policy"], "audio-master-native-person-clamp-v1")
        self.assertAlmostEqual(proof["display_window"]["end"], 31 / 30)


if __name__ == "__main__":
    unittest.main()
