"""Offline TalkNet resume checks; no models, GPU, or real FFmpeg are used."""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as legacy
import dubstudio_runner_v2 as runner


class AnalysisResumeSafetyTests(unittest.TestCase):
    def execute_fixture(self, previous_root=False, legacy_archive=False, fail_analysis=False):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                         prefix=".test-analysis-resume-") as temporary:
            root = Path(temporary)
            talknet = root / "TalkNet"
            demo = talknet / "demo"
            demo.mkdir(parents=True)
            directory = root / "analysis"
            video = root / "video.mp4"
            video.write_bytes(b"immutable original video protocol fixture")
            stem = video.stem
            expected_evidence = {}
            if previous_root:
                old_root = demo / stem
                (old_root / "pywork").mkdir(parents=True)
                (old_root / "pywork/old-tracks.pckl").write_bytes(b"previous track evidence")
                expected_evidence["pywork/old-tracks.pckl"] = b"previous track evidence"
            if legacy_archive:
                archive = demo / f"{stem}.previous-123456"
                (archive / "nested").mkdir(parents=True)
                (archive / "nested/legacy-proof.json").write_bytes(b"legacy archive evidence")
                expected_evidence["nested/legacy-proof.json"] = b"legacy archive evidence"
            captured = {}

            def analyse(arguments, *args, **kwargs):
                requested_stem = arguments[arguments.index("--videoName") + 1]
                requested_demo = Path(arguments[arguments.index("--videoFolder") + 1])
                # Model TalkNet's glob(videoName + '.*') before it creates its
                # output directory: no archived folder may be an input match.
                captured["input_matches"] = list(requested_demo.glob(f"{requested_stem}.*"))
                self.assertEqual(captured["input_matches"], [demo / f"{stem}.mp4"])
                self.assertEqual(captured["input_matches"][0].read_bytes(), video.read_bytes())
                if fail_analysis:
                    raise RuntimeError("mocked TalkNet interruption")
                work = requested_demo / requested_stem
                (work / "pywork").mkdir(parents=True)
                (work / "pyavi").mkdir()
                track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
                (work / "pywork/tracks.pckl").write_bytes(pickle.dumps([track]))
                (work / "pywork/scores.pckl").write_bytes(pickle.dumps([[-1] * 25]))
                (work / "pyavi/video.avi").write_bytes(b"normalized analysis protocol fixture")

            info = {"duration": 1, "streams": [{"codec_type": "video", "width": 100, "height": 100}]}
            config = {"directory": str(directory), "video": str(video), "duration": 1,
                      "operation": "lipsync_video", "face_selection": "auto",
                      "output": str(directory / "analysis.json")}
            with patch.dict(runner.os.environ, {"DUBSTUDIO_TALKNET_ROOT": str(talknet)}), \
                 patch.object(runner, "media_info", return_value=info), \
                 patch.object(runner, "run_command") as command, \
                 patch.object(runner.shutil, "rmtree") as deletion, \
                 patch.object(legacy, "write_progress"), \
                 patch.object(legacy, "run_streaming", side_effect=analyse) as streaming, \
                 patch.object(legacy, "video_frame_count", return_value=25):
                if fail_analysis:
                    with self.assertRaisesRegex(RuntimeError, "mocked TalkNet interruption"):
                        runner.stage_main("analysis", config)
                else:
                    runner.stage_main("analysis", config)
                command.assert_not_called()
                deletion.assert_not_called()
                streaming.assert_called_once()

            hidden_archives = list(demo.glob(".previous-*"))
            self.assertEqual(len(hidden_archives), int(previous_root) + int(legacy_archive))
            self.assertTrue(all(archive.is_dir() for archive in hidden_archives))
            self.assertFalse(list(demo.glob(f"{stem}.previous-*")))
            for relative, data in expected_evidence.items():
                preserved = [archive / relative for archive in hidden_archives if (archive / relative).is_file()]
                self.assertEqual(len(preserved), 1, relative)
                self.assertEqual(preserved[0].read_bytes(), data)
            self.assertEqual(video.read_bytes(), b"immutable original video protocol fixture")
            if not fail_analysis:
                proof = runner.read_json(Path(config["output"]))
                self.assertTrue(proof["scene_analysis_complete"])
                self.assertEqual(proof["confirmed_face_segments"][0]["speaker_confirmation"], "external_audio")
                self.assertEqual(Path(proof["normalized_video"]).read_bytes(), b"normalized analysis protocol fixture")

    def test_existing_analysis_root_is_hidden_and_evidence_preserved(self):
        self.execute_fixture(previous_root=True)

    def test_legacy_prefixed_archive_is_migrated_before_talknet_glob(self):
        self.execute_fixture(legacy_archive=True)

    def test_current_and_legacy_evidence_survive_interrupted_resumption(self):
        self.execute_fixture(previous_root=True, legacy_archive=True, fail_analysis=True)


if __name__ == "__main__":
    unittest.main()
