"""Offline face/audio contract checks; these never assert MuseTalk visual quality."""

import json
import pickle
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_colab_worker as legacy
import dubstudio_runner_v2 as runner
import test_runner_v2 as protocol_tests


class FaceEvidenceTests(unittest.TestCase):
    def test_external_audio_auto_face_is_not_claimed_as_original_speaking_face(self):
        track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
        groups, ambiguity = runner.identity_segments([track], [[-1] * 25], 25, 100, 100, [], "auto", "lipsync_video")
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["speaker_confirmation"], "external_audio")
        self.assertEqual(groups[0]["face_confirmation"], "detected")
        self.assertIn("áudio enviado", groups[0]["coordinate_source"])
        self.assertNotIn("falante confirmado", groups[0]["coordinate_source"])

    def test_external_manual_selection_still_needs_real_detected_track(self):
        groups, ambiguity = runner.identity_segments([], [], 25, 100, 100, [], {"bbox": [0.1, 0.1, 0.4, 0.4], "normalized": True}, "lipsync_video")
        self.assertFalse(groups)
        self.assertTrue(ambiguity)

    def test_short_and_ambiguous_face_tracks_still_fail_safely(self):
        short = {"track": {"frame": list(range(19)), "bbox": [[5, 5, 80, 80]] * 19}}
        for tracks, scores in (([short], [[1] * 19]), ([short, short], [[1] * 19, [1] * 19])):
            with self.subTest(count=len(tracks)):
                groups, ambiguity = runner.identity_segments(tracks, scores, 25, 100, 100, [], "auto", "lipsync_video")
                self.assertFalse(groups)
                self.assertTrue(ambiguity)

    def test_uploaded_audio_uses_actual_video_stream_duration_not_container_duration(self):
        duration = runner.input_stream_duration({"duration": 3.1, "streams": [{"codec_type": "video", "duration": "3.0"}]}, "video")
        self.assertEqual(duration, 3.0)
        runner.validate_uploaded_audio_duration(3.0, duration)
        for value in (3.001, 3.5, float("nan"), float("inf"), -1):
            with self.subTest(duration=value), self.assertRaises(runner.PipelineError):
                runner.validate_uploaded_audio_duration(value, duration)

    def test_invalid_stream_duration_fails_before_facial_work(self):
        for value in (float("nan"), float("inf"), 0, -1, "N/A"):
            with self.subTest(duration=value), self.assertRaises(runner.PipelineError):
                runner.input_stream_duration({"duration": 3, "streams": [{"codec_type": "video", "duration": value}]}, "video")


class LipSyncOrchestrationTests(unittest.TestCase):
    setUp = protocol_tests.ProcessProtocolTests.setUp
    tearDown = protocol_tests.ProcessProtocolTests.tearDown
    save = protocol_tests.ProcessProtocolTests.save

    def configure(self, operation):
        self.manifest["operation"] = operation
        self.manifest["options"]["require_lipsync"] = True
        if operation == "lipsync_video":
            audio = self.root / "input/audio.wav"
            audio.write_bytes(b"external audio protocol fixture")
            self.manifest["inputs"]["audio"] = {"path": "input/audio.wav", "sha256": runner.sha256(audio)}
            self.manifest["target_languages"] = ["und"]
        self.save()

    def fake_stage(self, cache, key, label, config, required):
        if config["stage"] == "analysis":
            self.stage_configs.append((key, config, list(required)))
            face_free = getattr(self, "face_free", False)
            payload = {"scene_analysis_complete": True, "ambiguity": [], "normalized_video": str(required[1])}
            if face_free:
                payload.update(no_visible_speaking_face=True, confirmed_face_segments=[],
                               preserved_no_speaker_visual_segments=[{"start": 0, "end": 2, "reason": "No detected faces"}])
            else:
                payload["confirmed_face_segments"] = [{"start_frame": 0, "end_frame": 25, "fps": 25, "speaker": "USER_AUDIO",
                                                       "coordinate_source": "S3FD: rosto detectado para áudio enviado",
                                                       "face_confirmation": "detected", "speaker_confirmation": "external_audio"}]
            runner.write_json(Path(config["output"]), payload)
            required[1].parent.mkdir(parents=True, exist_ok=True)
            required[1].write_bytes(b"normalised video protocol fixture")
        elif config["stage"] == "lipsync":
            self.stage_configs.append((key, config, list(required)))
            Path(config["output"]).write_bytes(b"render protocol fixture, not visual validation")
            payload = runner.read_json(Path(config["analysis"]))
            payload.update(lipsync_confirmed_segments=[{"start": 0, "end": 1}],
                           preserved_no_speaker_visual_segments=[{"start": 1, "end": 2, "reason": "No face"}],
                           coordinate_source="S3FD coordinates for user audio")
            runner.write_json(Path(config["proof"]), payload)
        else:
            return protocol_tests.ProcessProtocolTests.fake_stage(self, cache, key, label, config, required)
        cache.seconds[key], cache.used[key] = 0.001, False

    def execute(self, media=None):
        output = self.root / "result.zip"
        def command(arguments, *args, **kwargs):
            Path(arguments[-1]).write_bytes(b"MP3 protocol fixture")
        with patch.object(bootstrap, "gpu_precheck", return_value="protocol fixture GPU, not tested"), \
             patch.object(runner, "media_info", side_effect=media or (lambda _: {"duration": 2.0, "streams": [{"codec_type": "video", "duration": "2.0"}, {"codec_type": "audio", "duration": "2.0"}]})), \
             patch.object(runner.StageCache, "stage", autospec=True, side_effect=self.fake_stage), \
             patch.object(runner, "run_command", side_effect=command):
            result = runner.process(self.root, self.root / "work", self.root / "storage", output)
        record = result["outputs"][0]
        with zipfile.ZipFile(output) as archive:
            report = json.loads(archive.read(record["files"]["report"]))
        return result, report

    def test_external_audio_result_binds_audio_and_uses_honest_geometric_evidence(self):
        self.configure("lipsync_video")
        result, report = self.execute()
        self.assertEqual(result["outputs"][0]["status"], "complete")
        expected = {role: item["sha256"] for role, item in self.manifest["inputs"].items()}
        self.assertEqual(result["input_checksums"], expected)
        self.assertEqual(report["input_checksums"], expected)
        self.assertTrue(report["lipsync"]["applied"])
        self.assertEqual(report["lipsync"]["confirmed_segments"][0]["speaker_confirmation"], "external_audio")
        self.assertIn("fala original não confirmada", report["lipsync"]["coordinate_source"])
        analysis = next(config for _, config, _ in self.stage_configs if config["stage"] == "analysis")
        self.assertEqual(analysis["audio_hash"], expected["audio"])
        self.assertFalse(any(config["stage"] in {"asr", "translation", "synthesis"} for _, config, _ in self.stage_configs))

    def test_entirely_face_free_dub_is_preserved_without_false_lipsync_claim(self):
        self.configure("dub_video")
        self.face_free = True
        result, report = self.execute()
        self.assertEqual(result["outputs"][0]["status"], "complete")
        self.assertFalse(report["lipsync"]["applied"])
        self.assertFalse(report["lipsync"]["confirmed"])
        self.assertTrue(report["lipsync"]["scene_analysis_complete"])
        self.assertFalse(any(config["stage"] == "lipsync" for _, config, _ in self.stage_configs))

    def test_entirely_face_free_audio_lipsync_is_failed_not_delivered(self):
        self.configure("lipsync_video")
        self.face_free = True
        result, report = self.execute()
        self.assertEqual(result["outputs"][0]["status"], "failed")
        self.assertFalse(report["lipsync"]["applied"])
        self.assertNotIn("video", result["outputs"][0]["files"])

    def test_audio_overshoot_fails_before_analysis_and_never_truncates(self):
        self.configure("lipsync_video")
        with self.assertRaisesRegex(runner.PipelineError, "nenhuma fala"):
            self.execute(lambda file: {"duration": 2.1 if file.suffix == ".wav" else 2.0,
                "streams": [{"codec_type": "audio", "duration": "2.1"}] if file.suffix == ".wav"
                else [{"codec_type": "video", "duration": "2.0"}, {"codec_type": "audio", "duration": "2.0"}]})
        self.assertFalse(self.stage_configs)

    def test_global_failure_before_outputs_binds_both_inputs_in_result_and_report(self):
        self.configure("lipsync_video")
        work, storage = self.root / "work", self.root / "storage"
        output = self.root / "global-failure.zip"
        failure = runner.PipelineError("global failure before facial outputs protocol fixture")
        expected = {role: item["sha256"] for role, item in self.manifest["inputs"].items()}
        self.assertEqual(set(expected), {"video", "audio"})
        self.assertFalse((work / self.manifest["job_id"] / "outputs").exists())
        arguments = ["dubstudio_runner_v2.py", "--package-root", str(self.root),
                     "--work-root", str(work), "--storage-root", str(storage), "--output", str(output)]
        with patch.object(runner.sys, "argv", arguments), \
             patch.object(runner, "process", side_effect=failure) as process, \
             patch("builtins.print"), self.assertRaises(runner.PipelineError) as raised:
            runner.main()
        self.assertIs(raised.exception, failure)
        process.assert_called_once_with(self.root, work, storage, output)
        self.assertFalse(self.stage_configs)
        with zipfile.ZipFile(output) as archive:
            result = json.loads(archive.read("result.json"))
            self.assertEqual(len(result["outputs"]), 1)
            record = result["outputs"][0]
            self.assertEqual(record["status"], "failed")
            self.assertEqual(set(record["files"]), {"report"})
            relative = record["files"]["report"]
            report = json.loads(archive.read(relative))
            self.assertEqual(set(archive.namelist()), {"result.json", relative})
        for payload in (result, report):
            self.assertEqual(payload["operation"], "lipsync_video")
            self.assertEqual(payload["input_checksums"], expected)
            self.assertEqual(payload["input_sha256"], expected["video"])
            self.assertEqual(payload["job_id"], self.manifest["job_id"])
            self.assertEqual(payload["pipeline_version"], runner.PIPELINE_VERSION)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error"], str(failure))
        self.assertEqual(record["checksums"][relative], runner.sha256(work / self.manifest["job_id"] / relative))


class MuseRuntimeEnvironmentTests(unittest.TestCase):
    def test_lipsync_stage_removes_app_allocator_and_preserves_native_driver(self):
        # Exercise the stage's real os.environ boundary, not only the helper's
        # returned dict. No model, subprocess, or GPU is started by this fixture.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                         prefix=".test-muse-runtime-") as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            muse = runtime / "musetalk-env"
            cuda_library = muse / "lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
            cuda_library.mkdir(parents=True)
            directory = root / "lipsync"
            directory.mkdir()
            normalized = root / "talknet-normalized.avi"
            normalized.write_bytes(b"normalized video protocol fixture")
            analysis = {
                "normalized_video": str(normalized), "ambiguity": [],
                "confirmed_face_segments": [{"start_frame": 0, "end_frame": 25, "fps": 25}],
            }
            analysis_file = directory / "analysis.json"
            runner.write_json(analysis_file, analysis)
            config = {
                "directory": str(directory), "analysis": str(analysis_file),
                "video": str(root / "video.mp4"), "audio": str(root / "audio.wav"),
                "output": str(directory / "synced.mp4"),
                "proof": str(directory / "lipsync-proof.json"),
            }
            driver = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
            environment = {
                "DUBSTUDIO_TALKNET_ROOT": str(runtime / "TalkNet-ASD"),
                "DUBSTUDIO_MUSETALK_PYTHON": str(muse / "bin/python"),
                "DUBSTUDIO_DRIVER_LIBRARY_PATH": driver,
                "DUBSTUDIO_SYSTEM_PATH": "/usr/bin:/bin",
                "LD_LIBRARY_PATH": str(runtime / "app-env/lib/python3.10/site-packages/torch/lib") + ":" + driver,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
            captured = {}

            def render(video, audio, job_dir, proof, log, require_lipsync, fast_profile):
                captured.update(runner.os.environ)
                rendered = job_dir / "rendered.mp4"
                rendered.write_bytes(b"render protocol fixture, not visual validation")
                return rendered, []

            with patch.dict(runner.os.environ, environment, clear=True), \
                 patch.object(legacy, "write_progress"), \
                 patch.object(legacy, "run_streaming") as streaming, \
                 patch.object(legacy, "musetalk", side_effect=render) as renderer:
                runner.stage_main("lipsync", config)
            renderer.assert_called_once()
            streaming.assert_not_called()
            self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", captured)
            self.assertEqual(captured["DUBSTUDIO_DRIVER_LIBRARY_PATH"], driver)
            self.assertEqual(captured["LD_LIBRARY_PATH"], ":".join([
                str(muse / "lib/python3.10/site-packages/torch/lib"), str(cuda_library), driver,
            ]))
            self.assertNotIn(str(runtime / "app-env"), captured["LD_LIBRARY_PATH"])
            self.assertEqual(captured["PATH"], str(muse / "bin") + ":/usr/bin:/bin")
            self.assertEqual(Path(config["output"]).read_bytes(), b"render protocol fixture, not visual validation")
            self.assertEqual(runner.read_json(Path(config["proof"])), analysis)


class AnalysisAudioTests(unittest.TestCase):
    def test_analysis_replaces_original_audio_with_uploaded_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            talknet = root / "TalkNet"
            talknet.mkdir()
            directory = root / "analysis"
            video, audio = root / "video.mp4", root / "audio.wav"
            video.write_bytes(b"video with original audio protocol fixture")
            audio.write_bytes(b"new uploaded voice protocol fixture")
            commands = []
            def command(arguments, *args, **kwargs):
                commands.append(arguments)
                Path(arguments[-1]).write_bytes(b"analysis remux fixture")
            def analyse(arguments, *args, **kwargs):
                stem = arguments[arguments.index("--videoName") + 1]
                work = talknet / "demo" / stem
                (work / "pywork").mkdir(parents=True)
                (work / "pyavi").mkdir()
                track = {"track": {"frame": list(range(25)), "bbox": [[5, 5, 80, 80]] * 25}}
                (work / "pywork/tracks.pckl").write_bytes(pickle.dumps([track]))
                (work / "pywork/scores.pckl").write_bytes(pickle.dumps([[-1] * 25]))
                (work / "pyavi/video.avi").write_bytes(b"normalised analysis fixture")
            info = {"duration": 1, "streams": [{"codec_type": "video", "width": 100, "height": 100}, {"codec_type": "audio"}]}
            config = {"directory": str(directory), "video": str(video), "audio": str(audio), "duration": 1,
                      "operation": "lipsync_video", "face_selection": "auto", "output": str(directory / "analysis.json")}
            with patch.dict(runner.os.environ, {"DUBSTUDIO_TALKNET_ROOT": str(talknet)}), \
                 patch.object(runner, "media_info", return_value=info), \
                 patch.object(runner, "run_command", side_effect=command), \
                 patch.object(legacy, "write_progress"), \
                 patch.object(legacy, "run_streaming", side_effect=analyse), \
                 patch.object(legacy, "patch_talknet_video_normalization", return_value=legacy.TALKNET_NORMALIZATION_REVISION), \
                 patch.object(legacy, "video_frame_count", return_value=25):
                runner.stage_main("analysis", config)
            self.assertEqual(len(commands), 1)
            self.assertIn(str(audio), commands[0])
            self.assertIn("1:a:0", commands[0])
            self.assertIn("speaker-analysis-input", Path(commands[0][-1]).stem)
            proof = runner.read_json(Path(config["output"]))
            self.assertEqual(proof["confirmed_face_segments"][0]["speaker_confirmation"], "external_audio")


if __name__ == "__main__":
    unittest.main()
