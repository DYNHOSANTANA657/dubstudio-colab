"""CPU-only recovery regressions; models, GPU and downloads are mocked."""
import contextlib
import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner
import test_bootstrap_setup as setup_protocol
import test_lipsync_contract as lip_protocol


class CheckpointSchemaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                   prefix=".test-checkpoint-schema-")
        self.root = Path(self.temporary.name)
        self.job, self.saved = self.root / "job", self.root / "saved"
        self.job.mkdir()
        self.source = self.job / "source.mp4"
        self.source.write_bytes(b"immutable protocol source, not playable media")
        self.output = self.job / "voice.wav"
        self.output.write_bytes(b"existing output fixture, not speech")
        self.config = {"stage": "synthesis", "directory": str(self.job)}
        self.cache = runner.StageCache(self.job, self.saved)
        self.marker = self.job / "checkpoints/voice.json"
        self.signature = runner.fingerprint({"pipeline": runner.PIPELINE_VERSION,
                                             "worker": self.cache.worker_code,
                                             "config": self.config})
        self.isolation = contextlib.ExitStack()
        self.isolation.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.isolation.enter_context(patch.object(runner.subprocess, "run",
                                                  return_value=subprocess.CompletedProcess([], 0, "", "")))
        self.isolation.enter_context(patch.dict(runner.os.environ,
                                                {"DUBSTUDIO_RUNTIME_ROOT": str(self.root / "not-a-runtime")}))

    def tearDown(self):
        self.isolation.close()
        self.temporary.cleanup()

    def execute(self):
        self.cache.stage("voice", "Synthetic voice stage", self.config, [self.output])

    def assert_rebuilt(self, record):
        runner.write_json(self.marker, record)
        with patch.object(runner, "run_command",
                          side_effect=lambda *args, **kwargs: self.output.write_bytes(b"rebuilt protocol output")) as stage:
            self.execute()
        stage.assert_called_once()
        self.assertFalse(self.cache.used["voice"])
        rebuilt = runner.read_json(self.marker)
        self.assertEqual(rebuilt["signature"], self.signature)
        self.assertEqual(rebuilt["checksums"], {"voice.wav": runner.sha256(self.output)})
        self.assertEqual(self.source.read_bytes(), b"immutable protocol source, not playable media")

    def test_nonobject_json_markers_are_cache_misses(self):
        for value in ([], None, "interrupted metadata", 42, True):
            with self.subTest(record=value):
                self.assert_rebuilt(value)

    def test_nonobject_checksum_maps_are_cache_misses(self):
        for value in ([], None, "unexpected metadata", 42, True):
            with self.subTest(checksums=value):
                self.assert_rebuilt({"signature": self.signature, "checksums": value})

    def test_missing_checksum_map_requires_rebuild(self):
        self.assert_rebuilt({"signature": self.signature})

    def test_invalid_checksum_entries_cannot_reuse_existing_output(self):
        for value in (None, False, 17, [], {}, "wrong hash"):
            with self.subTest(checksum=value):
                self.assert_rebuilt({"signature": self.signature, "checksums": {"voice.wav": value}})

    def test_failed_rebuild_preserves_corrupt_marker_and_previous_checkpoint(self):
        previous = self.job / "checkpoints/separation.json"
        runner.write_json(previous, {"prior_evidence": "preserve exactly"})
        for record in ([], {"signature": self.signature, "checksums": []}):
            with self.subTest(record=record):
                runner.write_json(self.marker, record)
                previous_bytes, marker_bytes, source_bytes = previous.read_bytes(), self.marker.read_bytes(), self.source.read_bytes()
                with patch.object(runner, "run_command", side_effect=RuntimeError("mock interruption before output")):
                    with self.assertRaisesRegex(runner.PipelineError, "mock interruption"):
                        self.execute()
                self.assertFalse(self.cache.used["voice"])
                self.assertEqual(self.marker.read_bytes(), marker_bytes)
                self.assertEqual(previous.read_bytes(), previous_bytes)
                self.assertEqual(self.source.read_bytes(), source_bytes)
                self.assertEqual((self.saved / "checkpoints/voice.json").read_bytes(), marker_bytes)

    def test_valid_marker_and_hash_still_reuse_without_execution(self):
        runner.write_json(self.marker, {"signature": self.signature,
                                       "checksums": {"voice.wav": runner.sha256(self.output)}})
        with patch.object(runner, "run_command") as stage:
            self.execute()
        stage.assert_not_called()
        self.assertTrue(self.cache.used["voice"])
        self.assertEqual(self.cache.seconds["voice"], 0.0)


class ResnetDownloadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                   prefix=".test-resnet-download-")
        self.runtime = Path(self.temporary.name)
        self.commands = []
        self.completed = [self.runtime / "models/talknet/sfd_face.pth",
                          self.runtime / "models/talknet/pretrain_TalkSet.model",
                          self.runtime / "models/musetalk/face-parse-bisent/79999_iter.pth"]
        self.resnet = self.completed[-1].with_name("resnet18-5c106cde.pth")
        for path in self.completed:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"nonempty synthetic model protocol fixture")

    def tearDown(self):
        self.temporary.cleanup()

    def prepare(self, curl_bytes=b"download protocol fixture", curl_failure=False, missing=False):
        def execute(command, **kwargs):
            self.commands.append(list(command))
            if command[0] == "curl":
                if curl_failure:
                    raise subprocess.CalledProcessError(22, command)
                if not missing:
                    Path(command[command.index("-o") + 1]).write_bytes(curl_bytes)
        with patch.object(bootstrap, "execute", side_effect=execute):
            bootstrap.download_model_weights(self.runtime,
                                             {"DUBSTUDIO_MUSETALK_ROOT": str(self.runtime / "MuseTalk")})

    def test_empty_file_after_zero_exit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Peso obrigatório.*resnet18"):
            self.prepare(curl_bytes=b"")
        self.assertFalse((self.runtime / "TalkNet-ASD/pretrain_TalkSet.model").exists())

    def test_missing_file_after_zero_exit_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Peso obrigatório.*resnet18"):
            self.prepare(missing=True)
        self.assertFalse((self.runtime / "TalkNet-ASD/pretrain_TalkSet.model").exists())

    def test_successful_nonempty_download_reaches_materialization(self):
        self.prepare()
        self.assertEqual(self.resnet.read_bytes(), b"download protocol fixture")
        self.assertEqual(sum(command[0] == "curl" for command in self.commands), 1)
        self.assertEqual((self.runtime / "TalkNet-ASD/pretrain_TalkSet.model").read_bytes(),
                         self.completed[1].read_bytes())

    def test_existing_nonempty_resnet_is_preserved_and_not_downloaded(self):
        self.resnet.write_bytes(b"previous verified session fixture")
        self.prepare()
        self.assertFalse(any(command[0] == "curl" for command in self.commands))
        self.assertEqual(self.resnet.read_bytes(), b"previous verified session fixture")

    def test_existing_empty_resnet_is_downloaded_again(self):
        self.resnet.touch()
        self.prepare()
        self.assertEqual(sum(command[0] == "curl" for command in self.commands), 1)
        self.assertGreater(self.resnet.stat().st_size, 0)

    def test_failed_download_preserves_other_completed_models(self):
        prior = {path: path.read_bytes() for path in self.completed}
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare(curl_failure=True)
        self.assertFalse(self.resnet.exists())
        for path, content in prior.items():
            self.assertEqual(path.read_bytes(), content)


class SpatialReportContractTests(unittest.TestCase):
    def proof(self, **changes):
        value = {"policy": "tracked-face-box-only-inner-feather-v1",
                 "outside_face_source": "original-frame-clock",
                 "mask_outside_tracked_box": 0, "mask_expands_box": False,
                 "segments": 2,
                 "encoding_note": "H.264 lossy encoding may alter decoded pixels; generated content is spatially confined."}
        value.update(changes)
        return value

    def test_absent_legacy_contract_does_not_invent_spatial_evidence(self):
        self.assertIsNone(runner.public_spatial_composition(None, 2))

    def test_valid_contract_is_copied_not_mutated(self):
        proof = self.proof()
        published = runner.public_spatial_composition(proof, 2)
        self.assertEqual(published, proof)
        self.assertIsNot(published, proof)

    def test_private_and_unknown_fields_are_not_exported(self):
        proof = self.proof(private_path="/content/private/voice.wav", internal_log="mock secret log",
                           encoding_note="untrusted runtime detail /content/private")
        published = runner.public_spatial_composition(proof, 2)
        self.assertEqual(set(published), set(self.proof()))
        self.assertEqual(published["encoding_note"], self.proof()["encoding_note"])
        self.assertNotIn("private", json.dumps(published))

    def test_unknown_policy_and_source_are_rejected(self):
        for changes in ({"policy": "full-frame-overlay"}, {"outside_face_source": "generated-frame"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(runner.PipelineError, "composição espacial"):
                runner.public_spatial_composition(self.proof(**changes), 2)

    def test_unsafe_mask_or_coerced_types_are_rejected(self):
        for changes in ({"mask_outside_tracked_box": 1}, {"mask_outside_tracked_box": False},
                        {"mask_outside_tracked_box": "0"}, {"mask_expands_box": True},
                        {"mask_expands_box": 0}, {"segments": True}, {"segments": "2"}):
            with self.subTest(changes=changes), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition(self.proof(**changes), 2)

    def test_segment_count_is_bounded_and_matches_facial_evidence(self):
        for segments, expected in ((1, 2), (0, 0), (-1, -1), (30001, 30001), (2, True)):
            with self.subTest(segments=segments, expected=expected), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition(self.proof(segments=segments), expected)


class SpatialReportIntegrationTests(unittest.TestCase):
    """Synthetic orchestration only; no playable output or GPU rendering."""
    setUp = lip_protocol.LipSyncOrchestrationTests.setUp
    tearDown = lip_protocol.LipSyncOrchestrationTests.tearDown
    save = lip_protocol.LipSyncOrchestrationTests.save
    configure = lip_protocol.LipSyncOrchestrationTests.configure
    execute = lip_protocol.LipSyncOrchestrationTests.execute

    def fake_stage(self, cache, key, label, config, required):
        lip_protocol.LipSyncOrchestrationTests.fake_stage(self, cache, key, label, config, required)
        if config["stage"] == "lipsync":
            payload = runner.read_json(Path(config["proof"]))
            payload["spatial_composition"] = SpatialReportContractTests().proof(segments=self.segment_count)
            payload["spatial_composition"]["private_log"] = "never export runtime logs"
            runner.write_json(Path(config["proof"]), payload)

    def test_spatial_evidence_is_copied_into_final_report_by_whitelist(self):
        self.configure("lipsync_video")
        self.segment_count = 1
        result, report = self.execute()
        self.assertEqual(result["outputs"][0]["status"], "complete")
        self.assertEqual(report["spatial_composition"], SpatialReportContractTests().proof(segments=1))
        self.assertNotIn("private_log", report["spatial_composition"])

    def test_mismatched_contract_prevents_approved_video_output(self):
        self.configure("lipsync_video")
        self.segment_count = 2
        result, report = self.execute()
        self.assertEqual(result["outputs"][0]["status"], "failed")
        self.assertNotIn("video", result["outputs"][0]["files"])
        self.assertNotIn("spatial_composition", report)


class ReadyMarkerSchemaTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                                   prefix=".test-ready-marker-")
        self.root = Path(self.temporary.name)
        self.marker = self.root / ".ready.json"
        self.fingerprint = "test-requirements-fingerprint"

    def tearDown(self):
        self.temporary.cleanup()

    def test_matching_object_confirms_ready(self):
        self.marker.write_text(json.dumps({"fingerprint": self.fingerprint}), encoding="utf-8")
        self.assertTrue(bootstrap.environment_is_ready(self.marker, self.fingerprint))

    def test_nonobject_json_does_not_certify_environment(self):
        for value in ([], None, "unexpected marker", 42, True):
            with self.subTest(value=value):
                self.marker.write_text(json.dumps(value), encoding="utf-8")
                self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))

    def test_missing_and_interrupted_marker_are_not_ready(self):
        self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))
        self.marker.write_text('{"fingerprint":', encoding="utf-8")
        self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))

    def test_mismatched_or_missing_fingerprint_requires_recheck(self):
        for value in ({}, {"fingerprint": "old requirements"}, {"fingerprint": None}, {"fingerprint": []}):
            with self.subTest(value=value):
                self.marker.write_text(json.dumps(value), encoding="utf-8")
                self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))

    def test_rejected_marker_remains_untouched_with_model_and_checkpoint(self):
        self.marker.write_bytes(b"[]")
        model, checkpoint = self.root / "weights.pth", self.root / "checkpoint.json"
        model.write_bytes(b"preserve model bytes")
        checkpoint.write_bytes(b"preserve checkpoint bytes")
        self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))
        self.assertEqual(self.marker.read_bytes(), b"[]")
        self.assertEqual(model.read_bytes(), b"preserve model bytes")
        self.assertEqual(checkpoint.read_bytes(), b"preserve checkpoint bytes")

    def test_unreadable_marker_is_not_ready(self):
        with patch.object(Path, "read_text", side_effect=OSError("mock permission denied")):
            self.assertFalse(bootstrap.environment_is_ready(self.marker, self.fingerprint))


class ReadyMarkerPreparationTests(unittest.TestCase):
    """Setup orchestration is mocked; no dependency/GPU installation is run."""
    setUp = setup_protocol.DependencyContractTests.setUp
    tearDown = setup_protocol.DependencyContractTests.tearDown
    prepare_mocked = setup_protocol.DependencyContractTests.prepare_mocked

    def ready_markers(self):
        app_requirements = Path(bootstrap.__file__).with_name("requirements-app.txt")
        muse_requirements = Path(bootstrap.__file__).with_name("requirements-musetalk.txt")
        self.app_fingerprint = hashlib.sha256(app_requirements.read_bytes() + bootstrap.PIPELINE_VERSION.encode()).hexdigest()
        self.muse_fingerprint = hashlib.sha256(muse_requirements.read_bytes() + bootstrap.MUSE_COMMIT.encode()).hexdigest()
        self.app_marker = self.runtime / "app-env/.ready.json"
        self.app_marker.write_text(json.dumps({"fingerprint": self.app_fingerprint}), encoding="utf-8")
        self.marker.write_text(json.dumps({"fingerprint": self.muse_fingerprint}), encoding="utf-8")
        return app_requirements, muse_requirements

    def test_invalid_app_marker_rechecks_only_app_not_ready_muse(self):
        app_requirements, muse_requirements = self.ready_markers()
        self.app_marker.write_text("[]", encoding="utf-8")
        muse_original = self.marker.read_bytes()
        self.prepare_mocked()
        commands = [command for command, _ in self.commands]
        self.assertTrue(any(str(app_requirements) in command for command in commands))
        self.assertFalse(any(str(muse_requirements) in command for command in commands))
        self.assertTrue(bootstrap.environment_is_ready(self.app_marker, self.app_fingerprint))
        self.assertEqual(self.marker.read_bytes(), muse_original)

    def test_invalid_muse_marker_rechecks_only_muse_preserving_app_and_cache(self):
        app_requirements, muse_requirements = self.ready_markers()
        self.marker.write_text("null", encoding="utf-8")
        app_original = self.app_marker.read_bytes()
        cache = self.runtime / "cache/model-blob"
        checkpoint = self.runtime / "jobs/synthetic/checkpoint.json"
        for path in (cache, checkpoint):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"preserve synthetic evidence")
        self.prepare_mocked()
        commands = [command for command, _ in self.commands]
        self.assertFalse(any(str(app_requirements) in command for command in commands))
        self.assertTrue(any(str(muse_requirements) in command for command in commands))
        self.assertTrue(bootstrap.environment_is_ready(self.marker, self.muse_fingerprint))
        self.assertEqual(self.app_marker.read_bytes(), app_original)
        self.assertEqual(cache.read_bytes(), b"preserve synthetic evidence")
        self.assertEqual(checkpoint.read_bytes(), b"preserve synthetic evidence")


if __name__ == "__main__":
    unittest.main()
