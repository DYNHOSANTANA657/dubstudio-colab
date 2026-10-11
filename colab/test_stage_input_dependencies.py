"""Real StageCache checkpoints over synthetic inputs, not media/model validation."""
import copy
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_runner_v2 as runner
import test_runner_v2 as protocol_tests


class StageInputDependencyTests(unittest.TestCase):
    # Reuse only fixture helpers, never inherit or recount its existing tests.
    setUp = protocol_tests.ProcessProtocolTests.setUp
    tearDown = protocol_tests.ProcessProtocolTests.tearDown
    save = protocol_tests.ProcessProtocolTests.save
    execute_fixture = protocol_tests.ProcessProtocolTests.execute_fixture

    def fake_stage(self, cache, key, label, config, required):
        if config["stage"] != "analysis":
            return protocol_tests.ProcessProtocolTests.fake_stage(self, cache, key, label, config, required)
        self.stage_configs.append((key, config, list(required)))
        runner.write_json(Path(config["output"]), {
            "no_visible_speaking_face": True, "scene_analysis_complete": True,
            "preserved_no_speaker_visual_segments": [{"start": 0.0, "end": 2.0}],
        })
        required[1].write_bytes(b"normalized-video protocol fixture")
        cache.seconds[key], cache.used[key] = 0.001, False

    def prepare_configs(self, **options):
        self.manifest["options"].update(options)
        self.save()
        self.stage_configs.clear()
        result, _ = self.execute_fixture()
        self.assertEqual(result["outputs"][0]["status"], "complete")
        return {config["stage"]: (key, config, required)
                for key, config, required in self.stage_configs}

    def exercise_checkpoint(self, captured, path_key, hash_key, *, consumed=True, dependency_path=None):
        """Run StageCache itself; only the heavy child stage and GPU observer are fake."""
        key, config, required = captured
        config = copy.deepcopy(config)
        dependency = Path(dependency_path if dependency_path is not None else config[path_key])
        initial = dependency.read_bytes()
        original_outputs = {path: path.read_bytes() for path in required}
        initial_hash = runner.sha256(dependency)
        expected = initial_hash if consumed else None
        self.assertIn(hash_key, config)
        self.assertEqual(config[hash_key], expected)
        root = self.root / "work" / self.manifest["job_id"]
        cache = runner.StageCache(root, root)
        marker = root / "checkpoints" / f"{key}.json"

        def stage_command(command, *args, **kwargs):
            request = runner.read_json(Path(command[-1]))
            payload = b"synthetic-generated:" + (dependency.read_bytes() if consumed else b"disabled")
            self.assertEqual(request[hash_key], config[hash_key])
            for file in required:
                file.write_bytes(payload)

        try:
            with patch.object(runner, "run_command", side_effect=stage_command) as execute, \
                    patch.object(runner.threading, "Thread"), \
                    patch.dict(os.environ, {"DUBSTUDIO_RUNTIME_ROOT": str(self.root / "absent-runtime")}):
                cache.stage(key, "Synthetic dependency", config, required)
                first_marker = marker.read_bytes()
                cache.stage(key, "Synthetic dependency", config, required)
                self.assertTrue(cache.used[key])
                self.assertEqual(execute.call_count, 1)
                # Same path and same length: content, not filename/stat size, changed.
                dependency.write_bytes(bytes([initial[0] ^ 1]) + initial[1:])
                self.assertEqual(dependency.stat().st_size, len(initial))
                changed_hash = runner.sha256(dependency)
                self.assertNotEqual(changed_hash, initial_hash)
                if consumed:
                    config[hash_key] = changed_hash
                cache.stage(key, "Synthetic dependency", config, required)
                self.assertEqual(execute.call_count, 2 if consumed else 1)
                self.assertEqual(cache.used[key], not consumed)
                if consumed:
                    self.assertNotEqual(marker.read_bytes(), first_marker)
                    cache.stage(key, "Synthetic dependency", config, required)
                    self.assertTrue(cache.used[key])
                    self.assertEqual(execute.call_count, 2)
                else:
                    self.assertEqual(marker.read_bytes(), first_marker)
        finally:
            dependency.write_bytes(initial)
            for path, content in original_outputs.items():
                path.write_bytes(content)

    def test_vocals_content_invalidates_asr_diarization_and_references(self):
        configs = self.prepare_configs()
        for stage in ("asr", "diarization", "references"):
            with self.subTest(stage=stage):
                self.exercise_checkpoint(configs[stage], "vocals", "vocals_hash")

    def test_video_content_invalidates_separation_and_analysis(self):
        configs = self.prepare_configs(require_lipsync=True)
        for stage in ("separation", "analysis"):
            with self.subTest(stage=stage):
                self.exercise_checkpoint(configs[stage], "video", "video_hash")

    def test_background_content_invalidates_mix_when_preserved(self):
        configs = self.prepare_configs(preserve_ambience=True)
        self.exercise_checkpoint(configs["mix"], "background", "background_hash")

    def test_disabled_background_does_not_invalidate_mix(self):
        configs = self.prepare_configs(preserve_ambience=False)
        self.assertIsNone(configs["mix"][1]["background"])
        background = self.root / "work" / self.manifest["job_id"] / "shared/background.wav"
        self.exercise_checkpoint(configs["mix"], "background", "background_hash", consumed=False, dependency_path=background)

    def test_burned_subtitle_content_invalidates_export(self):
        for mode in ("burn", "burned"):
            with self.subTest(mode=mode):
                configs = self.prepare_configs(subtitle_mode=mode)
                self.exercise_checkpoint(configs["export"], "srt", "srt_hash")

    def test_separate_or_disabled_subtitles_do_not_invalidate_video_export(self):
        for mode in ("none", "srt"):
            with self.subTest(mode=mode):
                configs = self.prepare_configs(subtitle_mode=mode)
                self.exercise_checkpoint(configs["export"], "srt", "srt_hash", consumed=False)

    def test_new_input_hash_is_cache_miss_and_failed_rebuild_preserves_old_marker(self):
        key, current, required = self.prepare_configs()["asr"]
        current = copy.deepcopy(current)
        legacy = {name: value for name, value in current.items() if name != "vocals_hash"}
        root = self.root / "work" / self.manifest["job_id"]
        cache = runner.StageCache(root, root)
        marker = root / "checkpoints" / f"{key}.json"

        def successful_stage(*args, **kwargs):
            required[0].write_bytes(b"legacy synthetic transcript")

        with patch.object(runner.threading, "Thread"), \
                patch.dict(os.environ, {"DUBSTUDIO_RUNTIME_ROOT": str(self.root / "absent-runtime")}):
            with patch.object(runner, "run_command", side_effect=successful_stage) as execute:
                cache.stage(key, "Synthetic legacy", legacy, required)
                self.assertEqual(execute.call_count, 1)
            old_marker = marker.read_bytes()
            with patch.object(runner, "run_command", side_effect=runner.PipelineError("Synthetic interruption")) as execute:
                with self.assertRaisesRegex(runner.PipelineError, "Synthetic interruption"):
                    cache.stage(key, "Synthetic updated", current, required)
                self.assertEqual(execute.call_count, 1)
            self.assertFalse(cache.used[key])
            self.assertEqual(marker.read_bytes(), old_marker)
            with patch.object(runner, "run_command", side_effect=successful_stage) as execute:
                cache.stage(key, "Synthetic updated", current, required)
                self.assertEqual(execute.call_count, 1)
            self.assertNotEqual(marker.read_bytes(), old_marker)


if __name__ == "__main__":
    unittest.main()
