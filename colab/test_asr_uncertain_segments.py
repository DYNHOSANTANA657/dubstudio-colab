"""Fake-Whisper CPU regressions; not model/GPU accuracy approval."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import dubstudio_runner_v2 as runner

HERE = Path(__file__).resolve().parent


def word(text, start, end, probability=0.99):
    return types.SimpleNamespace(word=text, start=start, end=end, probability=probability)


def segment(start, end, text, words):
    return types.SimpleNamespace(start=start, end=end, text=text, words=words)


def good_segments():
    return [segment(0, 2, "Uma frase comum", [word("Uma", 0, 0.4), word("frase", 0.4, 1.2), word("comum", 1.2, 2)])]


def isolated_garbage():
    # 300 confident words + segment 20 reproducing observed failure values;
    # confidence averaged across the whole clip is approximately 0.9723.
    result = []
    for index in range(19):
        start, end = 74.07 * index / 19, 74.07 * (index + 1) / 19
        count = 15 if index < 16 else 20
        result.append(segment(start, end, "fala confiável", [word("voz", start + (end-start)*i/count,
                      start + (end-start)*(i+1)/count, 0.982) for i in range(count)]))
    result.append(segment(74.08, 74.2, "Sofallas P textures", [
        word("Sofallas", 74.08, 74.16, 0.004249), word("P", 74.16, 74.16, 0.000787),
        word("textures", 74.16, 74.2, 0.0000046),
    ]))
    return result


class AsrUncertainSegmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.temporary.name)
        self.output = self.root / "transcript.json"
        self.previous = b'{"approved_previous":"preserve exact bytes"}'
        self.output.write_bytes(self.previous)
        self.config = {"output": str(self.output), "vocals": str(self.root / "vocals.wav"),
                       "source_language": "pt-BR", "profile": "fast_lipsync"}
        self.patches = contextlib.ExitStack()
        self.patches.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.patches.close()
        self.temporary.cleanup()

    def fake_whisper(self, fast, quality=None, duration=80):
        constructed, references, arguments = [], [], []
        plans = {"large-v3-turbo": fast, "large-v3": quality if quality is not None else fast}
        class FakeWhisperModel:
            def __init__(self, model, **kwargs):
                self.model = model
                constructed.append(model)
                references.append(weakref.ref(self))
                arguments.append(kwargs)
            def transcribe(self, vocals, **kwargs):
                arguments.append({"vocals": vocals, **kwargs})
                result = plans[self.model]
                if isinstance(result, Exception):
                    raise result
                return iter(result), types.SimpleNamespace(duration=duration, language="pt", language_probability=0.99)
        module = types.ModuleType("faster_whisper")
        module.WhisperModel = FakeWhisperModel
        self.patches.enter_context(patch.dict(sys.modules, {"faster_whisper": module}))
        return constructed, references, arguments

    def execute_quality_child(self, references):
        def execute(command):
            self.assertTrue(all(reference() is None for reference in references), "Fast model or iterator still retains VRAM owner")
            self.assertEqual(command[1], str(Path(runner.__file__).resolve()))
            self.assertEqual(command[-4:-2], ["--stage", "asr"])
            request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
            self.assertEqual(request["profile"], "quality")
            self.assertEqual(request["vocals"], self.config["vocals"])
            self.assertNotEqual(request["output"], str(self.output))
            self.assertEqual(self.output.read_bytes(), self.previous)
            runner.asr_stage(request)
        return execute

    def test_one_ordinary_low_probability_word_in_strong_block_does_not_retry(self):
        ordinary = [segment(0, 2, "vejo a pessoa", [word("vejo", 0, 0.9), word("a", 0.9, 1, 0.25), word("pessoa", 1, 2)])]
        self.fake_whisper(ordinary)
        with patch.object(runner, "run_command", side_effect=AssertionError("An ordinary word must not trigger retry")):
            runner.asr_stage(self.config)
        self.assertEqual(len(json.loads(self.output.read_text())["segments"][0]["words"]), 3)

    def test_exit_zero_without_new_output_preserves_old_and_blocks(self):
        self.fake_whisper(isolated_garbage())
        with patch.object(runner, "run_command", return_value=None) as child:
            with self.assertRaisesRegex(runner.PipelineError, "sem produzir uma nova transcrição aprovada"):
                runner.asr_stage(self.config)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_isolated_garbage_is_blocked_even_when_whole_clip_average_exceeds_point_97(self):
        self.fake_whisper(isolated_garbage())
        with self.assertRaisesRegex(runner.PipelineError, "trecho 20"), patch.object(runner, "run_command", side_effect=RuntimeError("quality unavailable")):
            runner.asr_stage(self.config)
        observed = json.loads((self.root / "asr-fast-candidate.json").read_text())
        self.assertGreater(observed["word_probability"], 0.97)
        self.assertFalse(observed["approved"])
        self.assertTrue(any("trecho 20" in reason and "confiança ponderada" in reason for reason in observed["uncertainties"]))
        self.assertTrue(any("palavra 2" in reason and "tempos de palavra" in reason for reason in observed["uncertainties"]))
        self.assertEqual(observed["segments"][-1]["text"], "Sofallas P textures")
        self.assertEqual([item["word"] for item in observed["segments"][-1]["words"]], ["Sofallas", "P", "textures"])
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_one_full_clip_quality_retry_after_fast_model_release_then_atomic_promotion(self):
        names, references, arguments = self.fake_whisper(isolated_garbage(), good_segments())
        with patch.object(runner, "run_command", side_effect=self.execute_quality_child(references)) as child:
            runner.asr_stage(self.config)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(names, ["large-v3-turbo", "large-v3"])
        self.assertTrue(all(reference() is None for reference in references))
        approved = json.loads(self.output.read_text())
        self.assertEqual(approved["model"], "large-v3")
        quality = json.loads((self.root / "asr-quality-candidate.json").read_text())
        self.assertTrue(quality["approved"])
        self.assertEqual(approved["segments"], quality["segments"])
        self.assertEqual(approved["asr_request_id"], quality["asr_request_id"])
        self.assertTrue(arguments[0]["device"] == "cuda" and arguments[1]["word_timestamps"])
        self.assertFalse(list(self.root.glob("asr-quality-approved-pending-*.json")))

    def test_quality_uncertainty_blocks_without_recursing_and_preserves_both_candidates_and_old_approval(self):
        names, references, _ = self.fake_whisper(isolated_garbage())
        with patch.object(runner, "run_command", side_effect=self.execute_quality_child(references)) as child:
            with self.assertRaisesRegex(runner.PipelineError, "revise os trechos indicados"):
                runner.asr_stage(self.config)
        self.assertEqual(child.call_count, 1)
        self.assertEqual(names, ["large-v3-turbo", "large-v3"])
        self.assertEqual(self.output.read_bytes(), self.previous)
        for name in ("asr-fast-candidate.json", "asr-quality-candidate.json"):
            self.assertEqual(len(json.loads((self.root / name).read_text())["segments"][-1]["words"]), 3)

    def test_successful_child_without_fresh_output_is_a_manual_review_block_without_previous_approval(self):
        self.fake_whisper(isolated_garbage())
        absent_output = self.root / "first-transcript.json"
        with patch.object(runner, "run_command", return_value=None) as child:
            with self.assertRaisesRegex(runner.PipelineError, "sem produzir uma nova transcrição aprovada.*revise os trechos"):
                runner.asr_stage({**self.config, "output": str(absent_output)})
        self.assertEqual(child.call_count, 1)
        self.assertFalse(absent_output.exists())
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_stale_quality_candidate_cannot_approve_fresh_retry_output(self):
        self.fake_whisper(isolated_garbage())
        def forged_child(command):
            request = json.loads(Path(command[-1]).read_text())
            observed = {"model": "large-v3", "segments": [], "language": "pt", "probability": 0.99,
                        "asr_request_id": request["asr_request_id"]}
            runner.write_json(Path(request["output"]), observed)
            runner.write_json(self.root / "asr-quality-candidate.json", {**observed, "approved": True, "duration": 80, "asr_request_id": "old-retry-id"})
        with patch.object(runner, "run_command", side_effect=forged_child):
            with self.assertRaisesRegex(runner.PipelineError, "não corresponde ao candidato large-v3 aprovado"):
                runner.asr_stage(self.config)
        self.assertEqual(self.output.read_bytes(), self.previous)
        self.assertEqual(len(list(self.root.glob("asr-quality-approved-pending-*.json"))), 1)

    def test_fresh_but_invalid_quality_output_is_independently_checked(self):
        self.fake_whisper(isolated_garbage())
        def forged_child(command):
            request = json.loads(Path(command[-1]).read_text())
            invalid = [{"id": 1, "start": 0, "end": 1, "text": "palavra", "words": [
                {"word": "palavra", "start": 0, "end": 0, "probability": 0.99}]}]
            observed = {"model": "large-v3", "segments": invalid, "language": "pt", "probability": 0.99,
                        "asr_request_id": request["asr_request_id"]}
            runner.write_json(Path(request["output"]), observed)
            runner.write_json(self.root / "asr-quality-candidate.json", {**observed, "approved": True, "duration": 80})
        with patch.object(runner, "run_command", side_effect=forged_child):
            with self.assertRaisesRegex(runner.PipelineError, "tempos ou confiança inválidos"):
                runner.asr_stage(self.config)
        self.assertEqual(self.output.read_bytes(), self.previous)

    def test_finite_positive_word_spans_are_required_independent_of_confidence(self):
        for start, end in ((0, 0), (-1, 1), (0, float("nan")), (float("inf"), 1), (True, 1), (0, "1"), (0, 10**1000)):
            with self.subTest(start=start, end=end):
                result = runner.assess_asr_transcript([{"id": 1, "start": 0, "end": 2, "text": "teste", "words": [
                    {"word": "teste", "start": start, "end": end, "probability": 0.999}]}], 80, 0.99)
                self.assertFalse(result["approved"])
                self.assertTrue(any("tempos de palavra" in issue for issue in result["uncertainties"]))

    def test_invalid_probabilities_never_disappear_from_the_quality_decision(self):
        for probability in (None, float("nan"), float("inf"), True, -0.1, 1.1):
            with self.subTest(probability=probability):
                result = runner.assess_asr_transcript([{"id": 1, "start": 0, "end": 2, "text": "teste", "words": [
                    {"word": "teste", "start": 0, "end": 1, "probability": probability},
                    {"word": "certo", "start": 1, "end": 2, "probability": 0.99}]}], 80, 0.99)
                self.assertFalse(result["approved"])
                self.assertTrue(any("confiança inválida" in issue for issue in result["uncertainties"]))

    def test_duration_weighting_does_not_reject_one_uncertain_word_in_an_otherwise_strong_block(self):
        result = runner.assess_asr_transcript([{"id": 1, "start": 0, "end": 2, "text": "nome frase", "words": [
            {"word": "nome", "start": 0, "end": 0.1, "probability": 0.05},
            {"word": "frase", "start": 0.1, "end": 2, "probability": 0.99}]}], 80, 0.99)
        self.assertTrue(result["approved"])
        self.assertLess(result["word_probability"], 0.6)
        self.assertGreater(result["duration_weighted_word_probability"], 0.9)

    def test_tiny_positive_segment_is_not_rounded_into_zero_duration(self):
        self.fake_whisper([segment(0.1001, 0.1002, "sim", [word("sim", 0.1001, 0.1002)])])
        with patch.object(runner, "run_command", side_effect=AssertionError("No retry needed")):
            runner.asr_stage({**self.config, "profile": "quality"})
        approved = json.loads(self.output.read_text())
        self.assertEqual(approved["segments"][0]["start"], 0.1001)
        self.assertGreater(approved["segments"][0]["end"], approved["segments"][0]["start"])

    def test_transcription_exception_preserves_candidate_and_previous_approved_output(self):
        self.fake_whisper(RuntimeError("synthetic decoding failure"))
        with patch.object(runner, "run_command", side_effect=AssertionError("No decoding retry policy requested")):
            with self.assertRaisesRegex(runner.PipelineError, "candidato disponível foi preservado"):
                runner.asr_stage(self.config)
        self.assertEqual(self.output.read_bytes(), self.previous)
        self.assertEqual(json.loads((self.root / "asr-fast-candidate.json").read_text())["approved"], False)


if __name__ == "__main__":
    unittest.main()
