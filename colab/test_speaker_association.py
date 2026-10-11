"""CPU evidence contracts; synthetic voices are not real diarization/ML approval."""
import contextlib
import copy
import io
import os
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker
import dubstudio_runner_v2 as runner


def turn(start, end, identity=None, text="Fala original"):
    item = {"start": start, "end": end, "text": text}
    if identity is not None:
        item["speaker"] = identity
    return item


def face(start=0, end=25, x=5):
    return {"track": {"frame": list(range(start, end)), "bbox": [[x, 5, x + 30, 55]] * (end - start)}}


@contextlib.contextmanager
def mocked_embedding_stage(labels=None):
    """No ML/GPU: exercise only control flow and cleanup with recorded calls."""
    class Wave:
        shape = (1, 16000 * 16)
        def __getitem__(self, slices):
            clip = types.SimpleNamespace(shape=(1, max(0, min(slices[1].stop, self.shape[1]) - slices[1].start)))
            clip.to = lambda device: clip
            return clip
    class Vector:
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def numpy(self): return self
        def reshape(self, *args): return self
        def __truediv__(self, other): return self
    encoder = types.SimpleNamespace(encode_batch=unittest.mock.Mock(side_effect=lambda clip: Vector()))
    fit = unittest.mock.Mock(side_effect=lambda matrix: list(labels) if labels is not None else list(range(len(matrix))))
    cluster = unittest.mock.Mock(return_value=types.SimpleNamespace(fit_predict=fit))
    download = lambda **kwargs: None
    hub = types.SimpleNamespace(hf_hub_download=download)
    empty = unittest.mock.Mock()
    modules = {
        "numpy": types.SimpleNamespace(linalg=types.SimpleNamespace(norm=lambda vector: 1), stack=lambda vectors: vectors),
        "torch": types.SimpleNamespace(inference_mode=contextlib.nullcontext, cuda=types.SimpleNamespace(empty_cache=empty)),
        "torchaudio": types.SimpleNamespace(load=lambda file: (Wave(), 16000)),
        "huggingface_hub": hub,
        "sklearn.cluster": types.SimpleNamespace(AgglomerativeClustering=cluster),
        "speechbrain.inference.speaker": types.SimpleNamespace(EncoderClassifier=types.SimpleNamespace(from_hparams=lambda **kwargs: encoder)),
    }
    with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, modules):
        yield types.SimpleNamespace(cluster=cluster, fit=fit, hub=hub, download=download, empty=empty, encoder=encoder)


class ExplicitSpeakerCountTests(unittest.TestCase):
    def insufficient(self, requested, source, valid_count):
        before = copy.deepcopy(source)
        with mocked_embedding_stage() as calls:
            with self.assertRaisesRegex(runner.PipelineError, rf"informou {requested} locutores.*somente {valid_count} trechos"):
                runner.diarize_stage({"vocals": "synthetic", "directory": ".", "speaker_count": requested}, source)
            calls.cluster.assert_not_called()
            calls.fit.assert_not_called()
            self.assertEqual(calls.encoder.encode_batch.call_count, valid_count)
            self.assertIs(calls.hub.hf_hub_download, calls.download)
            calls.empty.assert_called_once()
        self.assertEqual(source, before)

    def test_two_requested_with_one_embedding_never_becomes_one(self):
        self.insufficient("2", [turn(0, 1)], 1)

    def test_three_requested_inside_one_asr_segment_is_not_confirmed(self):
        self.insufficient("3", [turn(0, 3, text="Três vozes podem falar neste único segmento ASR.")], 1)

    def test_three_requested_with_only_two_embeddings_never_becomes_two(self):
        self.insufficient("3", [turn(0, 1), turn(2, 3)], 2)

    def test_short_segments_do_not_inflate_available_evidence_count(self):
        self.insufficient("3", [turn(0, 1), turn(1.05, 1.2), turn(1.25, 1.4), turn(2, 3)], 2)

    def test_maximum_requested_count_cannot_be_clamped_to_evidence(self):
        self.insufficient("12", [turn(0, 1), turn(2, 3)], 2)

    def test_no_valid_embeddings_reports_explicit_count_not_generic_success(self):
        self.insufficient("3", [turn(0, .2), turn(.3, .5), turn(.6, .8)], 0)

    def test_sufficient_embeddings_keep_requested_clustering_configuration(self):
        with mocked_embedding_stage() as calls:
            output = runner.diarize_stage({"vocals": "synthetic", "directory": ".", "speaker_count": "3"}, [turn(0, 1), turn(2, 3), turn(4, 5)])
            calls.cluster.assert_called_once_with(n_clusters=3, metric="cosine", linkage="average")
        self.assertEqual([item["speaker"] for item in output], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"])
        self.assertTrue(all("speaker_confidence" not in item for item in output))

    def test_auto_one_embedding_keeps_automatic_behavior_without_forcing_three(self):
        with mocked_embedding_stage() as calls:
            output = runner.diarize_stage({"vocals": "synthetic", "directory": ".", "speaker_count": "auto"}, [turn(0, 1)])
            calls.cluster.assert_not_called()
        self.assertEqual(output[0]["speaker"], "SPEAKER_00")
        self.assertNotIn("speaker_confidence", output[0])

    def test_invalid_explicit_count_rejected_before_model_imports(self):
        for requested in ("0", "13", "3.5", "invalid", False):
            with self.subTest(requested=requested), patch.dict(sys.modules, {"numpy": None}):
                with self.assertRaisesRegex(runner.PipelineError, "inteiro de 1 a 12"):
                    runner.diarize_stage({"speaker_count": requested}, [turn(0, 1)])

    def test_single_declared_speaker_skips_ml_and_clears_stale_confidence(self):
        source = {"language": "pt", "segments": [turn(0, 1, "SPEAKER_02", "Texto original")], "duration": 1}
        source["segments"][0].update(speaker_confidence=1.0, words=[{"word": "Texto", "start": 0, "end": .4}])
        before = copy.deepcopy(source)
        for requested in ("1", 1):
            with self.subTest(requested=requested), patch.object(runner, "read_json", return_value=source), \
                    patch.object(runner, "write_json") as written, patch.object(runner, "diarize_stage") as model, \
                    patch.object(Path, "mkdir"), patch.object(worker, "write_progress"):
                runner.stage_main("diarization", {"directory": ".", "transcript": "synthetic.json", "output": "synthetic-out.json", "speaker_count": requested})
            model.assert_not_called()
            result = written.call_args.args[1]
            item = result["segments"][0]
            self.assertEqual(item["speaker"], "SPEAKER_00")
            self.assertEqual(item["speaker_assignment"], "user_declared_count")
            self.assertNotIn("speaker_confidence", item)
            self.assertEqual(result["speaker_count_evidence"], {"method": "user_declared_count", "requested": 1, "acoustic_validation": False})
            for field in ("start", "end", "text", "words"):
                self.assertEqual(item[field], source["segments"][0][field])
            self.assertEqual(result["language"], "pt")
        self.assertEqual(source, before)

    def test_failed_count_does_not_write_stage_output_or_remove_checkpoints(self):
        here = Path(__file__).resolve().parent
        temporary_root = "\\\\?\\" + str(here) if os.name == "nt" else str(here)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            checkpoint = directory / "transcript.json"
            runner.write_json(checkpoint, {"language": "pt", "segments": [turn(0, 1)]})
            before = checkpoint.read_bytes()
            output = directory / "diarization.json"
            with mocked_embedding_stage(), patch.object(worker, "write_progress"):
                with self.assertRaisesRegex(runner.PipelineError, "checkpoints anteriores foram preservados"):
                    runner.stage_main("diarization", {"directory": str(directory), "transcript": str(checkpoint), "output": str(output), "vocals": "synthetic", "speaker_count": "3"})
            self.assertFalse(output.exists())
            self.assertEqual(checkpoint.read_bytes(), before)


class DiarizationContextTests(unittest.TestCase):
    def short(self, left_gap=.05, right_gap=.05):
        return [turn(0, 1), turn(1 + left_gap, 1.25 + left_gap), turn(1.25 + left_gap + right_gap, 2.5)]

    def reject(self, segments, mapping, pattern="Diarização incerta"):
        before = copy.deepcopy(segments)
        with self.assertRaisesRegex(runner.PipelineError, pattern):
            runner.assign_diarized_speakers(segments, mapping)
        self.assertEqual(segments, before)

    def test_unique_bilateral_same_voice_preserves_text_and_timing(self):
        source = self.short()
        source[1].update(words=[{"word": "Sim", "start": 1.05, "end": 1.3}], text="Sim")
        before = copy.deepcopy(source)
        output = runner.assign_diarized_speakers(source, {0: 2, 2: 2})
        self.assertEqual(source, before)
        self.assertEqual(output[1]["speaker"], "SPEAKER_02")
        self.assertEqual(output[1]["speaker_assignment"], "bilateral_same_speaker_context")
        for field in ("text", "words", "start", "end"):
            self.assertEqual(output[1][field], source[1][field])

    def test_different_neighbors_never_choose_nearest_start(self):
        source = [turn(0, 1), turn(1.05, 1.3), turn(2, 3), turn(4, 5)]
        source[1]["id"] = 2
        self.reject(source, {0: 0, 2: 1, 3: 2}, r"trecho 2.*1\.050.*1\.300.*vozes diferentes")

    def test_initial_short_turn_has_no_one_sided_inheritance(self):
        self.reject([turn(0, .25), turn(.3, 2)], {1: 0}, "dois turnos")

    def test_final_short_turn_has_no_one_sided_inheritance(self):
        self.reject([turn(0, 1), turn(1.05, 1.3)], {0: 0}, "dois turnos")

    def test_left_gap_exact_threshold_allowed(self):
        output = runner.assign_diarized_speakers(self.short(left_gap=.2), {0: 0, 2: 0})
        self.assertEqual(output[1]["speaker"], "SPEAKER_00")

    def test_right_gap_exact_threshold_allowed(self):
        output = runner.assign_diarized_speakers(self.short(right_gap=.2), {0: 1, 2: 1})
        self.assertEqual(output[1]["speaker"], "SPEAKER_01")

    def test_threshold_tiny_float_noise_is_stable(self):
        for noise in (-1e-8, 1e-8):
            output = runner.assign_diarized_speakers(self.short(.2 + noise, .2 + noise), {0: 1, 2: 1})
            self.assertEqual(output[1]["speaker"], "SPEAKER_01")

    def test_left_gap_above_threshold_rejected(self):
        self.reject(self.short(left_gap=.200002), {0: 0, 2: 0}, "intervalo maior")

    def test_right_gap_above_threshold_rejected(self):
        self.reject(self.short(right_gap=.200002), {0: 0, 2: 0}, "intervalo maior")

    def test_touching_neighbor_intervals_are_valid(self):
        output = runner.assign_diarized_speakers(self.short(0, 0), {0: 0, 2: 0})
        self.assertEqual(output[1]["speaker_assignment"], "bilateral_same_speaker_context")

    def test_list_order_is_not_temporal_evidence(self):
        source = self.short()
        output = runner.assign_diarized_speakers([source[2], source[1], source[0]], {0: 1, 2: 1})
        self.assertEqual(output[1]["speaker"], "SPEAKER_01")

    def test_intervening_unembedded_turn_cannot_be_skipped(self):
        source = [turn(0, 1), turn(1.01, 1.1), turn(1.11, 1.2), turn(1.21, 2.5)]
        self.reject(source, {0: 0, 3: 0}, "vizinho também")

    def test_tied_previous_candidates_are_ambiguous(self):
        source = [turn(0, 1), turn(.1, 1), turn(1.05, 1.3), turn(1.35, 2.5)]
        self.reject(source, {0: 0, 1: 0, 3: 0}, "mais de um turno")

    def test_tied_next_candidates_are_ambiguous(self):
        source = [turn(0, 1), turn(1.05, 1.3), turn(1.35, 2.5), turn(1.35, 2.7)]
        self.reject(source, {0: 0, 2: 0, 3: 0}, "mais de um turno")

    def test_short_turn_overlapping_other_voice_is_ambiguous(self):
        source = [turn(0, 1.1), turn(1.05, 1.3), turn(1.35, 2.5)]
        self.reject(source, {0: 0, 2: 0}, "sobreposta")

    def test_language_and_script_do_not_change_association(self):
        for text in ("Sí", "Tak", "はい", "نعم", "Sim", "Да"):
            source = self.short()
            source[1]["text"] = text
            output = runner.assign_diarized_speakers(source, {0: 1, 2: 1})
            self.assertEqual(output[1]["text"], text)
            self.assertEqual(output[1]["speaker"], "SPEAKER_01")

    def test_invalid_times_never_confirm_voice(self):
        for value in (float("nan"), float("inf"), -.1, "invalid", None):
            source = self.short()
            source[1]["start"] = value
            before = copy.deepcopy(source)
            with self.assertRaisesRegex(runner.PipelineError, "tempos inválidos"):
                runner.assign_diarized_speakers(source, {0: 0, 2: 0})
            self.assertEqual(source[0], before[0])
            self.assertNotIn("speaker", source[1])

    def test_invalid_embedding_mapping_never_creates_voice(self):
        for mapping in ({}, {0: True}, {0: -1}, {0: 12}, {5: 0}, {"0": 0}):
            self.reject(self.short(), mapping, "evidência válida")

    def test_fully_embedded_three_voices_keep_their_own_identity(self):
        source = [turn(0, 1), turn(1.2, 2.2), turn(2.4, 3.4)]
        output = runner.assign_diarized_speakers(source, {0: 2, 1: 0, 2: 1})
        self.assertEqual([item["speaker"] for item in output], ["SPEAKER_02", "SPEAKER_00", "SPEAKER_01"])
        self.assertTrue(all(item["speaker_assignment"] == "ecapa_clustering" for item in output))

    def test_review_error_does_not_partially_rewrite_existing_checkpoint(self):
        source = self.short()
        source[0]["speaker"] = "PREVIOUS_REVIEW"
        self.reject(source, {0: 0, 2: 1})


class FacialVoiceEvidenceTests(unittest.TestCase):
    def associate(self, transcript, tracks=None, selection="auto", operation="dub_video", scores=None):
        tracks = tracks if tracks is not None else [face()]
        scores = scores if scores is not None else [[1] * len(item["track"]["frame"]) for item in tracks]
        return runner.identity_segments(tracks, scores, 100, 160, 100, transcript, selection, operation)

    def test_positive_face_outside_transcript_never_defaults_speaker00(self):
        groups, ambiguity = self.associate([turn(2, 3, "SPEAKER_02")])
        self.assertEqual(groups, [])
        self.assertIn("sem fala transcrita", ambiguity[0]["reason"])

    def test_positive_face_without_transcript_requires_review(self):
        groups, ambiguity = self.associate([])
        self.assertEqual(groups, [])
        self.assertIn("sem fala transcrita", ambiguity[0]["reason"])

    def test_face_selection_cannot_invent_original_voice(self):
        groups, ambiguity = self.associate([], selection={"track_id": 0})
        self.assertEqual(groups, [])
        self.assertIn("sem fala transcrita", ambiguity[0]["reason"])

    def test_transcribed_speech_without_speaker_never_defaults(self):
        groups, ambiguity = self.associate([turn(0, 1)])
        self.assertEqual(groups, [])
        self.assertIn("sem identidade vocal", ambiguity[0]["reason"])

    def test_invalid_speaker_types_never_confirm(self):
        for speaker in (None, "", " ", False, {}, 0):
            groups, ambiguity = self.associate([turn(0, 1, speaker)])
            self.assertEqual(groups, [])
            self.assertTrue(ambiguity)

    def test_one_unassigned_turn_cannot_hide_behind_assigned_turn(self):
        groups, ambiguity = self.associate([turn(0, .5, "SPEAKER_01"), turn(.5, 1)])
        self.assertEqual(groups, [])
        self.assertTrue(ambiguity)

    def test_external_audio_proves_geometry_not_original_speaker(self):
        groups, ambiguity = self.associate([], selection={"track_id": 0}, operation="lipsync_video", scores=[[-1] * 25])
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["speaker"], "USER_AUDIO")
        self.assertEqual(groups[0]["speaker_confirmation"], "external_audio")

    def test_external_audio_does_not_inherit_original_transcript_voice(self):
        groups, ambiguity = self.associate([turn(0, 1, "SPEAKER_02")], operation="lipsync_video")
        self.assertFalse(ambiguity)
        self.assertEqual(groups[0]["speaker"], "USER_AUDIO")

    def test_genuine_narration_no_detected_face_is_not_ambiguous(self):
        groups, ambiguity = self.associate([turn(0, 1, "SPEAKER_02")], tracks=[])
        self.assertEqual(groups, [])
        self.assertFalse(ambiguity)

    def test_nonpositive_face_is_not_a_fabricated_voice(self):
        groups, ambiguity = self.associate([turn(0, 1, "SPEAKER_02")], scores=[[0] * 25])
        self.assertEqual(groups, [])
        self.assertFalse(ambiguity)

    def test_three_turns_bind_three_faces_to_the_correct_voice(self):
        tracks = [face(index * 25, (index + 1) * 25, x=index * 40) for index in range(3)]
        transcript = [turn(index, index + 1, f"SPEAKER_{index:02d}") for index in range(3)]
        groups, ambiguity = self.associate(transcript, tracks)
        self.assertFalse(ambiguity)
        self.assertEqual([(item["track_id"], item["speaker"]) for item in groups], [(index, f"SPEAKER_{index:02d}") for index in range(3)])

    def test_overlapping_speaker_turns_require_review(self):
        groups, ambiguity = self.associate([turn(0, 1, "SPEAKER_00"), turn(.2, .8, "SPEAKER_01")])
        self.assertEqual(groups, [])
        self.assertIn("mais de um locutor", ambiguity[0]["reason"])

    def test_three_positive_faces_do_not_silently_choose_first(self):
        groups, ambiguity = self.associate([turn(0, 1, "SPEAKER_00")], [face(x=index * 40) for index in range(3)])
        self.assertEqual(groups, [])
        self.assertEqual(len(ambiguity), 25)


class DiarizationStageAndReferenceTests(unittest.TestCase):
    def test_real_stage_uses_context_guard_and_restores_client_on_error(self):
        class Wave:
            shape = (1, 16000 * 5)
            def __getitem__(self, slices):
                clip = types.SimpleNamespace(shape=(1, slices[1].stop - slices[1].start))
                clip.to = lambda device: clip
                return clip
        class Vector:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return self
            def reshape(self, *args): return self
            def __truediv__(self, other): return self
        encoder = types.SimpleNamespace(encode_batch=lambda clip: Vector())
        download = lambda **kwargs: None
        hub = types.SimpleNamespace(hf_hub_download=download)
        empty = unittest.mock.Mock()
        modules = {
            "numpy": types.SimpleNamespace(linalg=types.SimpleNamespace(norm=lambda vector: 1), stack=lambda vectors: vectors),
            "torch": types.SimpleNamespace(inference_mode=contextlib.nullcontext, cuda=types.SimpleNamespace(empty_cache=empty)),
            "torchaudio": types.SimpleNamespace(load=lambda file: (Wave(), 16000)),
            "huggingface_hub": hub,
            "sklearn.cluster": types.SimpleNamespace(AgglomerativeClustering=lambda **kwargs: types.SimpleNamespace(fit_predict=lambda matrix: [0, 1, 2])),
            "speechbrain.inference.speaker": types.SimpleNamespace(EncoderClassifier=types.SimpleNamespace(from_hparams=lambda **kwargs: encoder)),
        }
        source = [turn(0, 1), turn(1.05, 1.3), turn(2, 3), turn(4, 5)]
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(runner.PipelineError, "vozes diferentes"):
                runner.diarize_stage({"vocals": "synthetic", "directory": ".", "speaker_count": "3"}, source)
        self.assertIs(hub.hf_hub_download, download)
        empty.assert_called_once()
        self.assertTrue(all("speaker" not in item for item in source))

    def test_three_references_retain_their_distinct_assigned_voice_audio(self):
        here = Path(__file__).resolve().parent
        temporary_root = "\\\\?\\" + str(here) if os.name == "nt" else str(here)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            directory = Path(temporary)
            source = directory / "synthetic-three-voices.wav"
            with wave.open(str(source), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(24000)
                for amplitude in (2000, 4000, 6000):
                    audio.writeframes(struct.pack("<h", amplitude) * 48000)
            transcript = [turn(index * 2, (index + 1) * 2, f"SPEAKER_{index:02d}") for index in range(3)]
            with contextlib.redirect_stdout(io.StringIO()):
                references = worker.voice_references(source, transcript, directory)
            self.assertEqual(set(references), {f"SPEAKER_{index:02d}" for index in range(3)})
            self.assertEqual(len({runner.sha256(path) for path in references.values()}), 3)
            for index in range(3):
                with wave.open(str(references[f"SPEAKER_{index:02d}"]), "rb") as audio:
                    decoded = struct.unpack("<h", audio.readframes(1))[0]
                    self.assertEqual(decoded, 2000 + index * 2000)


if __name__ == "__main__":
    unittest.main()
