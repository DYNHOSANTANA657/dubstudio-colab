"""CPU-only synthesis-cache contract tests: synthetic WAVs and explicit model stubs."""
from __future__ import annotations

import copy
import contextlib
import hashlib
import json
import io
import math
import os
import struct
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
import dubstudio_runner_v2 as runner


def wav(file: Path, seconds=0.1, rate=48000, amplitude=8000):
    with wave.open(str(file), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(struct.pack("<h", amplitude) * round(seconds * rate))


class CacheFixture(unittest.TestCase):
    def setUp(self):
        temp_root = "\\\\?\\" + str(HERE) if os.name == "nt" else str(HERE)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.reference = self.directory / "ref.wav"
        wav(self.reference, 1)
        self.model = {"assets": {"ve.pt": "a" * 64, "t3.safetensors": "b" * 64}, "revision": "c" * 40, "generation_defaults": {"temperature": 0.8}}
        self.item = {"id": 1, "start": 0.0, "end": 1.0, "text": "Olá.", "translated_text": "Hola.", "speaker": "SPEAKER_00"}
        self.cache = runner.RawVoiceCache(self.directory)
        self.generate = mock.Mock(side_effect=lambda file: wav(file))

    def identity(self, item=None, model=None, language="es"):
        return runner.raw_voice_identity(item or self.item, language, self.reference, model or self.model)

    def populate(self):
        key, identity = self.identity()
        return key, identity, self.cache.get_or_generate(key, identity, self.generate)


class RawCacheTests(CacheFixture):
    def test_complete_provenance_reuses_without_generation(self):
        key, identity, (file, metadata, cached) = self.populate()
        self.assertFalse(cached)
        self.assertTrue(self.cache.get_or_generate(key, identity, self.generate)[2])
        self.assertEqual(self.generate.call_count, 1)
        self.assertEqual(metadata["sha256"], runner.sha256(file))

    def test_legacy_ordinal_raw_is_never_adopted(self):
        wav(self.directory / "voice-0000.wav")
        self.populate()
        self.assertEqual(self.generate.call_count, 1)

    def test_missing_or_malformed_marker_regenerates(self):
        key, identity, _ = self.populate()
        marker = self.cache.directory / f"{key}.json"
        for bad in (None, [], {"identity_key": key}, "not-json"):
            if bad is None:
                marker.unlink()
            elif bad == "not-json":
                runner.write_json(marker, bad)
            else:
                runner.write_json(marker, bad)
            self.assertFalse(self.cache.get_or_generate(key, identity, self.generate)[2])
        self.assertEqual(self.generate.call_count, 5)

    def test_corrupt_raw_hash_regenerates_only_that_file(self):
        key, identity, (file, _, _) = self.populate()
        wav(file, amplitude=4000)
        self.assertFalse(self.cache.get_or_generate(key, identity, self.generate)[2])
        self.assertEqual(self.generate.call_count, 2)

    def test_text_voice_language_and_actual_weight_changes_invalidate(self):
        key, _, _ = self.populate()
        variations = [dict(self.item, translated_text="Buenos días."), dict(self.item, speaker="SPEAKER_01")]
        for item in variations:
            self.assertNotEqual(self.identity(item)[0], key)
        self.assertNotEqual(self.identity(language="pl")[0], key)
        model = copy.deepcopy(self.model)
        model["assets"]["ve.pt"] = "f" * 64
        self.assertNotEqual(self.identity(model=model)[0], key)
        wav(self.reference, amplitude=4000)
        self.assertNotEqual(self.identity()[0], key)

    def test_timing_scene_ordinal_and_fitting_are_not_raw_identity(self):
        key, _, _ = self.populate()
        changed = dict(self.item, id=99, start=13, end=14.5, scene=7, speed=1.2)
        self.assertEqual(self.identity(changed)[0], key)

    def test_generation_parameters_and_source_config_changes_invalidate(self):
        key = self.identity()[0]
        for field, value in (("generation_defaults", {"temperature": 0.7}), ("installed_sources", {"model.py": "d" * 64}), ("revision", "d" * 40)):
            changed = dict(self.model, **{field: value})
            self.assertNotEqual(self.identity(model=changed)[0], key)

    def test_failed_generation_has_no_completion_marker(self):
        key, identity = self.identity()
        def fail(file):
            wav(file)
            raise RuntimeError("simulated OOM")
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            self.cache.get_or_generate(key, identity, fail)
        self.assertFalse((self.cache.directory / f"{key}.json").exists())
        self.assertFalse((self.cache.directory / f"{key}.wav").exists())

    def test_marker_write_failure_does_not_validate_new_audio(self):
        key, identity = self.identity()
        with mock.patch.object(runner, "write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.cache.get_or_generate(key, identity, self.generate)
        self.assertFalse((self.cache.directory / f"{key}.json").exists())
        self.assertFalse(self.cache.get_or_generate(key, identity, self.generate)[2])

    def test_silent_or_truncated_wav_is_rejected_without_marker(self):
        key, identity = self.identity()
        with self.assertRaises(runner.PipelineError):
            self.cache.get_or_generate(key, identity, lambda file: wav(file, amplitude=0))
        raw = self.directory / "truncated.wav"
        wav(raw)
        raw_bytes = raw.read_bytes()
        with raw.open("wb") as output:
            output.write(raw_bytes[:-100])
        with self.assertRaises(runner.PipelineError):
            runner.raw_voice_metadata(raw)
        self.assertFalse((self.cache.directory / f"{key}.json").exists())

    def test_invalid_cache_key_rejected_before_path_use(self):
        with self.assertRaises(runner.PipelineError):
            self.cache.get_or_generate("../other", {}, self.generate)
        self.generate.assert_not_called()


class AcceptanceRetryTests(CacheFixture):
    def test_rejected_first_candidate_has_no_marker_then_second_is_accepted(self):
        key, identity = self.identity()
        calls = []
        def generate(file):
            seconds = 5.3 if not calls else 4.3
            calls.append(seconds)
            self.assertFalse((self.cache.directory / f"{key}.json").exists())
            wav(file, seconds)
        with contextlib.redirect_stdout(io.StringIO()):
            file, metadata, reused = self.cache.get_or_generate(key, identity, generate, accept=lambda file, metadata: metadata["duration"] <= 5, attempts=2)
        self.assertEqual(calls, [5.3, 4.3])
        self.assertFalse(reused)
        self.assertEqual(metadata["duration"], 4.3)
        self.assertEqual(len(list(self.cache.directory.glob("*.json"))), 1)
        self.assertEqual(len(list(self.cache.directory.glob("*.rejected.wav"))), 1)
        self.assertEqual(runner.raw_voice_metadata(file)["duration"], 4.3)

    def test_two_long_candidates_block_without_marker_or_third_attempt(self):
        key, identity = self.identity()
        generate = mock.Mock(side_effect=lambda file: wav(file, 5.3))
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(runner.PipelineError, "duas tentativas"):
            self.cache.get_or_generate(key, identity, generate, accept=lambda file, metadata: False, attempts=2)
        self.assertEqual(generate.call_count, 2)
        self.assertFalse((self.cache.directory / f"{key}.json").exists())
        self.assertEqual(len(list(self.cache.directory.glob("*.rejected.wav"))), 2)

    def test_prior_raw_and_marker_preserved_if_new_window_rejects_both_candidates(self):
        key, identity, (raw, _, _) = self.populate()
        marker = self.cache.directory / f"{key}.json"
        old_raw, old_marker = raw.read_bytes(), marker.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(runner.PipelineError):
            self.cache.get_or_generate(key, identity, lambda file: wav(file, 0.2), accept=lambda file, metadata: False, attempts=2)
        self.assertEqual(raw.read_bytes(), old_raw)
        self.assertEqual(marker.read_bytes(), old_marker)

    def test_gate_rehashes_pending_after_silence_only_transform(self):
        key, identity = self.identity()
        def accept(file, metadata):
            wav(file, 0.08)
            return True
        raw, metadata, _ = self.cache.get_or_generate(key, identity, self.generate, accept=accept)
        self.assertEqual(metadata["duration"], 0.08)
        self.assertEqual(metadata, runner.raw_voice_metadata(raw))
        self.assertEqual(runner.read_json(self.cache.directory / f"{key}.json")["audio"], metadata)

    def test_replaced_cached_candidate_archives_previous_audio(self):
        key, identity, (raw, metadata, _) = self.populate()
        old_sha = metadata["sha256"]
        self.cache.get_or_generate(key, identity, lambda file: wav(file, 0.2), accept=lambda file, data: data["duration"] > 0.15)
        previous = self.cache.directory / f"{key}.{old_sha}.previous.wav"
        self.assertEqual(runner.sha256(previous), old_sha)
        self.assertNotEqual(runner.sha256(raw), old_sha)


class PreflightTests(CacheFixture):
    def test_adjacent_duplicate_microsegment_requires_revision_without_removal(self):
        segments = [self.item, dict(self.item, id=2, start=1.0, end=1.02)]
        before = copy.deepcopy(segments)
        with self.assertRaisesRegex(runner.PipelineError, "nenhuma fala foi removida"):
            runner.synthesis_preflight(segments, {"SPEAKER_00": self.reference}, 2)
        self.assertEqual(segments, before)

    def test_duplicate_source_text_blocks_even_when_translation_differs(self):
        segments = [self.item, dict(self.item, id=2, start=1.0, end=1.16, translated_text="¡Hola!")]
        with self.assertRaisesRegex(runner.PipelineError, "repete"):
            runner.synthesis_preflight(segments, {"SPEAKER_00": self.reference}, 2)

    def test_genuine_short_speech_and_other_speaker_are_not_removed(self):
        segments = [dict(self.item, end=0.15, text="Sim", translated_text="Sí"), dict(self.item, id=2, start=0.15, end=0.3, text="Não", translated_text="No")]
        runner.synthesis_preflight(segments, {"SPEAKER_00": self.reference}, 2)
        segments[1].update(text="Sim", translated_text="Sí", speaker="SPEAKER_01")
        runner.synthesis_preflight(segments, {"SPEAKER_00": self.reference, "SPEAKER_01": self.reference}, 2)

    def test_repeated_short_confirmation_after_real_pause_is_not_false_duplicate(self):
        segments = [dict(self.item, start=0, end=0.15, text="Sim", translated_text="Sí"), dict(self.item, id=2, start=10, end=10.15, text="Sim", translated_text="Sí")]
        runner.synthesis_preflight(segments, {"SPEAKER_00": self.reference}, 11)

    def test_text_to_voice_zero_timestamps_are_valid(self):
        runner.synthesis_preflight([dict(self.item, start=0, end=0)], {"SPEAKER_00": self.reference}, 0)

    def test_empty_text_zero_nonfinite_and_bad_windows_block(self):
        for item in (dict(self.item, translated_text="..."), dict(self.item, end=0), dict(self.item, start=float("nan")), dict(self.item, end=3)):
            with self.subTest(item=item), self.assertRaises(runner.PipelineError):
                runner.synthesis_preflight([item], {"SPEAKER_00": self.reference}, 2)

    def test_missing_or_unknown_reference_blocks_before_heavy_imports(self):
        file = self.directory / "segments.json"
        runner.write_json(file, {"segments": [dict(self.item, speaker="OTHER")]})
        config = {"directory": str(self.directory), "segments": str(file), "language": "es", "duration": 2, "references": {"SPEAKER_00": str(self.reference)}}
        with mock.patch.object(runner, "prepare_chatterbox_model") as factory:
            with self.assertRaisesRegex(runner.PipelineError, "Locutor OTHER"):
                runner.synthesize_stage(config, None)
            factory.assert_not_called()

    def test_microsegments_fail_before_model_or_gpu_import(self):
        segments = [self.item, dict(self.item, id=2, start=1, end=1.02)]
        file = self.directory / "segments.json"
        runner.write_json(file, {"segments": segments})
        config = {"directory": str(self.directory), "segments": str(file), "language": "es", "duration": 2, "references": {"SPEAKER_00": str(self.reference)}}
        with mock.patch.object(runner, "prepare_chatterbox_model") as factory:
            with self.assertRaisesRegex(runner.PipelineError, "repete"):
                runner.synthesize_stage(config, None)
            factory.assert_not_called()


class ModelProvenanceTests(CacheFixture):
    def fake_module(self):
        module = types.ModuleType("fake_chatterbox")
        source = self.directory / "mtl_tts.py"
        source.write_text("# synthetic installed code, not real model\n", encoding="utf-8")
        module.__file__ = str(source)
        module.REPO_ID = "ResembleAI/chatterbox"
        class Engine:
            def generate(self, text, language_id, audio_prompt_path=None, exaggeration=0.5, temperature=0.8):
                pass
        module.ChatterboxMultilingualTTS = Engine
        snapshot = self.directory / "snapshots" / ("c" * 40)
        snapshot.mkdir(parents=True)
        assets = ["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"]
        for name in assets[:-1]:
            (snapshot / name).write_bytes(name.encode())
        return module, snapshot, assets

    def test_weights_sources_defaults_and_optional_absence_are_fingerprinted(self):
        module, snapshot, assets = self.fake_module()
        with mock.patch.object(runner.importlib.metadata, "version", side_effect=lambda name: "0.1.7" if name == "chatterbox-tts" else "fixture-1"):
            before = runner.chatterbox_model_fingerprint(module, snapshot, assets)
            self.assertEqual(before["assets"]["ve.pt"], runner.sha256(snapshot / "ve.pt"))
            self.assertIsNone(before["assets"]["conds.pt"])
            self.assertEqual(before["generation_defaults"]["temperature"], 0.8)
            (snapshot / "conds.pt").write_bytes(b"optional-conditionals")
            self.assertNotEqual(before, runner.chatterbox_model_fingerprint(module, snapshot, assets))

    def test_missing_empty_or_unpinned_model_assets_fail_closed(self):
        module, snapshot, assets = self.fake_module()
        with mock.patch.object(runner.importlib.metadata, "version", return_value="0.1.7"):
            (snapshot / "ve.pt").write_bytes(b"")
            with self.assertRaisesRegex(runner.PipelineError, "ausente ou vazio"):
                runner.chatterbox_model_fingerprint(module, snapshot, assets)
            with self.assertRaisesRegex(runner.PipelineError, "imutável"):
                runner.chatterbox_model_fingerprint(module, self.directory, assets)

    def test_loader_asset_contract_comes_from_installed_source(self):
        module, _, _ = self.fake_module()
        source = '    def from_pretrained(cls, device):\n        return snapshot_download(repo_id=REPO_ID, allow_patterns=["ve.pt", "t3_23lang.safetensors", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json", "conds.pt"])\n'
        module.ChatterboxMultilingualTTS.from_pretrained = lambda: None
        with mock.patch.object(runner.inspect, "getsource", return_value=source):
            assets = runner.chatterbox_asset_names(module)
            self.assertIn("t3_23lang.safetensors", assets)

    def test_loader_dynamic_unknown_asset_contract_is_rejected(self):
        module, _, _ = self.fake_module()
        module.ChatterboxMultilingualTTS.from_pretrained = lambda: None
        for source in ('def from_pretrained(cls, device):\n return snapshot_download(allow_patterns=unknown)\n', 'def from_pretrained(cls, device):\n return snapshot_download(allow_patterns=["../private"])\n'):
            with mock.patch.object(runner.inspect, "getsource", return_value=source), self.assertRaises(runner.PipelineError):
                runner.chatterbox_asset_names(module)


class LazyModelProvenanceTests(ModelProvenanceTests):
    """No real dependencies/network/GPU: verify the installed-loader adapter contract."""
    # Inherit fixture helpers, not the other case's test methods.
    test_weights_sources_defaults_and_optional_absence_are_fingerprinted = None
    test_missing_empty_or_unpinned_model_assets_fail_closed = None
    test_loader_asset_contract_comes_from_installed_source = None
    test_loader_dynamic_unknown_asset_contract_is_rejected = None

    def provider_fixture(self):
        module, snapshot, assets = self.fake_module()
        assets.append("Cangjie5_TC.json")
        (snapshot / "Cangjie5_TC.json").write_bytes(b"synthetic-tokenizer-config")
        module.ChatterboxMultilingualTTS.from_pretrained = lambda: None
        module.ChatterboxMultilingualTTS.from_local = mock.Mock(return_value=object())
        module.MTLTokenizer = type("MTLTokenizer", (), {})
        tokenizer_module = types.SimpleNamespace(hf_hub_download=mock.Mock(side_effect=AssertionError("network not allowed")))
        hub = types.SimpleNamespace(snapshot_download=mock.Mock(return_value=str(snapshot)))
        package = types.ModuleType("chatterbox")
        package.__path__ = []
        package.mtl_tts = module
        source = 'def from_pretrained(cls, device):\n return snapshot_download(allow_patterns=' + repr(assets) + ')\n'
        patches = contextlib.ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(mock.patch.dict(sys.modules, {"chatterbox": package, "chatterbox.mtl_tts": module, "huggingface_hub": hub}))
        patches.enter_context(mock.patch.object(runner.inspect, "getsource", return_value=source))
        patches.enter_context(mock.patch.object(runner.inspect, "getmodule", return_value=tokenizer_module))
        patches.enter_context(mock.patch.object(runner.importlib.metadata, "version", side_effect=lambda name: "0.1.7" if name == "chatterbox-tts" else "fixture-1"))
        return module, snapshot, hub, tokenizer_module

    def test_provider_hashes_disk_without_loading_model(self):
        module, snapshot, hub, _ = self.provider_fixture()
        fingerprint, load = runner.prepare_chatterbox_model()
        self.assertEqual(fingerprint["revision"], snapshot.name)
        module.ChatterboxMultilingualTTS.from_local.assert_not_called()
        self.assertTrue(hub.snapshot_download.call_args.kwargs["local_files_only"])
        load()
        module.ChatterboxMultilingualTTS.from_local.assert_called_once_with(snapshot, device="cuda")

    def test_partial_snapshot_resolves_missing_assets_only_at_same_revision(self):
        module, snapshot, hub, _ = self.provider_fixture()
        (snapshot / "ve.pt").unlink()
        def resolve(**kwargs):
            if kwargs.get("revision") == snapshot.name:
                (snapshot / "ve.pt").write_bytes(b"resolved-at-pinned-revision")
            return str(snapshot)
        hub.snapshot_download.side_effect = resolve
        runner.prepare_chatterbox_model()
        self.assertEqual(hub.snapshot_download.call_count, 2)
        self.assertEqual(hub.snapshot_download.call_args.kwargs["revision"], snapshot.name)
        module.ChatterboxMultilingualTTS.from_local.assert_not_called()

    def test_asset_mutation_before_lazy_load_blocks_without_gpu(self):
        module, snapshot, _, _ = self.provider_fixture()
        _, load = runner.prepare_chatterbox_model()
        (snapshot / "ve.pt").write_bytes(b"mutated")
        with self.assertRaisesRegex(runner.PipelineError, "mudaram"):
            load()
        module.ChatterboxMultilingualTTS.from_local.assert_not_called()

    def test_nested_tokenizer_download_pinned_and_restored(self):
        module, snapshot, _, tokenizer_module = self.provider_fixture()
        original = tokenizer_module.hf_hub_download
        def local_load(*args, **kwargs):
            self.assertEqual(tokenizer_module.hf_hub_download(repo_id="ResembleAI/chatterbox", filename="Cangjie5_TC.json", cache_dir="nested"), str(snapshot / "Cangjie5_TC.json"))
            return object()
        module.ChatterboxMultilingualTTS.from_local.side_effect = local_load
        _, load = runner.prepare_chatterbox_model()
        load()
        self.assertIs(tokenizer_module.hf_hub_download, original)
        original.assert_not_called()

    def test_unrecorded_tokenizer_download_cannot_be_silently_swallowed(self):
        module, _, _, tokenizer_module = self.provider_fixture()
        original = tokenizer_module.hf_hub_download
        def swallows_exception(*args, **kwargs):
            try:
                tokenizer_module.hf_hub_download(repo_id="unapproved/model", filename="unknown.json")
            except runner.PipelineError:
                pass
            return object()
        module.ChatterboxMultilingualTTS.from_local.side_effect = swallows_exception
        _, load = runner.prepare_chatterbox_model()
        with self.assertRaisesRegex(runner.PipelineError, "externa"):
            load()
        self.assertIs(tokenizer_module.hf_hub_download, original)


class FakeClip:
    def __init__(self, frames, rate=48000):
        self.frames, self.rate = frames, rate
    @classmethod
    def silent(cls, duration, frame_rate):
        return cls(round(duration * frame_rate / 1000), frame_rate)
    @classmethod
    def from_wav(cls, file):
        with wave.open(str(file), "rb") as audio:
            return cls(audio.getnframes(), audio.getframerate())
    def set_channels(self, channels):
        return self
    def set_frame_rate(self, rate):
        return FakeClip(round(self.frames * rate / self.rate), rate)
    def frame_count(self):
        return self.frames
    def overlay(self, clip, position):
        if round(position * self.rate / 1000) + clip.frames > self.frames:
            raise AssertionError("Stub detected a truncated speech ending")
        return self
    def __add__(self, clip):
        return FakeClip(self.frames + clip.frames, self.rate)
    def __len__(self):
        return round(self.frames * 1000 / self.rate)
    def export(self, file, format):
        wav(file, self.frames / self.rate, self.rate)


class SynthIntegrationTests(CacheFixture):
    def setUp(self):
        super().setUp()
        self.calls, self.loads, self.fits = [], [], []
        self.fail_index = None
        self.segments = [dict(self.item, id=index + 1, start=index, end=index + 0.9, translated_text=f"Oración número {index + 1}.") for index in range(17)]
        self.segments_file = self.directory / "segments.json"
        self.config = {"directory": str(self.directory), "segments": str(self.segments_file), "references": {"SPEAKER_00": str(self.reference)}, "duration": 17, "language": "es-ES"}
        self.modules = {"numpy": types.SimpleNamespace(isfinite=lambda samples: types.SimpleNamespace(all=lambda: True), square=lambda x: 1, mean=lambda x: 0.04, sqrt=math.sqrt),
                        "soundfile": types.SimpleNamespace(write=lambda file, samples, rate, subtype: wav(file, 0.1, rate)),
                        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None)),
                        "pydub": types.SimpleNamespace(AudioSegment=FakeClip)}
        outer = self
        class Samples:
            size = 4800
            def squeeze(self):
                return self
        class Utterance:
            def detach(self): return self
            def float(self): return self
            def cpu(self): return self
            def numpy(self): return Samples()
        class Engine:
            sr = 48000
            def generate(self, text, language_id, audio_prompt_path):
                outer.calls.append(text)
                if outer.fail_index and text == f"Oración número {outer.fail_index}.":
                    raise RuntimeError("simulated segment OOM")
                return Utterance()
        def load():
            self.loads.append(True)
            return Engine()
        self.loader = load
        def fit(raw, adjusted, ratio, available_frames):
            self.fits.append((runner.sha256(raw), available_frames))
            wav(adjusted, 0.1)
            return ratio
        self.fit = fit

    def run_synth(self):
        runner.write_json(self.segments_file, {"segments": self.segments})
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(sys.modules, self.modules), mock.patch.object(runner, "validate_voice_reference", return_value={"method": "CPU test VAD stub; no perceptual claim"}), mock.patch.object(runner, "prepare_chatterbox_model", return_value=(self.model, self.loader)), mock.patch.object(runner, "fit_voice_to_frames", side_effect=self.fit):
            runner.synthesize_stage(self.config, None)

    def test_audio_master_keeps_long_raw_speech_without_any_fitting(self):
        self.config.update(timing_mode="audio_master", identity={"input_sha256": "a" * 64}, source_video_sha256="b" * 64)
        self.modules["soundfile"].write = lambda file, samples, rate, subtype: wav(file, 3.1, rate)
        self.run_synth()
        self.assertEqual(self.fits, [])
        with wave.open(str(self.directory / "dialogue.wav"), "rb") as pcm:
            self.assertEqual(pcm.getnframes(), 17 * 148800)
        timeline = runner.read_json(self.directory / "audio-master.json")
        self.assertEqual(timeline["sample_count"], 17 * 148800)
        self.assertEqual(timeline["source_segments"], self.segments)
        self.assertEqual(runner.read_json(self.segments_file)["segments"], self.segments)
        self.assertEqual(runner.read_json(self.directory / "voice-proof.json")["timing_adjustments"], [])

    def test_audio_master_shorter_audio_ends_at_its_last_sample_not_video_end(self):
        self.config.update(timing_mode="audio_master", identity={"input_sha256": "a" * 64}, source_video_sha256="b" * 64)
        self.run_synth()
        self.assertEqual(self.fits, [])
        timeline = runner.read_json(self.directory / "audio-master.json")
        self.assertEqual(timeline["sample_count"], 81600)
        self.assertEqual(timeline["turns"][-1]["output_end_sample"], 81600)
        self.assertEqual(len(timeline["turns"]), 17)
        self.assertLess(timeline["sample_count"] / timeline["sample_rate"], self.config["duration"])

    def test_audio_master_reuses_only_raw_clips_and_preserves_all_turns(self):
        self.config.update(timing_mode="audio_master", identity={"input_sha256": "a" * 64}, source_video_sha256="b" * 64)
        self.run_synth()
        before = runner.sha256(self.directory / "dialogue.wav")
        self.calls.clear()
        self.loads.clear()
        self.run_synth()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.loads, [])
        self.assertEqual(self.fits, [])
        self.assertEqual(runner.sha256(self.directory / "dialogue.wav"), before)
        self.assertTrue(all(item["reused"] for item in runner.read_json(self.directory / "voice-proof.json")["raw_cache"]))

    def test_failure_at_segment16_keeps15_then_retry_generates_only16_and17(self):
        self.fail_index = 16
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            self.run_synth()
        self.assertEqual(len(list(self.cache.directory.glob("*.json"))), 15)
        self.assertFalse((self.directory / "voice-proof.json").exists())
        self.fail_index = None
        self.calls.clear()
        self.run_synth()
        self.assertEqual(self.calls, ["Oración número 16.", "Oración número 17."])
        proof = runner.read_json(self.directory / "voice-proof.json")
        self.assertEqual(sum(item["reused"] for item in proof["raw_cache"]), 15)
        self.assertEqual(len(proof["segments"]), 17)

    def test_all_cached_avoids_model_load_but_regenerates_all_fitted_derivatives(self):
        self.run_synth()
        self.loads.clear()
        self.calls.clear()
        self.fits.clear()
        for item in self.segments:
            item["end"] = item["start"] + 0.8
        self.run_synth()
        self.assertEqual(self.loads, [])
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.fits), 17)
        self.assertEqual(runner.raw_voice_metadata(self.directory / "dialogue.wav")["duration"], 17)

    def test_changed_translation_regenerates_only_affected_raw(self):
        self.run_synth()
        self.calls.clear()
        self.segments[5]["translated_text"] = "Texto revisado."
        self.run_synth()
        self.assertEqual(self.calls, ["Texto revisado."])

    def test_corrupt_one_cached_wav_regenerates_only_corrupted_segment(self):
        self.run_synth()
        proof = runner.read_json(self.directory / "voice-proof.json")
        key = proof["raw_cache"][7]["identity_key"]
        wav(self.cache.directory / f"{key}.wav", amplitude=4000)
        self.calls.clear()
        self.run_synth()
        self.assertEqual(self.calls, ["Oración número 8."])

    def test_text_to_voice_joins_complete_audio_and_actual_timestamps(self):
        self.segments = [dict(self.item, id=index + 1, start=0, end=0, translated_text=f"Texto {index}.") for index in range(2)]
        self.config["duration"] = 0
        self.run_synth()
        result = runner.read_json(self.directory / "voice-segments.json")["segments"]
        self.assertEqual(result[0]["end"], 0.1)
        self.assertEqual(result[1]["start"], 0.28)
        self.assertEqual(result[1]["end"], 0.38)
        self.assertEqual(runner.raw_voice_metadata(self.directory / "dialogue.wav")["duration"], 0.38)


class TimingAcceptanceIntegrationTests(SynthIntegrationTests):
    test_failure_at_segment16_keeps15_then_retry_generates_only16_and17 = None
    test_all_cached_avoids_model_load_but_regenerates_all_fitted_derivatives = None
    test_changed_translation_regenerates_only_affected_raw = None
    test_corrupt_one_cached_wav_regenerates_only_corrupted_segment = None
    test_text_to_voice_joins_complete_audio_and_actual_timestamps = None
    test_audio_master_keeps_long_raw_speech_without_any_fitting = None
    test_audio_master_shorter_audio_ends_at_its_last_sample_not_video_end = None
    test_audio_master_reuses_only_raw_clips_and_preserves_all_turns = None

    def test_5_04_second_candidate_retries_same_text_then_accepts_4_36_seconds(self):
        self.segments = [dict(self.item, start=0, end=2)]
        self.config["duration"] = 2
        durations = iter((5.04, 4.36))
        self.modules["soundfile"] = types.SimpleNamespace(write=lambda file, samples, rate, subtype: wav(file, next(durations), rate))
        self.run_synth()
        self.assertEqual(self.calls, ["Hola.", "Hola."])
        proof = runner.read_json(self.directory / "voice-proof.json")
        self.assertEqual(proof["timing_adjustments"][0]["factor"], 2.18)
        self.assertEqual(len(list(self.cache.directory.glob("*.json"))), 1)
        self.assertEqual(len(list(self.cache.directory.glob("*.rejected.wav"))), 1)
        self.assertEqual(runner.raw_voice_metadata(self.cache.directory / (proof["raw_cache"][0]["identity_key"] + ".wav"))["duration"], 4.36)

    def test_final_fitting_factor_over_cap_never_receives_raw_marker(self):
        self.segments = [dict(self.item, start=0, end=2)]
        self.config["duration"] = 2
        fit = self.fit
        def exceeds_cap(*args):
            fit(*args)
            return 2.501
        self.fit = exceeds_cap
        with self.assertRaisesRegex(runner.PipelineError, "duas tentativas"):
            self.run_synth()
        self.assertEqual(self.calls, ["Hola.", "Hola."])
        self.assertFalse(list(self.cache.directory.glob("*.json")))
        self.assertFalse((self.directory / "voice-proof.json").exists())

    def test_real_slot_budget_caps_final_factor_instead_of_speaking_time_only(self):
        self.segments = [dict(self.item, start=0, end=2)]
        self.config["duration"] = 2
        self.modules["soundfile"] = types.SimpleNamespace(write=lambda file, samples, rate, subtype: wav(file, 4.0, rate))
        with mock.patch.object(runner, "duration_slot", return_value=1.5), self.assertRaisesRegex(runner.PipelineError, "duas tentativas"):
            self.run_synth()
        self.assertEqual(self.calls, ["Hola.", "Hola."])
        self.assertFalse(list(self.cache.directory.glob("*.json")))
        self.assertEqual(self.fits, [])


if __name__ == "__main__":
    unittest.main()
