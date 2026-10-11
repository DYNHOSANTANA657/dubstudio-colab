"""CPU-only timing/import contracts, not perceptual speaker or lip-sync tests."""

import copy
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

import studio_bridge as bridge
from colab.dubstudio_audio_master import build_timeline
from pipeline import Job, PipelineError, probe_video


def encoded(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")


def subtitles(timeline, key):
    # Independently construct the notebook's canonical SRT format from the
    # physical PCM timeline, not the original source-video slot boundaries.
    def timestamp(samples):
        millis = round(samples / timeline["sample_rate"] * 1000)
        hours, millis = divmod(millis, 3600000)
        minutes, millis = divmod(millis, 60000)
        seconds, millis = divmod(millis, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"
    source = {item["id"]: item for item in timeline["source_segments"]}
    rows = []
    for index, turn in enumerate(timeline["turns"], 1):
        rows.extend([str(index), f"{timestamp(turn['output_start_sample'])} --> {timestamp(turn['output_end_sample'])}",
                     source[turn["source_segment_id"]][key].strip(), ""])
    return "\n".join(rows).encode("utf-8")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
class AudioMasterImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory(prefix="am-media-")
        cls.shared = Path(cls.folder.name)
        cls.source = cls.shared / "source.mp4"
        cls.make_video(cls.source, 3)
        cls.media = {}
        for duration in (2, 5, 5.003):
            video, wav, mp3 = (cls.shared / f"{duration}.{suffix}" for suffix in ("mp4", "wav", "mp3"))
            cls.make_video(video, duration)
            subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i",
                            f"sine=frequency=600:sample_rate=48000:duration={duration}",
                            "-ac", "2", "-c:a", "pcm_s16le", str(wav)], check=True, capture_output=True)
            subprocess.run(["ffmpeg", "-y", "-i", str(wav), "-c:a", "libmp3lame", str(mp3)],
                           check=True, capture_output=True)
            cls.media[duration] = {"video": video.read_bytes(), "master_audio": wav.read_bytes(),
                                   "wav": wav.read_bytes(), "mp3": mp3.read_bytes()}

    @classmethod
    def make_video(cls, target, duration):
        subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=green:s=160x90:r=25",
                        "-f", "lavfi", "-i", f"sine=frequency=600:sample_rate=48000:duration={duration}",
                        "-frames:v", str(math.ceil(duration * 25)), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        str(target)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="am-job-")
        self.root = Path(self.temp.name)
        original = self.root / "original.mp4"
        shutil.copy2(self.source, original)
        self.job = Job(id="abc123abc123", directory=self.root, input_path=original,
                       options={"operation": "dub_video", "target_languages": ["es-ES"],
                                "voice_clone_consent": True, "require_lipsync": False},
                       metadata=probe_video(original))
        bridge.prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / bridge.MANIFEST_NAME).read_text(encoding="utf-8"))

    def tearDown(self):
        self.temp.cleanup()

    def result(self, duration=5, *, edit_timeline=None, edit_report=None, edit_blobs=None, omit=()):
        operation, language = self.manifest["operation"], self.manifest["target_languages"][0]
        segments = [{"id": index + 1, "start": float(index), "end": float(index + 1),
                     "speaker": ("A", "B", "A")[index], "text": f"Original {index}",
                     "translated_text": f"Traducción {index}"} for index in range(3)]
        sample_count = round(duration * 48000)
        counts = [32000, 32000, 32000] if duration == 2 else [96000, 48000, sample_count - 144000]
        if operation == "lipsync_video":
            segments = [{"id": 1, "start": 0.0, "end": 3.0, "speaker": "USER_AUDIO", "text": "Áudio autorizado", "translated_text": "Áudio autorizado"}]
            counts = [sample_count]
        timeline = build_timeline(segments, counts)
        timeline.update(input_sha256=self.manifest["input_sha256"],
                        source_video_sha256=self.manifest["inputs"]["video"]["sha256"], target_language=language,
                        audio_master={"path": "dialogue.wav", "sha256": hashlib.sha256(self.media[duration]["master_audio"]).hexdigest(),
                                      "sample_rate": 48000, "sample_count": sample_count},
                        source_video={"fps_num": 25, "fps_den": 1, "frame_count": 75, "width": 160, "height": 90},
                        video_base={"path": "base-video.mp4", "sha256": "a" * 64, "fps_num": 25, "fps_den": 1,
                                    "frame_count": math.ceil(duration * 25), "width": 160, "height": 90})
        if operation == "lipsync_video":
            timeline["submitted_audio_sha256"] = self.manifest["inputs"]["audio"]["sha256"]
        mapping, cursor = [], 0
        for index, turn in enumerate(timeline["turns"]):
            end = math.ceil(duration * 25) if index == len(timeline["turns"]) - 1 else math.floor(turn["output_end_sample"] / 48000 * 25 + 0.5)
            donor_start, donor_end = (0, 75) if operation == "lipsync_video" else (index * 25, (index + 1) * 25)
            mapping.append({"source_segment_id": turn["source_segment_id"], "speaker": turn["speaker"],
                            "source_start_frame": donor_start, "source_end_frame": donor_end,
                            "output_start_frame": cursor, "output_end_frame": end,
                            "method": "repeat" if end - cursor > donor_end - donor_start else "trim",
                            "source_face_track_id": 0 if turn["speaker"] == "A" else 1, "face_confirmation": "user_selected",
                            "speaker_confirmation": "user_selected", "coordinate_source": "synthetic contract fixture"})
            cursor = end
        timeline["visual_mapping"] = mapping
        if edit_timeline:
            edit_timeline(timeline)
        binding = {"schema_version": bridge.SCHEMA_VERSION, "pipeline_version": bridge.PIPELINE_VERSION,
                   "job_id": self.job.id, "input_sha256": self.manifest["input_sha256"], "operation": operation,
                   "input_checksums": {role: item["sha256"] for role, item in self.manifest["inputs"].items()}}
        report = {**binding, "target_language": language, "timing_mode": "audio_master",
                  "summary": {"duration": float(duration)}, "voice_clone": {"required": operation == "dub_video", "confirmed": operation == "dub_video", "authorized": True},
                  "lipsync": {"required": False, "confirmed": False, "applied": False},
                  "timing_adjustments": [], "master_audio_sha256": timeline["audio_master"]["sha256"],
                  "timeline_sha256": hashlib.sha256(encoded(timeline)).hexdigest(),
                  "timing": {key: timeline[key] for key in ("policy", "sample_rate", "sample_count", "source_segments_sha256", "video_base")}}
        if self.manifest["options"]["require_lipsync"]:
            # Synthetic report metadata exercises the importer contract only;
            # it is explicitly not a claim of rendered mouth/voice quality.
            report["lipsync"] = {"required": True, "confirmed": True, "applied": True, "ambiguity": [],
                "confirmed_segments": [{"start": item["output_start_frame"] / 25, "end": item["output_end_frame"] / 25,
                                        "speaker": item["speaker"], "coordinate_source": "synthetic contract fixture"} for item in mapping]}
            report["preserved_no_speaker_visual_segments"] = []
            report["spatial_composition"] = {
                "policy": "perioral-mapped-source-dwpose68-inner-feather-v1", "outside_face_source": "mapped-source-frame-clock",
                "mask_outside_tracked_box": 0, "mask_expands_box": False, "segments": len(mapping),
                "region": "mouth-and-lower-nose", "source_landmarks": "dwpose68-pinned", "eyes_forehead_alpha": 0,
                "human_visual_review": "pending", "encoding_note": "H.264 lossy encoding may alter decoded pixels; generated content is spatially confined.",
                "source_video_sha256": timeline["source_video_sha256"], "timeline_sha256": report["timeline_sha256"],
                "output_base_sha256": timeline["video_base"]["sha256"], "audio_master_sha256": report["master_audio_sha256"],
                "evidence": [{"index": index + 1, "frame_count": item["output_end_frame"] - item["output_start_frame"],
                              "source_sha256": timeline["video_base"]["sha256"], "mask_sha256": "b" * 64,
                              "output_sha256": "c" * 64, "model_sha256": "d" * 64, "landmarks_sha256": "e" * 64}
                             for index, item in enumerate(mapping)]}
        if edit_report:
            edit_report(report)
        files = {"video": "video.mp4", "master_audio": "dialogue.wav", "wav": "mix.wav", "mp3": "voice.mp3",
                 "timeline": "timeline.json", "report": "report.json", "segments": "translated.json",
                 "original_srt": "original.srt", "translation_srt": "translation.srt"}
        blobs = {**self.media[duration], "timeline": encoded(timeline), "report": encoded(report),
                 "segments": encoded({"target_language": language, "segments": segments}),
                 "original_srt": subtitles(timeline, "text"),
                 "translation_srt": subtitles(timeline, "translated_text")}
        if edit_blobs:
            edit_blobs(blobs)
        paths = {role: f"outputs/{language}/{name}" for role, name in files.items() if role not in omit}
        checksums = {path: hashlib.sha256(blobs[role]).hexdigest() for role, path in paths.items()}
        result = {**binding, "outputs": [{"language": language, "status": "complete", "files": paths, "checksums": checksums}]}
        output = self.root / "returned.zip"
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("result.json", encoded(result))
            for role, path in paths.items():
                archive.writestr(path, blobs[role])
        return output

    def test_new_project_defaults_to_audio_master(self):
        self.assertEqual(self.manifest["options"]["timing_mode"], "audio_master")
        with zipfile.ZipFile(self.root / bridge.PACKAGE_NAME) as archive:
            self.assertIn("worker/dubstudio_audio_master.py", archive.namelist())

    def test_longer_audio_video_is_accepted_against_master_not_source(self):
        output = bridge.import_studio_result(self.job, self.result())
        proof = output["outputs"][0]["validation"]
        self.assertEqual(proof["audio_master"]["sample_count"], 240000)
        self.assertEqual(proof["timing_mode"], "audio_master")
        self.assertEqual(proof["human_visual_review"], "pending")
        json.dumps(output)

    def test_shorter_audio_video_is_accepted(self):
        output = bridge.import_studio_result(self.job, self.result(2))
        self.assertAlmostEqual(output["outputs"][0]["validation"]["media"]["video"]["video_duration"], 2)

    def test_fractional_audio_end_accepts_only_frame_quantisation(self):
        output = bridge.import_studio_result(self.job, self.result(5.003))
        proof = output["outputs"][0]["validation"]
        self.assertEqual(proof["audio_master"]["sample_count"], 240144)
        self.assertAlmostEqual(proof["media"]["video"]["video_duration"], 5.04)

    def test_missing_audio_master_files_are_rejected(self):
        for role in ("timeline", "master_audio"):
            with self.subTest(role=role), self.assertRaisesRegex(PipelineError, "incompleto"):
                bridge.import_studio_result(self.job, self.result(omit={role}))

    def test_wrong_source_binding_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "outro vídeo"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value.update(source_video_sha256="0" * 64)))

    def test_changed_source_snapshot_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "mapa de áudio natural"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["source_segments"][0].update(speaker="C")))

    def test_sample_gap_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "mapa de áudio natural"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["turns"][1].update(output_start_sample=96001)))

    def test_speed_changes_are_rejected(self):
        with self.assertRaisesRegex(PipelineError, "ajuste de velocidade"):
            bridge.import_studio_result(self.job, self.result(edit_report=lambda value: value.update(timing_adjustments=[{"factor": 1.2}])))

    def test_video_cut_to_old_duration_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "duração da faixa de vídeo"):
            bridge.import_studio_result(self.job, self.result(edit_blobs=lambda blobs: blobs.update(video=self.source.read_bytes())))

    def test_wrong_master_pcm_count_is_rejected(self):
        with self.assertRaises(PipelineError):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["audio_master"].update(sample_count=239999)))

    def test_wrong_mapped_person_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "mapa de áudio natural"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["visual_mapping"][1].update(speaker="A")))

    def test_source_frame_clock_forgery_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "quadros originais"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["source_video"].update(frame_count=76)))

    def test_subtitle_snapshot_text_changes_are_rejected(self):
        with self.assertRaisesRegex(PipelineError, "revisão aprovada"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["source_segments"][0].update(translated_text="Texto trocado")))

    def test_subtitles_cannot_keep_original_video_clock(self):
        for role in ("original_srt", "translation_srt"):
            def edit(blobs):
                blobs[role] = blobs[role].replace(b"00:00:02,000", b"00:00:01,000")
            with self.subTest(role=role), self.assertRaisesRegex(PipelineError, "relógio de amostras"):
                bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_subtitle_texts_cannot_be_swapped_between_turns(self):
        def edit(blobs):
            text = blobs["translation_srt"].decode("utf-8")
            blobs["translation_srt"] = text.replace("Traducción 0", "Traducción 1", 1).encode("utf-8")
        with self.assertRaisesRegex(PipelineError, "textos, à ordem"):
            bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_original_and_translation_subtitles_cannot_be_swapped(self):
        def edit(blobs):
            blobs["original_srt"], blobs["translation_srt"] = blobs["translation_srt"], blobs["original_srt"]
        with self.assertRaisesRegex(PipelineError, "textos, à ordem"):
            bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_subtitle_turn_cannot_be_omitted_or_duplicated(self):
        for duplicate in (False, True):
            def edit(blobs):
                cues = blobs["translation_srt"].decode("utf-8").split("\n\n")
                selected = cues[:1] + cues[2:] if not duplicate else cues[:1] + cues
                blobs["translation_srt"] = "\n\n".join(selected).encode("utf-8")
            with self.subTest(duplicate=duplicate), self.assertRaisesRegex(PipelineError, "textos, à ordem"):
                bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_subtitle_numbering_must_follow_natural_turn_order(self):
        def edit(blobs):
            blobs["translation_srt"] = blobs["translation_srt"].replace(b"\n\n2\n", b"\n\n4\n")
        with self.assertRaisesRegex(PipelineError, "textos, à ordem"):
            bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_subtitle_cues_cannot_be_reordered(self):
        def edit(blobs):
            cues = blobs["original_srt"].decode("utf-8").split("\n\n")
            blobs["original_srt"] = "\n\n".join([cues[1], cues[0], *cues[2:]]).encode("utf-8")
        with self.assertRaisesRegex(PipelineError, "textos, à ordem"):
            bridge.import_studio_result(self.job, self.result(edit_blobs=edit))

    def test_unconfirmed_face_mapping_is_rejected(self):
        with self.assertRaisesRegex(PipelineError, "identidade facial"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value["visual_mapping"][0].update(face_confirmation="unknown")))

    def test_saved_manifest_without_mode_keeps_legacy_contract(self):
        legacy = copy.deepcopy(self.manifest)
        legacy["options"].pop("timing_mode")
        (self.root / bridge.MANIFEST_NAME).write_bytes(encoded(legacy))
        bridge.prepare_studio_package(self.job)
        saved = json.loads((self.root / bridge.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertEqual(saved["options"]["timing_mode"], "source_video")

    def test_new_uploaded_audio_can_exceed_original_video(self):
        audio = self.root / "submitted.wav"
        audio.write_bytes(self.media[5]["master_audio"])
        self.job.options.update(operation="lipsync_video", timing_mode="audio_master", target_languages=["und"])
        self.job.metadata["studio_inputs"] = {"audio": {"path": audio.name}}
        bridge.prepare_studio_package(self.job)
        saved = json.loads((self.root / bridge.MANIFEST_NAME).read_text(encoding="utf-8"))
        self.assertIn("audio", saved["inputs"])

    def prepare_uploaded_audio(self):
        audio = self.root / "submitted.wav"
        audio.write_bytes(self.media[5]["master_audio"])
        self.job.options.update(operation="lipsync_video", timing_mode="audio_master", target_languages=["und"])
        self.job.metadata["studio_inputs"] = {"audio": {"path": audio.name}}
        bridge.prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / bridge.MANIFEST_NAME).read_text(encoding="utf-8"))

    def test_uploaded_long_audio_roundtrip_uses_pcm_not_old_video_end(self):
        self.prepare_uploaded_audio()
        output = bridge.import_studio_result(self.job, self.result())
        proof = output["outputs"][0]["validation"]
        self.assertEqual(proof["audio_master"]["duration"], 5)
        self.assertEqual(proof["human_visual_review"], "pending")

    def test_mapped_spatial_requires_all_four_bindings(self):
        self.prepare_uploaded_audio()
        for field in ("source_video_sha256", "timeline_sha256", "output_base_sha256", "audio_master_sha256"):
            with self.subTest(field=field), self.assertRaisesRegex(PipelineError, "composição sobre o vídeo adaptado"):
                bridge.import_studio_result(self.job, self.result(edit_report=lambda value: value["spatial_composition"].update({field: "0" * 64})))

    def test_uploaded_audio_binding_cannot_be_swapped(self):
        self.prepare_uploaded_audio()
        with self.assertRaisesRegex(PipelineError, "áudio enviado"):
            bridge.import_studio_result(self.job, self.result(edit_timeline=lambda value: value.update(submitted_audio_sha256="0" * 64)))

    def add_neutral_boundary(self, report, start=0.0, end=0.04):
        report["lipsync"]["confirmed_segments"][0]["start"] = end
        report["spatial_composition"]["evidence"][0]["frame_count"] = 125 - round(end * 25)
        report["preserved_no_speaker_visual_segments"] = [{
            "start": start, "end": end, "kind": "canonical_neutral_boundary",
            "policy": "audio-master-native-boundary-preservation-v1", "source_segment_id": 1,
            "speaker": "USER_AUDIO", "maximum_seconds": 0.04, "machine_confirmed": False,
            "reason": "Fronteira canônica sem inferência; quadro original da pessoa preservado, sem score positivo",
        }]

    def test_explicit_single_frame_neutral_boundary_is_not_fabricated_score(self):
        self.prepare_uploaded_audio()
        output = bridge.import_studio_result(self.job, self.result(edit_report=self.add_neutral_boundary))
        self.assertEqual(output["outputs"][0]["validation"]["human_visual_review"], "pending")

    def test_neutral_boundary_must_not_exceed_one_frame(self):
        self.prepare_uploaded_audio()
        with self.assertRaisesRegex(PipelineError, "fronteira neutra"):
            bridge.import_studio_result(self.job, self.result(edit_report=lambda value: self.add_neutral_boundary(value, end=0.08)))

    def test_neutral_boundary_cannot_be_inside_a_speaking_turn(self):
        self.prepare_uploaded_audio()
        with self.assertRaisesRegex(PipelineError, "fronteira neutra"):
            bridge.import_studio_result(self.job, self.result(edit_report=lambda value: self.add_neutral_boundary(value, start=0.48, end=0.52)))

    def test_neutral_boundary_cannot_claim_machine_confirmation(self):
        self.prepare_uploaded_audio()
        def edit(report):
            self.add_neutral_boundary(report)
            report["preserved_no_speaker_visual_segments"][0]["machine_confirmed"] = True
        with self.assertRaisesRegex(PipelineError, "fronteira neutra"):
            bridge.import_studio_result(self.job, self.result(edit_report=edit))

    def test_generic_preservation_cannot_hide_unrendered_speaking_face(self):
        self.prepare_uploaded_audio()
        def edit(report):
            self.add_neutral_boundary(report)
            report["preserved_no_speaker_visual_segments"][0].pop("kind")
        with self.assertRaisesRegex(PipelineError, "rosto falante foi preservado"):
            bridge.import_studio_result(self.job, self.result(edit_report=edit))

    def test_spatial_proof_must_count_native_applied_frames(self):
        self.prepare_uploaded_audio()
        with self.assertRaisesRegex(PipelineError, "quadros nativos"):
            bridge.import_studio_result(self.job, self.result(edit_report=lambda value: value["spatial_composition"]["evidence"][0].update(frame_count=124)))

    def test_lipsync_window_cannot_cross_next_person(self):
        self.job.options["require_lipsync"] = True
        bridge.prepare_studio_package(self.job)
        self.manifest = json.loads((self.root / bridge.MANIFEST_NAME).read_text(encoding="utf-8"))
        with self.assertRaisesRegex(PipelineError, "cruza uma troca de pessoa"):
            bridge.import_studio_result(self.job, self.result(edit_report=lambda value: value["lipsync"]["confirmed_segments"][0].update(end=2.04)))

    def test_legacy_uploaded_audio_still_blocks_truncation(self):
        audio = self.root / "submitted.wav"
        audio.write_bytes(self.media[5]["master_audio"])
        self.job.options.update(operation="lipsync_video", timing_mode="source_video", target_languages=["und"])
        self.job.metadata["studio_inputs"] = {"audio": {"path": audio.name}}
        with self.assertRaisesRegex(PipelineError, "áudio enviado excede"):
            bridge.prepare_studio_package(self.job)


if __name__ == "__main__":
    unittest.main()
