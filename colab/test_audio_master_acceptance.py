"""Independent audio-master acceptance contracts; not model/GPU quality approval."""
import copy
import unittest

import dubstudio_audio_master as audio_master


def source_segments():
    return [
        {"id": 1, "speaker": "A", "start": 0.0, "end": 3.0, "text": "Fala da primeira pessoa."},
        {"id": 2, "speaker": "B", "start": 3.0, "end": 5.0, "text": "Fala da segunda pessoa."},
        {"id": 3, "speaker": "A", "start": 5.0, "end": 6.0, "text": "A primeira volta."},
    ]


class AudioMasterAcceptanceTests(unittest.TestCase):
    def build(self, segments=None, lengths=None):
        return audio_master.build_timeline(segments or source_segments(), lengths or [336000, 48000, 96000])

    def test_seven_second_voice_for_three_second_source_is_not_shortened(self):
        timeline = self.build()
        turn = timeline["turns"][0]
        self.assertEqual(turn["pcm_samples"], 7 * 48000)
        self.assertEqual(turn["output_end_sample"] - turn["output_start_sample"], 7 * 48000)
        self.assertEqual(turn["source_end"] - turn["source_start"], 3)

    def test_shorter_voice_cuts_the_video_budget_not_stretches_the_voice(self):
        timeline = self.build(lengths=[48000, 24000, 48000])
        self.assertEqual(timeline["sample_count"], 120000)
        self.assertEqual([turn["pcm_samples"] for turn in timeline["turns"]], [48000, 24000, 48000])
        self.assertLess(timeline["sample_count"] / timeline["sample_rate"], 6)

    def test_turn_order_a_b_a_is_preserved_instead_of_grouping_by_voice(self):
        timeline = self.build()
        self.assertEqual([turn["speaker"] for turn in timeline["turns"]], ["A", "B", "A"])
        self.assertEqual([turn["source_segment_id"] for turn in timeline["turns"]], [1, 2, 3])

    def test_three_people_keep_all_three_distinct_speaker_bindings(self):
        source = source_segments()
        source[2]["speaker"] = "C"
        timeline = self.build(source)
        self.assertEqual([turn["speaker"] for turn in timeline["turns"]], ["A", "B", "C"])
        audio_master.validate_timeline(timeline, source)

    def test_overlapping_original_speech_is_serialized_without_losing_samples(self):
        source = source_segments()
        source[1]["start"], source[1]["end"] = 2.0, 4.0
        source[2]["start"], source[2]["end"] = 2.0, 5.0
        timeline = self.build(source)
        self.assertEqual([turn["speaker"] for turn in timeline["turns"]], ["A", "B", "A"])
        self.assertEqual(timeline["sample_count"], 480000)
        for previous, current in zip(timeline["turns"], timeline["turns"][1:]):
            self.assertEqual(previous["output_end_sample"], current["output_start_sample"])

    def test_stable_source_start_sort_keeps_pcm_attached_to_its_original_id(self):
        ordered = source_segments()
        shuffled = [ordered[2], ordered[0], ordered[1]]
        timeline = self.build(shuffled, [96000, 336000, 48000])
        self.assertEqual([turn["source_segment_id"] for turn in timeline["turns"]], [1, 2, 3])
        self.assertEqual([turn["pcm_samples"] for turn in timeline["turns"]], [336000, 48000, 96000])

    def test_input_review_and_source_timestamps_remain_immutable(self):
        source = source_segments()
        source[0]["words"] = [{"word": "Fala", "start": .2, "end": .7}]
        frozen = copy.deepcopy(source)
        timeline = self.build(source)
        self.assertEqual(source, frozen)
        self.assertEqual(timeline["source_segments"], frozen)
        timeline["source_segments"][0]["words"][0]["word"] = "changed only in copy"
        self.assertEqual(source, frozen)

    def test_no_artificial_gap_or_original_padding_enters_sample_master(self):
        source = source_segments()
        source[1]["start"], source[1]["end"] = 100.0, 101.0
        source[2]["start"], source[2]["end"] = 120.0, 121.0
        timeline = self.build(source, [1, 2, 3])
        self.assertEqual(timeline["sample_count"], 6)
        self.assertEqual([(turn["output_start_sample"], turn["output_end_sample"])
                          for turn in timeline["turns"]], [(0, 1), (1, 3), (3, 6)])

    def test_one_sample_tail_is_preserved_without_duration_rounding(self):
        timeline = self.build(lengths=[48000, 48000, 1])
        self.assertEqual(timeline["sample_count"], 96001)
        self.assertEqual(timeline["turns"][-1]["output_end_sample"], 96001)
        audio_master.validate_timeline(timeline)

    def test_target_languages_may_have_independent_output_durations(self):
        polish = self.build(lengths=[336000, 48000, 96000])
        spanish = self.build(lengths=[192000, 96000, 72000])
        self.assertEqual(polish["source_segments_sha256"], spanish["source_segments_sha256"])
        self.assertNotEqual(polish["sample_count"], spanish["sample_count"])
        self.assertEqual(polish["source_segments"], spanish["source_segments"])

    def test_integer_length_mapping_produces_same_turns_as_aligned_list(self):
        ordered = self.build()
        mapped = self.build(lengths={1: 336000, 2: 48000, 3: 96000})
        self.assertEqual(mapped["turns"], ordered["turns"])

    def test_policy_is_explicit_and_does_not_claim_perceptual_quality(self):
        timeline = self.build()
        self.assertEqual(timeline["timing_mode"], "audio_master")
        self.assertEqual(timeline["policy"], "audio-master-speaker-turns-v1")
        self.assertNotIn("quality_approved", timeline)
        self.assertNotIn("timbre_identical", timeline)
        self.assertNotIn("gpu_validated", timeline)

    def test_foreign_or_changed_source_review_cannot_validate_this_timeline(self):
        timeline = self.build()
        for field, value in (("text", "Outra tradução"), ("speaker", "OTHER"), ("end", 2.9), ("id", 99)):
            source = source_segments()
            source[0][field] = value
            with self.subTest(field=field), self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline, source)

    def test_mutated_snapshot_or_digest_fails_instead_of_rewriting_approval(self):
        for mutation in (lambda value: value["source_segments"][0].update(text="altered"),
                         lambda value: value.update(source_segments_sha256="0" * 64)):
            timeline = self.build()
            mutation(timeline)
            with self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline)

    def test_wrong_turn_speaker_or_source_interval_is_rejected(self):
        changes = [("speaker", "C"), ("source_segment_id", 99), ("source_start", .01), ("source_end", 3.01)]
        for key, value in changes:
            timeline = self.build()
            timeline["turns"][0][key] = value
            with self.subTest(key=key), self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline)

    def test_output_gap_overlap_truncation_and_wrong_master_total_are_rejected(self):
        changes = [(1, "output_start_sample", 336001), (1, "output_start_sample", 335999),
                   (2, "output_end_sample", 479999), (0, "output_start_sample", 1),
                   (0, "pcm_samples", 336001)]
        for index, key, value in changes:
            timeline = self.build()
            timeline["turns"][index][key] = value
            with self.subTest(index=index, key=key), self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline)
        timeline = self.build()
        timeline["sample_count"] += 1
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.validate_timeline(timeline)

    def test_missing_or_reordered_turn_is_not_a_complete_result(self):
        timeline = self.build()
        timeline["turns"].pop()
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.validate_timeline(timeline)
        timeline = self.build()
        timeline["turns"][0], timeline["turns"][1] = timeline["turns"][1], timeline["turns"][0]
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.validate_timeline(timeline)

    def test_bool_nonfinite_fractional_and_zero_sample_counts_are_blocked(self):
        for value in (True, False, 0, -1, 1.5, float("nan"), float("inf"), "48000"):
            with self.subTest(value=value), self.assertRaises(audio_master.AudioMasterError):
                self.build(lengths=[48000, value, 48000])

    def test_length_count_mismatch_or_missing_mapping_entry_is_blocked(self):
        for lengths in ([48000], [48000] * 4, {1: 48000, 2: 48000}):
            with self.subTest(lengths=lengths), self.assertRaises(audio_master.AudioMasterError):
                self.build(lengths=lengths)

    def test_empty_invalid_or_missing_speaker_and_duplicate_ids_fail_safely(self):
        for segments in ([], [{"id": 1, "start": 0, "end": 1}],
                         [source_segments()[0], source_segments()[0]]):
            with self.subTest(segments=segments), self.assertRaises(audio_master.AudioMasterError):
                audio_master.build_timeline(segments, [48000] * len(segments))

    def test_invalid_original_timestamps_are_not_invented_or_ignored(self):
        for key, value in (("start", -.1), ("start", float("nan")), ("end", float("inf")),
                           ("end", 0), ("start", True)):
            source = source_segments()
            source[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(audio_master.AudioMasterError):
                self.build(source)

    def test_invalid_sample_rate_and_policy_are_rejected(self):
        for rate in (True, 0, -1, float("nan"), "48000"):
            with self.subTest(rate=rate), self.assertRaises(audio_master.AudioMasterError):
                audio_master.build_timeline(source_segments(), [48000] * 3, sample_rate=rate)
        for key, value in (("timing_mode", "legacy"), ("policy", "unknown")):
            timeline = self.build()
            timeline[key] = value
            with self.subTest(key=key), self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline)

    def test_tts_content_can_have_natural_duration_without_a_source_video(self):
        source = [{"id": 1, "speaker": "REFERENCE", "start": 0, "end": 1,
                   "text": "Texto autorizado com duração natural."}]
        timeline = self.build(source, [48000 * 12 + 7])
        self.assertEqual(timeline["sample_count"], 48000 * 12 + 7)
        self.assertNotIn("video_mapping", timeline)
        audio_master.validate_timeline(timeline, source)


def confirmed_analysis():
    groups = []
    for index, (speaker, first, last) in enumerate((("A", 0, 75), ("B", 75, 125), ("A", 125, 150))):
        groups.append({"speaker": speaker, "fps": 25, "start_frame": first, "end_frame": last,
                       "boxes": [[10, 10, 80, 90]] * (last - first), "track_id": index,
                       "face_confirmation": "detected", "speaker_confirmation": "active_speaker",
                       "coordinate_source": "synthetic confirmed geometry for CPU contract test"})
    return {"confirmed_face_segments": groups, "ambiguity": []}


class AudioMasterVisualAcceptanceTests(unittest.TestCase):
    metadata = {"streams": [{"codec_type": "video", "avg_frame_rate": "30/1", "nb_frames": "180",
                             "width": 100, "height": 100}]}

    def map(self, lengths=None, analysis=None):
        timeline = audio_master.build_timeline(source_segments(), lengths or [336000, 48000, 96000])
        return audio_master.enrich_video_mapping(timeline, copy.deepcopy(self.metadata),
                                                 confirmed_analysis() if analysis is None else analysis)

    def test_video_duration_comes_from_final_audio_not_original_six_seconds(self):
        timeline, _ = self.map()
        self.assertEqual(timeline["source_video"]["frame_count"], 180)
        self.assertEqual(timeline["video_base"]["frame_count"], 300)
        self.assertEqual(timeline["sample_count"] / timeline["sample_rate"], 10)

    def test_longer_turn_repeats_its_person_and_shorter_turn_trims_its_person(self):
        timeline, groups = self.map()
        maps = timeline["visual_mapping"]
        self.assertEqual([item["method"] for item in maps], ["repeat", "trim", "repeat"])
        self.assertEqual([item["speaker"] for item in maps], ["A", "B", "A"])
        self.assertEqual([(item["source_start_frame"], item["source_end_frame"]) for item in maps],
                         [(0, 90), (90, 150), (150, 180)])
        self.assertEqual([group["track_id"] for group in groups], [0, 1, 2])

    def test_loops_run_forward_at_normal_speed_and_never_enter_another_person(self):
        timeline, _ = self.map()
        for item in timeline["visual_mapping"]:
            frames = [audio_master.source_frame_for_output(item, index)
                      for index in range(item["output_start_frame"], item["output_end_frame"])]
            self.assertTrue(all(item["source_start_frame"] <= frame < item["source_end_frame"] for frame in frames))
            self.assertTrue(all(current == previous + 1 or current == item["source_start_frame"]
                                for previous, current in zip(frames, frames[1:])))

    def test_output_switches_follow_each_persons_pcm_end_with_frame_precision(self):
        timeline, _ = self.map([336007, 48009, 96011])
        fps = timeline["video_base"]["fps_num"] / timeline["video_base"]["fps_den"]
        for turn, item in zip(timeline["turns"], timeline["visual_mapping"]):
            self.assertLessEqual(abs(item["output_end_frame"] / fps - turn["output_end_sample"] / 48000), 1 / fps)
        self.assertLessEqual(timeline["video_base"]["frame_count"] / fps - timeline["sample_count"] / 48000, 1 / fps)

    def test_source_and_analysis_are_not_mutated_by_video_mapping(self):
        source = audio_master.build_timeline(source_segments(), [336000, 48000, 96000])
        analysis = confirmed_analysis()
        frozen_source, frozen_analysis = copy.deepcopy(source), copy.deepcopy(analysis)
        audio_master.enrich_video_mapping(source, self.metadata, analysis)
        self.assertEqual(source, frozen_source)
        self.assertEqual(analysis, frozen_analysis)

    def test_absent_one_speakers_face_cannot_borrow_the_other_face(self):
        analysis = confirmed_analysis()
        analysis["confirmed_face_segments"] = [analysis["confirmed_face_segments"][0],
                                               analysis["confirmed_face_segments"][2]]
        with self.assertRaises(audio_master.AudioMasterError):
            self.map(analysis=analysis)

    def test_ambiguous_tracks_block_before_assembling_a_false_complete_video(self):
        for key in ("ambiguity", "ambiguous_intervals"):
            analysis = confirmed_analysis()
            analysis[key] = [{"start": 3, "end": 4, "reason": "unknown speaker face"}]
            with self.subTest(key=key), self.assertRaises(audio_master.AudioMasterError):
                self.map(analysis=analysis)

    def test_low_confidence_voice_or_face_evidence_is_not_a_selected_person(self):
        for key, value in (("speaker_confirmation", "unknown"), ("face_confirmation", "guessed")):
            analysis = confirmed_analysis()
            analysis["confirmed_face_segments"][1][key] = value
            with self.subTest(key=key), self.assertRaises(audio_master.AudioMasterError):
                self.map(analysis=analysis)

    def test_missing_or_outside_source_boxes_block_instead_of_inventing_geometry(self):
        for boxes in ([], [[-1, 10, 80, 90]] * 50, [[10, 10, 101, 90]] * 50):
            analysis = confirmed_analysis()
            analysis["confirmed_face_segments"][1]["boxes"] = boxes
            with self.subTest(boxes=str(boxes)[:50]), self.assertRaises(audio_master.AudioMasterError):
                self.map(analysis=analysis)

    def test_no_face_narration_requires_explicit_confirmation_and_cannot_be_lipsynced(self):
        timeline = audio_master.build_timeline(source_segments(), [336000, 48000, 96000])
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.enrich_video_mapping(timeline, self.metadata, {}, require_faces=False)
        analysis = {"narration_no_face_confirmed": True}
        mapped, _ = audio_master.enrich_video_mapping(timeline, self.metadata, analysis, require_faces=False)
        self.assertTrue(all(item["face_confirmation"] == "no_face_confirmed" for item in mapped["visual_mapping"]))
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.enrich_video_mapping(timeline, self.metadata, analysis, require_faces=True)

    def test_video_frame_map_cannot_be_used_outside_its_own_turn(self):
        timeline, _ = self.map()
        item = timeline["visual_mapping"][1]
        for frame in (item["output_start_frame"] - 1, item["output_end_frame"]):
            with self.subTest(frame=frame), self.assertRaises(audio_master.AudioMasterError):
                audio_master.source_frame_for_output(item, frame)

    def test_tampered_donor_from_another_person_is_not_accepted_with_same_speaker_label(self):
        timeline, _ = self.map()
        item = timeline["visual_mapping"][0]
        item["source_start_frame"], item["source_end_frame"] = 90, 150
        with self.assertRaises(audio_master.AudioMasterError):
            audio_master.validate_timeline(timeline)

    def test_truncated_or_extended_video_proof_does_not_pass_audio_master_validation(self):
        for change in (-1, 1):
            timeline, _ = self.map()
            timeline["video_base"]["frame_count"] += change
            with self.subTest(change=change), self.assertRaises(audio_master.AudioMasterError):
                audio_master.validate_timeline(timeline)


if __name__ == "__main__":
    unittest.main()
