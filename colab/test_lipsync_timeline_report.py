"""Evidence must cover v:0 before the notebook labels a result complete."""
import copy
import unittest

from dubstudio_runner_v2 import PipelineError, validate_lipsync_timeline


class LipsyncTimelineReportTests(unittest.TestCase):
    def setUp(self):
        self.report = {
            "operation": "lipsync_video",
            "lipsync": {"required": True, "confirmed": True, "applied": True,
                        "ambiguity": [], "confirmed_segments": [
                            {"start": 0.0, "end": 7.92},
                            {"start": 18.48, "end": 20.0}]},
            "preserved_no_speaker_visual_segments": [{"start": 7.92, "end": 18.48}],
        }

    def test_real_source_bounded_timeline_passes(self):
        validate_lipsync_timeline(self.report, 20.0)

    def test_actual_2008_regression_fails_in_notebook(self):
        self.report["lipsync"]["confirmed_segments"][-1]["end"] = 20.08
        with self.assertRaises(PipelineError):
            validate_lipsync_timeline(self.report, 20.0)

    def test_gaps_overlap_missing_tail_and_unsorted_rejected(self):
        for mutate in (lambda r: r["preserved_no_speaker_visual_segments"][0].update(start=8.1),
                       lambda r: r["preserved_no_speaker_visual_segments"][0].update(start=7.8),
                       lambda r: r["lipsync"]["confirmed_segments"][-1].update(end=19.8),
                       lambda r: r["lipsync"]["confirmed_segments"].reverse()):
            report = copy.deepcopy(self.report)
            mutate(report)
            with self.subTest(report=report), self.assertRaises(PipelineError):
                validate_lipsync_timeline(report, 20.0)

    def test_invalid_times_rejected(self):
        for value in (True, "20.0", float("inf"), float("nan"), None, -1):
            report = copy.deepcopy(self.report)
            report["lipsync"]["confirmed_segments"][-1]["end"] = value
            with self.subTest(value=value), self.assertRaises(PipelineError):
                validate_lipsync_timeline(report, 20.0)

    def test_ambiguity_and_no_applied_proof_rejected(self):
        for field, value in (("ambiguity", ["two faces"]), ("confirmed", False), ("applied", False)):
            report = copy.deepcopy(self.report)
            report["lipsync"][field] = value
            with self.subTest(field=field), self.assertRaises(PipelineError):
                validate_lipsync_timeline(report, 20.0)

    def test_faceless_narration_only_allowed_for_dubbing(self):
        self.report["lipsync"].update(confirmed_segments=[], confirmed=False, applied=False,
                                     no_visible_speaking_face=True, scene_analysis_complete=True)
        self.report["preserved_no_speaker_visual_segments"] = [{"start": 0.0, "end": 20.0}]
        with self.assertRaises(PipelineError):
            validate_lipsync_timeline(self.report, 20.0)
        self.report["operation"] = "dub_video"
        validate_lipsync_timeline(self.report, 20.0)

    def test_optional_lipsync_disabled_does_not_require_face_windows(self):
        validate_lipsync_timeline({"operation": "dub_video", "lipsync": {"required": False}}, 20.0)


if __name__ == "__main__":
    unittest.main()
