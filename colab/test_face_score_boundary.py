"""Face association regressions on the canonical TalkNet frame clock."""
import unittest
import dubstudio_runner_v2 as runner


def track(start=0, end=25):
    return {"track": {"frame": list(range(start, end)), "bbox": [[10, 10, 60, 70]] * (end-start)}}


class FaceScoreBoundaryTests(unittest.TestCase):
    def associate(self, tracks, scores, transcript=(), selection="auto", count=100):
        return runner.identity_segments(tracks, scores, count, 100, 100, list(transcript), selection, "dub_video")

    def test_zero_and_negative_zero_are_not_positive_speaking_evidence(self):
        for zero in (0.0, -0.0):
            with self.subTest(score=zero):
                groups, ambiguous = self.associate([track(), track()], [[3.4]*25, [zero]*25],
                                                   [{"start": 0, "end": 1, "speaker": "MAIN"}])
                self.assertFalse(ambiguous)
                self.assertEqual([g["track_id"] for g in groups], [0])

    def test_two_positive_candidates_still_require_review(self):
        groups, ambiguous = self.associate([track(), track()], [[3.4]*25, [.01]*25])
        self.assertFalse(groups)
        self.assertTrue(ambiguous)

    def test_no_positive_evidence_is_not_confirmation(self):
        groups, ambiguous = self.associate([track()], [[0.0]*25])
        self.assertFalse(groups)
        self.assertFalse(ambiguous)

    def test_half_frame_boundary_at_cut_does_not_change_identity(self):
        transcript = [{"start": 0, "end": 6.74, "speaker": "INTRO"},
                      {"start": 6.88, "end": 7.8, "speaker": "MAIN"}]
        groups, ambiguous = self.associate([track(168, 200)], [[3.0]*32], transcript, count=200)
        self.assertFalse(ambiguous)
        self.assertEqual(groups[0]["speaker"], "MAIN")
        self.assertEqual(transcript[0]["end"], 6.74)

    def test_actual_one_frame_second_voice_still_fails(self):
        transcript = [{"start": 0, "end": 6.76, "speaker": "INTRO"},
                      {"start": 6.88, "end": 7.8, "speaker": "MAIN"}]
        groups, ambiguous = self.associate([track(168, 200)], [[3.0]*32], transcript, count=200)
        self.assertFalse(groups)
        self.assertIn("mais de um locutor", ambiguous[0]["reason"])

    def test_short_automatic_track_remains_blocked(self):
        groups, ambiguous = self.associate([track(0,18)], [[3.0]*18])
        self.assertFalse(groups)
        self.assertIn("curto", ambiguous[0]["reason"])

    def test_nonfinite_scores_do_not_confirm_speaking(self):
        for value in (float('inf'),float('-inf'),float('nan')):
            groups, _ = self.associate([track()], [[value]*25])
            self.assertFalse(groups)

    def test_odd_half_frame_and_float_noise_at_cut(self):
        for boundary in (6.78,6.78+1e-10,6.78-1e-10):
            transcript = [{"start":0,"end":boundary,"speaker":"INTRO"},
                          {"start":6.88,"end":7.8,"speaker":"MAIN"}]
            groups, ambiguous = self.associate([track(169,200)], [[3.0]*31],transcript,count=200)
            self.assertFalse(ambiguous)
            self.assertEqual(groups[0]['speaker'],'MAIN')

    def test_single_neutral_boundary_in_same_positive_track_is_continuous(self):
        scores = [2.0]*25
        scores[4] = 0.0
        groups, ambiguous = self.associate([track()], [scores], [{"start": 0, "end": 1, "speaker": "MAIN"}])
        self.assertFalse(ambiguous)
        self.assertEqual(len(groups),1)
        self.assertEqual(len(groups[0]['boxes']),25)
        self.assertEqual(groups[0]['neutral_boundary_frames'],1)

    def test_negative_gap_is_never_filled(self):
        scores = [2.0]*25
        scores[4] = -.01
        groups, ambiguous = self.associate([track()], [scores])
        self.assertFalse(groups)
        self.assertTrue(ambiguous)

    def test_neutral_boundary_cannot_bridge_different_tracks(self):
        groups, ambiguous = self.associate([track(0,5),track(5,25)], [[2,2,2,2,0],[2]*20])
        self.assertFalse(groups)
        self.assertTrue(ambiguous)

    def test_two_neutral_frames_are_not_invented(self):
        scores = [2.0]*25
        scores[4:6] = [0.0,0.0]
        groups, ambiguous = self.associate([track()], [scores])
        self.assertFalse(groups)
        self.assertTrue(ambiguous)


if __name__ == "__main__":
    unittest.main()
