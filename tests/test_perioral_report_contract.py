"""Synthetic contract tests; never an acoustic/visual quality approval."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from colab import dubstudio_runner_v2 as runner
from colab import dubstudio_colab_worker as worker
import studio_bridge as bridge


def contract(count=1):
    return {'policy': worker.PERIORAL_MASK_CONTRACT, 'outside_face_source': 'original-frame-clock',
            'mask_outside_tracked_box': 0, 'mask_expands_box': False, 'segments': count,
            'region': 'mouth-and-lower-nose', 'source_landmarks': 'dwpose68-pinned',
            'eyes_forehead_alpha': 0, 'human_visual_review': 'pending',
            'evidence': [dict(index=i, frame_count=50, source_sha256='a'*64, mask_sha256='b'*64,
                              output_sha256='c'*64, model_sha256='d'*64, landmarks_sha256='e'*64)
                         for i in range(1, count + 1)],
            'encoding_note': 'H.264 lossy encoding may alter decoded pixels; generated content is spatially confined.'}


class PerioralReportTests(unittest.TestCase):
    def test_new_contract_round_trip_preserves_pending_visual_review(self):
        value = contract(2)
        published = runner.public_spatial_composition(value, 2)
        self.assertEqual(value, published)
        self.assertIsNot(value, published)
        bridge._validate_spatial_composition({'spatial_composition': published}, 2)

    def test_eye_or_forehead_alpha_and_coercions_rejected(self):
        for value in (1, True, False, '0', None):
            with self.subTest(value=value), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition({**contract(), 'eyes_forehead_alpha': value}, 1)

    def test_invented_human_review_and_regions_rejected(self):
        for key, value in (('human_visual_review', 'approved'), ('region', 'full-face'),
                           ('source_landmarks', 'from-bounding-box')):
            with self.subTest(key=key), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition({**contract(), key: value}, 1)

    def test_missing_or_mismatched_per_window_evidence_rejected(self):
        for evidence in (None, [], [contract()['evidence'][0]]):
            with self.subTest(evidence=evidence), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition({**contract(2), 'evidence': evidence}, 2)

    def test_physical_hashes_types_and_order_required(self):
        for key, value in (('source_sha256', ''), ('mask_sha256', 'private-path'), ('model_sha256', None),
                           ('landmarks_sha256', 'F'*64), ('output_sha256', 'a'*63),
                           ('index', True), ('index', 2), ('frame_count', True), ('frame_count', 0)):
            proof = contract()
            proof['evidence'][0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(runner.PipelineError):
                runner.public_spatial_composition(proof, 1)

    def test_mixed_sources_rejected(self):
        proof = contract(2)
        proof['evidence'][1]['source_sha256'] = 'f'*64
        with self.assertRaises(runner.PipelineError):
            runner.public_spatial_composition(proof, 2)

    def test_unknown_private_fields_stripped_before_publish_and_refused_on_import(self):
        proof = contract()
        proof['private_path'] = '/content/secret/voice.wav'
        proof['evidence'][0]['private_log'] = 'private'
        self.assertEqual(runner.public_spatial_composition(proof, 1), contract())
        with self.assertRaises(bridge.PipelineError):
            bridge._validate_spatial_composition({'spatial_composition': proof}, 1)

    def test_missing_physical_composition_proof_cannot_create_report(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.mp4'
            source.write_bytes(b'synthetic source')
            with self.assertRaises(worker.WorkerError):
                worker.perioral_composition_summary(Path(directory), source, 1)

    def test_completed_physical_files_required_and_changed_bytes_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, mask, output = root/'source.mp4', root/'musetalk-perioral-source-mask-v1-01.mkv', root/'lipsync-segmented-perioral-v1-01.mp4'
            for file in (source, mask, output):
                file.write_bytes(b'explicit synthetic bytes ' + file.name.encode())
            proof = dict(contract()['evidence'][0], contract=worker.PERIORAL_MASK_CONTRACT,
                         source_sha256=worker.sha256(source), mask_sha256=worker.sha256(mask), output_sha256=worker.sha256(output),
                         composition_completed=True, eyes_forehead_alpha=0, feather='inward-only', human_visual_review='pending')
            mask.with_suffix('.perioral.json').write_text(json.dumps(proof))
            summary = worker.perioral_composition_summary(root, source, 1)
            bridge._validate_spatial_composition({'spatial_composition': summary}, 1)
            self.assertEqual(summary['human_visual_review'], 'pending')
            mask.write_bytes(b'changed')
            with self.assertRaises(worker.WorkerError):
                worker.perioral_composition_summary(root, source, 1)


if __name__ == '__main__':
    unittest.main()
