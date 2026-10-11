"""Canonical TalkNet timestamps; offline tests do not certify MuseTalk quality."""
import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import dubstudio_colab_worker as worker


FIXTURE = '''
images = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic"
def normalization(args):
    if args.duration == 0:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.threads, args.output))
    else:
        command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
                   (args.videoPath, args.threads, args.start, args.end, args.output))
    return command
'''


class NormalizationPatchTests(unittest.TestCase):
    def test_only_two_known_commands_change_and_patch_is_idempotent(self):
        source = FIXTURE + '\nweights = "mandatory-model.pth"\n'
        repaired = worker.canonical_talknet_source(source)
        ast.parse(repaired)
        self.assertEqual(repaired.count("fps=25:start_time=0"), 2)
        self.assertEqual(repaired.count("-c:a pcm_s16le"), 2)
        self.assertNotIn("-async 1 -r 25", repaired)
        self.assertIn('weights = "mandatory-model.pth"', repaired)
        self.assertIn('images = "ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic"', repaired)
        self.assertEqual(worker.canonical_talknet_source(repaired), repaired)

    def test_unexpected_additional_async_command_is_not_ignored(self):
        extra = '\nextra = "ffmpeg -y -i %s -qscale:v 2 -threads %d -async 2 -r 25 %s"\n'
        with self.assertRaises(worker.WorkerError):
            worker.canonical_talknet_source(FIXTURE + extra)

    def test_unrecognized_or_partially_repaired_source_fails_closed(self):
        changed = FIXTURE.replace("-async 1 -r 25", "-async 2 -r 25", 1)
        for source in (changed, FIXTURE + '\nextra = "-async 1 -r 25"\n',
                       "def invalid(:", FIXTURE.replace("-qscale:v 2 -threads %d -async 1", "-qscale:v 3 -threads %d -async 1", 1)):
            with self.subTest(source=source), self.assertRaises(worker.WorkerError):
                worker.canonical_talknet_source(source)

    def test_original_backup_and_atomic_replacement_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "demoTalkNet.py"
            target.write_text(FIXTURE, encoding="utf-8")
            initial = target.read_bytes()
            revision = worker.patch_talknet_video_normalization(root)
            repaired = target.read_bytes()
            self.assertEqual(revision, worker.TALKNET_NORMALIZATION_REVISION)
            self.assertEqual(target.with_suffix(".py.dubstudio-normalization-original").read_bytes(), initial)
            worker.patch_talknet_video_normalization(root)
            self.assertEqual(target.read_bytes(), repaired)
            self.assertEqual(target.with_suffix(".py.dubstudio-normalization-original").read_bytes(), initial)
            self.assertFalse(target.with_suffix(".py.normalization-writing").exists())

    def test_invalid_code_never_replaces_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "demoTalkNet.py"
            target.write_bytes(b"unexpected upstream code\n")
            with self.assertRaises(worker.WorkerError):
                worker.patch_talknet_video_normalization(root)
            self.assertEqual(target.read_bytes(), b"unexpected upstream code\n")
            self.assertFalse(target.with_suffix(".py.dubstudio-normalization-original").exists())


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for real CPU media regression")
class RealNormalizationTests(unittest.TestCase):
    def test_avi_frames_follow_source_timestamps_without_encoded_audio_padding(self):
        def execute(arguments):
            return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *arguments],
                                  capture_output=True, check=True, timeout=30).stdout
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent,
                                         prefix=".test-normalization-") as temporary:
            root = Path(temporary)
            original, avi = root / "original.mp4", root / "video.avi"
            execute(["-f", "lavfi", "-i", "testsrc2=size=120x80:rate=30:duration=2",
                     "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                     "-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-c:a", "aac", str(original)])
            execute(["-i", str(original), "-qscale:v", "2", "-threads", "2", "-async", "1",
                     "-vf", "setpts=PTS-STARTPTS,fps=25:start_time=0", "-c:a", "pcm_s16le", str(avi)])
            def frames(path, expression, count):
                return execute(["-i", str(path), "-vf", f"select='{expression}',format=gray",
                                "-vsync", "0", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", "gray", "-"])
            for seconds in (0.8, 1.2, 1.6):
                source_index, avi_index = round(seconds * 30), round(seconds * 25)
                target = frames(avi, f"eq(n,{avi_index})", 1)
                candidates = frames(original, f"between(n,{source_index-3},{source_index+3})", 7)
                size = 120 * 80
                self.assertEqual(len(target), size)
                self.assertEqual(len(candidates), 7 * size)
                errors = [sum((a-b)**2 for a,b in zip(target,candidates[i*size:(i+1)*size]))
                          for i in range(7)]
                self.assertEqual(min(range(7), key=errors.__getitem__), 3,
                                 f"AVI frame at {seconds}s must match the source at the same timestamp")


if __name__ == "__main__":
    unittest.main()
