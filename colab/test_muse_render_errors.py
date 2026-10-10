"""Offline MuseTalk CLI/error regressions; no real model, media, or GPU claims."""

import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_colab_worker as worker


class MuseRenderErrorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.result_dir = self.root / "results"
        self.result_dir.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def process(self, text, *, muse=True, exit_code=0):
        script = f"import sys; print({text!r}); sys.exit({exit_code})"
        command = [sys.executable, "-u", "-c", script]
        if muse:
            # Python -c accepts these extra args without executing a model.
            # This exercises the actual subprocess error classifier unchanged.
            command.append("scripts.inference")
        with contextlib.redirect_stdout(io.StringIO()):
            return worker.run_streaming(command)

    def result(self, output="Results saved", returncode=0):
        return subprocess.CompletedProcess(["python", "-m", "scripts.inference"], returncode, output, "")

    def test_real_cli_exit_zero_with_swallowed_oom_is_failure(self):
        with self.assertRaisesRegex(worker.WorkerError, "out of memory"):
            self.process("Error occurred during processing: CUDA out of memory. Tried to allocate 256 MiB")

    def test_real_cli_exit_zero_with_nonmemory_task_error_is_failure(self):
        with self.assertRaisesRegex(worker.WorkerError, "invalid face coordinates"):
            self.process("Error occurred during processing: invalid face coordinates")

    def test_regular_nonmuse_process_is_not_reclassified_from_message(self):
        result = self.process("A diagnostic example says out of memory", muse=False)
        self.assertEqual(result.returncode, 0)

    def test_oom_cause_survives_long_diagnostic_tail(self):
        text = "Error occurred during processing: CUDA out of memory\n" + "allocator diagnostics\n" * 400
        with self.assertRaisesRegex(worker.WorkerError, "out of memory"):
            self.process(text)

    def test_nonzero_real_process_keeps_original_diagnostic(self):
        with self.assertRaisesRegex(worker.WorkerError, "custom renderer failure"):
            self.process("custom renderer failure", exit_code=9)

    def test_no_output_preserves_model_error_returned_with_zero(self):
        result = self.result("Loaded weights\nError occurred during processing: unexpected model dimensions")
        with self.assertRaisesRegex(worker.WorkerError, "unexpected model dimensions"):
            worker.select_musetalk_output(self.result_dir, set(), result, "No new video.")

    def test_previous_video_is_rejected_even_if_overwritten_in_place(self):
        old = self.result_dir / "old.mp4"
        old.write_bytes(b"old fixture")
        before = {old.resolve()}
        old.write_bytes(b"overwritten old path still requires a clean run directory")
        with self.assertRaisesRegex(worker.WorkerError, "No new video"):
            worker.select_musetalk_output(self.result_dir, before, self.result(), "No new video.")

    def test_empty_new_video_is_rejected(self):
        (self.result_dir / "empty.mp4").touch()
        with self.assertRaises(worker.WorkerError):
            worker.select_musetalk_output(self.result_dir, set(), self.result(), "No new video.")

    def test_only_new_nonempty_video_is_selected(self):
        old = self.result_dir / "old.mp4"
        fresh = self.result_dir / "new.mp4"
        old.write_bytes(b"old fixture")
        fresh.write_bytes(b"new fixture, stream validation occurs separately")
        self.assertEqual(worker.select_musetalk_output(self.result_dir, {old.resolve()}, self.result(), "No new video."), fresh)

    def test_task_failure_rejects_new_partial_file_too(self):
        (self.result_dir / "temp_partial.mp4").write_bytes(b"partial fixture")
        with self.assertRaisesRegex(worker.WorkerError, "out of memory"):
            worker.select_musetalk_output(self.result_dir, set(), self.result("Error occurred during processing: CUDA out of memory"), "No new video.")

    def test_empty_output_does_not_fabricate_success(self):
        with self.assertRaisesRegex(worker.WorkerError, "No new video"):
            worker.select_musetalk_output(self.result_dir, set(), self.result(""), "No new video.")

    def test_all_three_render_paths_preserve_swallowed_cli_cause(self):
        video, audio = self.root / "video.mp4", self.root / "audio.wav"
        video.write_bytes(b"mock video, probe is isolated")
        audio.write_bytes(b"mock audio, probe is isolated")
        talknet, muse = self.root / "TalkNet", self.root / "MuseTalk"
        normalized = talknet / "demo/video/pyavi/video.avi"
        normalized.parent.mkdir(parents=True)
        normalized.write_bytes(b"mock normalized frames")
        for relative in (
            "musetalkV15/musetalk.json", "musetalkV15/unet.pth", "sd-vae/config.json",
            "sd-vae/diffusion_pytorch_model.bin", "whisper/config.json", "whisper/pytorch_model.bin",
            "whisper/preprocessor_config.json", "dwpose/dw-ll_ucoco_384.pth",
            "face-parse-bisent/79999_iter.pth", "face-parse-bisent/resnet18-5c106cde.pth",
        ):
            weight = muse / "models" / relative
            weight.parent.mkdir(parents=True, exist_ok=True)
            weight.write_bytes(b"mock weight, never loaded")
        boxes = [(10, 10, 80, 80)] * 50
        segment = {"start_frame": 0, "end_frame": 50, "fps": 25.0, "boxes": boxes}
        media = {"duration": 2.0, "streams": [{"codec_type": "video", "width": 100, "height": 100}]}
        failure = self.result("Error occurred during processing: CUDA out of memory")
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.dict(worker.os.environ, {"DUBSTUDIO_TALKNET_ROOT": str(talknet), "DUBSTUDIO_MUSETALK_ROOT": str(muse)}))
            stack.enter_context(patch.object(worker, "ffprobe", return_value=media))
            stack.enter_context(patch.object(worker, "video_stream_duration", return_value=2.0))
            stack.enter_context(patch.object(worker, "video_frame_count", return_value=50))
            stack.enter_context(patch.object(worker, "resample_face_boxes", return_value=(boxes, "mock confirmed coordinates")))
            stack.enter_context(patch.object(worker, "run"))
            stack.enter_context(patch.object(worker, "run_streaming", return_value=failure))
            stack.enter_context(patch.object(worker, "write_progress"))
            stack.enter_context(patch.object(worker, "lipsync_coordinate_mode", return_value="musetalk"))
            stack.enter_context(patch.object(worker, "prepare_musetalk_media", return_value=(video, audio)))
            for renderer, analysis in (
                (worker.render_confirmed_face_segment, {"confirmed_face_segment": segment}),
                (worker.render_confirmed_talknet_segments, {"confirmed_face_segments": [segment]}),
            ):
                with self.subTest(renderer=renderer.__name__), self.assertRaisesRegex(worker.WorkerError, "out of memory"):
                    renderer(video, audio, self.root, analysis, self.root / "test.log", True)
            with self.subTest(renderer="musetalk"), self.assertRaisesRegex(worker.WorkerError, "out of memory"):
                worker.musetalk(video, audio, self.root, {}, self.root / "test.log", True, True)


if __name__ == "__main__":
    unittest.main()
