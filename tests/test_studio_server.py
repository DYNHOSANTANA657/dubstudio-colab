"""Local HTTP integration tests for Studio v2: no GPU or account writes."""

import hashlib
import http.client
import io
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import studio_server as server
from pipeline import JobStore


class QuietHandler(server.Handler):
    def log_message(self, fmt, *args):
        pass


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class StudioServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory()
        root = Path(cls.shared.name)
        cls.video = root / "sample.mp4"
        cls.audio = root / "sample.wav"
        subprocess.run([
            shutil.which("ffmpeg"), "-y", "-f", "lavfi", "-i", "color=c=green:s=160x90:r=25",
            "-f", "lavfi", "-i", "sine=frequency=450:sample_rate=48000", "-t", "3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(cls.video),
        ], check=True, capture_output=True)
        subprocess.run([shutil.which("ffmpeg"), "-y", "-i", str(cls.video), "-vn", "-t", "3", str(cls.audio)], check=True, capture_output=True)
        cls.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=2)
        cls.shared.cleanup()

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.patches = [
            patch.object(server, "RUNTIME", self.root),
            patch.object(server, "JOBS_DIR", self.root / "jobs"),
            patch.object(server, "CONFIG_PATH", self.root / "config.json"),
            patch.object(server, "STORE", JobStore()),
            patch.object(server, "JOB_LOCKS", {}),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.folder.cleanup()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=30)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        value = response.read()
        metadata = dict(response.getheaders())
        code = response.status
        connection.close()
        return code, metadata, value

    def json_request(self, method, path, payload=None, headers=None):
        fields = {"Content-Type": "application/json", **(headers or {})}
        body = json.dumps(payload).encode() if payload is not None and method not in {"GET", "DELETE"} else None
        code, metadata, body = self.request(method, path, body, fields)
        return code, metadata, json.loads(body)

    def multipart(self, fields, files):
        boundary = f"studio-test-{uuid.uuid4().hex}"
        chunks = []
        for key, value in fields.items():
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
        for key, (filename, data) in files.items():
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode())
            chunks.append(data)
            chunks.append(b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        return b"".join(chunks), {"Content-Type": f"multipart/form-data; boundary={boundary}"}

    def create_job(self, operation="dub_video", languages=None, options=None):
        fields = {
            "operation": operation, "voice_clone_consent": "true",
            "timing_mode": "source_video",
            "target_languages": json.dumps(languages or ["pl-PL"]),
            "text": "Olá mundo.", "review_mode": "before_voice",
        }
        fields.update(options or {})
        files = {}
        if operation != "text_to_voice":
            files["video"] = ("../../sample.mp4", self.video.read_bytes())
        if operation == "lipsync_video":
            files["audio"] = ("audio.wav", self.audio.read_bytes())
        if operation == "text_to_voice":
            files["reference"] = ("reference.wav", self.audio.read_bytes())
        body, headers = self.multipart(fields, files)
        code, _, payload = self.request("POST", "/api/jobs", body, headers)
        self.assertEqual(code, 201, payload)
        job = json.loads(payload)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            _, _, job = self.json_request("GET", f"/api/jobs/{job['id']}")
            if job["status"] not in {"queued", "preparing_colab"}:
                break
            time.sleep(0.025)
        self.assertEqual(job["status"], "awaiting_colab", job)
        return job

    def review_for(self, job):
        manifest = server.STORE.get(job["id"]).metadata["studio"]["manifest"]
        return {
            "schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": job["id"], "input_sha256": manifest["input_sha256"],
            "source_language": "pt", "translations": {language: [
                {"id": 1, "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00", "text": "Olá", "translated_text": "Cześć"},
            ] for language in manifest["target_languages"]},
        }

    def wait_for_preparation(self, job):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            code, _, current = self.json_request("GET", f"/api/jobs/{job['id']}")
            self.assertEqual(code, 200)
            if current["status"] not in {"queued", "preparing_colab"}:
                return current
            time.sleep(0.025)
        self.fail("Local preparation did not finish")

    def zip_bytes(self, files):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        return output.getvalue()

    def result_for(self, job, fail_language=None):
        manifest = server.STORE.get(job["id"]).metadata["studio"]["manifest"]
        result = {"schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": job["id"], "input_sha256": manifest["input_sha256"], "outputs": [],
                  "operation": manifest["operation"], "input_checksums": {role: item["sha256"] for role, item in manifest["inputs"].items()}}
        contents = {}
        for language in manifest["target_languages"]:
            if language == fail_language:
                result["outputs"].append({"language": language, "status": "failed", "error": "Insufficient GPU memory"})
                continue
            base = f"outputs/{language}"
            report = {"schema_version": "2.0", "pipeline_version": "3.0.0", "job_id": job["id"], "input_sha256": manifest["input_sha256"], "target_language": language,
                      "operation": manifest["operation"], "input_checksums": {role: item["sha256"] for role, item in manifest["inputs"].items()},
                      "summary": {"duration": 3.0}, "voice_clone": {"required": True, "confirmed": True, "authorized": True},
                      "lipsync": {"required": True, "confirmed": True, "applied": True, "ambiguity": [], "confirmed_segments": [{"start": 0, "end": 3, "coordinate_source": "TalkNet"}]},
                      "preserved_no_speaker_visual_segments": []}
            segments = {"target_language": language, "segments": self.review_for(job)["translations"][language]}
            files = {"video": f"{base}/video.mp4", "report": f"{base}/report.json", "segments": f"{base}/segments.json", "original_srt": f"{base}/original.srt", "translation_srt": f"{base}/translation.srt"}
            contents.update({files["video"]: self.video.read_bytes(), files["report"]: json.dumps(report).encode(), files["segments"]: json.dumps(segments).encode(), files["original_srt"]: b"1\n00:00:00,000 --> 00:00:01,000\nOla\n", files["translation_srt"]: b"1\n00:00:00,000 --> 00:00:01,000\nCzesc\n"})
            result["outputs"].append({"language": language, "status": "complete", "files": files, "checksums": {name: hashlib.sha256(contents[name]).hexdigest() for name in files.values()}})
        contents["result.json"] = json.dumps(result).encode()
        return self.zip_bytes(contents)

    def test_three_operations_create_packages_and_never_publish_unvalidated_media(self):
        for operation in ("dub_video", "lipsync_video", "text_to_voice"):
            with self.subTest(operation=operation):
                job = self.create_job(operation)
                code, headers, data = self.request("GET", f"/api/jobs/{job['id']}/colab-package")
                self.assertEqual(code, 200)
                self.assertIn("attachment", headers["Content-Disposition"])
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["operation"], operation)
                    self.assertTrue(manifest["authorization"]["voice_clone_confirmed"])
                    if operation == "text_to_voice":
                        self.assertEqual(set(manifest["inputs"]), {"reference"})
                self.assertFalse(job["has_output"])
                code, _, _ = self.request("GET", f"/api/jobs/{job['id']}/outputs/pl-PL/video")
                self.assertEqual(code, 409)

    def test_missing_required_uploads_are_rejected_before_creating_any_directory(self):
        cases = (("dub_video", {}), ("lipsync_video", {"video": ("video.mp4", self.video.read_bytes())}), ("text_to_voice", {}))
        for operation, files in cases:
            with self.subTest(operation=operation), patch.object(server.tempfile, "mkdtemp", wraps=server.tempfile.mkdtemp) as temporary:
                body, headers = self.multipart({"operation": operation, "voice_clone_consent": "true", "target_languages": '["pl-PL"]', "text": "Olá."}, files)
                code, _, payload = self.request("POST", "/api/jobs", body, headers)
                self.assertEqual(code, 400, payload)
                temporary.assert_not_called()
                self.assertEqual(server.STORE.list(), [])
                self.assertEqual(list((self.root / "jobs").glob("*")), [])

    def test_empty_or_invalid_media_rolls_back_only_its_unpublished_upload(self):
        original_hashes = (hashlib.sha256(self.video.read_bytes()).hexdigest(), hashlib.sha256(self.audio.read_bytes()).hexdigest())
        cases = (
            ("dub_video", {"video": ("video.mp4", b"")}),
            ("dub_video", {"video": ("video.mp4", b"not video")}),
            ("lipsync_video", {"video": ("video.mp4", self.video.read_bytes()), "audio": ("audio.wav", b"not audio")}),
            ("text_to_voice", {"reference": ("voice.wav", b"not audio")}),
        )
        for operation, files in cases:
            with self.subTest(operation=operation, sizes=[len(item[1]) for item in files.values()]):
                body, headers = self.multipart({"operation": operation, "voice_clone_consent": "true", "target_languages": '["pl-PL"]', "text": "Olá."}, files)
                code, _, payload = self.request("POST", "/api/jobs", body, headers)
                self.assertEqual(code, 400, payload)
                self.assertEqual(server.STORE.list(), [])
                self.assertEqual(list((self.root / "jobs").glob("*")), [])
        self.assertEqual(original_hashes, (hashlib.sha256(self.video.read_bytes()).hexdigest(), hashlib.sha256(self.audio.read_bytes()).hexdigest()))

    def test_probe_or_staged_record_failure_preserves_http_error_and_publishes_nothing(self):
        body, headers = self.multipart({"voice_clone_consent": "true"}, {"video": ("video.mp4", self.video.read_bytes())})
        for target, error in (("probe_video", server.PipelineError("synthetic probe failure")), ("atomic_json", OSError("synthetic record failure"))):
            with self.subTest(target=target), patch.object(server, target, side_effect=error):
                code, _, payload = self.request("POST", "/api/jobs", body, headers)
                self.assertEqual(code, 400, payload)
                self.assertEqual(json.loads(payload)["error"], str(error))
                self.assertEqual(server.STORE.list(), [])
                self.assertEqual(list((self.root / "jobs").glob("*")), [])

    def test_failed_upload_commit_removes_staging_without_exposing_a_project(self):
        body, headers = self.multipart({"voice_clone_consent": "true"}, {"video": ("video.mp4", self.video.read_bytes())})
        with patch.object(Path, "rename", side_effect=OSError("synthetic commit failure")):
            code, _, payload = self.request("POST", "/api/jobs", body, headers)
        self.assertEqual(code, 400, payload)
        self.assertEqual(json.loads(payload)["error"], "synthetic commit failure")
        self.assertEqual(server.STORE.list(), [])
        self.assertEqual(list((self.root / "jobs").glob("*")), [])

    def test_upload_id_collision_never_overwrites_registered_or_unregistered_files(self):
        job = self.create_job()
        before = {str(path.relative_to(self.root)): path.read_bytes() for path in (self.root / "jobs").rglob("*") if path.is_file()}
        body, headers = self.multipart({"voice_clone_consent": "true"}, {"video": ("video.mp4", self.video.read_bytes())})
        with patch.object(server.secrets, "token_hex", return_value=job["id"]):
            code, _, payload = self.request("POST", "/api/jobs", body, headers)
        self.assertEqual(code, 400, payload)
        self.assertEqual({str(path.relative_to(self.root)): path.read_bytes() for path in (self.root / "jobs").rglob("*") if path.is_file()}, before)
        orphan = self.root / "jobs" / "111111111111"
        orphan.mkdir()
        (orphan / "original.txt").write_bytes(b"existing unregistered file")
        with patch.object(server.secrets, "token_hex", return_value=orphan.name):
            code, _, payload = self.request("POST", "/api/jobs", body, headers)
        self.assertEqual(code, 400, payload)
        self.assertEqual((orphan / "original.txt").read_bytes(), b"existing unregistered file")
        self.assertEqual(len(server.STORE.list()), 1)

    def test_upload_rollback_requires_exact_root_name_and_creation_identity(self):
        job_id = "222222222222"
        root = self.root / "jobs"
        root.mkdir()
        staging = root / f".upload-{job_id}-synthetic"
        staging.mkdir()
        (staging / "input.mp4").write_bytes(b"new upload bytes")
        created = staging.stat()
        identity = (created.st_dev, created.st_ino)
        active = root / job_id
        active.mkdir()
        (active / "original.mp4").write_bytes(b"existing project")
        outside = self.root / "outside"
        outside.mkdir()
        cases = ((active, job_id, identity), (outside, job_id, identity), (staging, "333333333333", identity), (staging, job_id, (created.st_dev, created.st_ino + 1)))
        for path, owner, stamp in cases:
            with self.subTest(path=path.name, owner=owner), patch.object(server.shutil, "rmtree") as delete:
                with self.assertRaises(ValueError):
                    server.discard_owned_upload_stage(path, owner, stamp)
                delete.assert_not_called()
        with patch.object(Path, "is_symlink", return_value=True), patch.object(server.shutil, "rmtree") as delete:
            with self.assertRaisesRegex(ValueError, "link"):
                server.discard_owned_upload_stage(staging, job_id, identity)
            delete.assert_not_called()
        server.discard_owned_upload_stage(staging, job_id, identity)
        self.assertFalse(staging.exists())
        self.assertEqual((active / "original.mp4").read_bytes(), b"existing project")
        self.assertTrue(outside.is_dir())

    def test_successful_upload_commits_final_paths_and_no_staging_leaks(self):
        for operation in ("dub_video", "lipsync_video", "text_to_voice"):
            with self.subTest(operation=operation):
                public = self.create_job(operation)
                job = server.STORE.get(public["id"])
                directory = server.managed_project_directory("jobs", job.id)
                self.assertEqual(job.directory, directory)
                self.assertTrue(job.input_path.is_file())
                self.assertTrue(job.input_path.resolve().is_relative_to(directory.resolve()))
                for key in ("audio_path", "reference_path"):
                    if key in job.options:
                        file = Path(job.options[key])
                        self.assertTrue(file.is_file())
                        self.assertEqual(file.parent, directory)
                record = json.loads((directory / "job.json").read_text(encoding="utf-8"))
                self.assertNotIn(".upload-", json.dumps(record))
                self.assertEqual(list((self.root / "jobs").glob(".upload-*")), [])

    def test_review_and_approved_package_roundtrip_through_http(self):
        job = self.create_job()
        review = self.review_for(job)
        body, headers = self.multipart({}, {"review": ("review.zip", self.zip_bytes({"review.json": json.dumps(review).encode()}))})
        code, _, data = self.request("POST", f"/api/jobs/{job['id']}/review", body, headers)
        self.assertEqual(code, 200, data)
        self.assertEqual(json.loads(data)["status"], "review_required")
        review["translations"]["pl-PL"][0]["translated_text"] = "Dzień dobry"
        code, _, data = self.json_request("POST", f"/api/jobs/{job['id']}/review-approved", review)
        self.assertEqual(code, 200, data)
        self.assertEqual(data["status"], "review_approved")
        code, _, package = self.request("GET", data["download_url"])
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(package)) as archive:
            self.assertIn("manifest.json", archive.namelist())
            self.assertEqual(json.loads(archive.read("review-approved.json"))["translations"]["pl-PL"][0]["translated_text"], "Dzień dobry")

    def test_verified_partial_results_and_missing_downloads(self):
        job = self.create_job(languages=["pl-PL", "en-US"])
        body, headers = self.multipart({}, {"result": ("result.zip", self.result_for(job, fail_language="en-US"))})
        code, _, data = self.request("POST", f"/api/jobs/{job['id']}/colab-result", body, headers)
        self.assertEqual(code, 200, data)
        payload = json.loads(data)
        self.assertEqual(payload["status"], "partial")
        self.assertEqual(len(payload["metadata"]["outputs"]), 2)
        code, headers, video = self.request("GET", f"/api/jobs/{job['id']}/outputs/pl-PL/video")
        self.assertEqual(code, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertEqual(video, self.video.read_bytes())
        code, _, _ = self.request("GET", f"/api/jobs/{job['id']}/outputs/en-US/video")
        self.assertEqual(code, 409)

    def test_invalid_import_preserves_existing_validated_version(self):
        job = self.create_job()
        body, headers = self.multipart({}, {"result": ("result.zip", self.result_for(job))})
        self.assertEqual(self.request("POST", f"/api/jobs/{job['id']}/colab-result", body, headers)[0], 200)
        previous = server.STORE.get(job["id"]).metadata["studio_results"]
        unsafe = self.zip_bytes({"../outside.txt": b"x"})
        body, headers = self.multipart({}, {"result": ("result.zip", unsafe)})
        self.assertEqual(self.request("POST", f"/api/jobs/{job['id']}/colab-result", body, headers)[0], 400)
        self.assertEqual(server.STORE.get(job["id"]).metadata["studio_results"], previous)
        self.assertFalse((self.root / "outside.txt").exists())

    def test_host_and_origin_defend_against_remote_requests_and_rebinding(self):
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}, {"Origin": "null"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("GET", "/api/health", headers=headers)[0], 403)
                self.assertEqual(self.json_request("POST", "/api/settings", headers=headers)[0], 403)
        local_host = f"localhost:{self.httpd.server_port}"
        self.assertEqual(self.request("GET", "/api/health", headers={"Host": local_host, "Origin": f"http://{local_host}"})[0], 200)

    def test_second_listener_cannot_bind_or_disrupt_the_original_server(self):
        descriptor = self.httpd.fileno()
        self.assertFalse(self.httpd.allow_reuse_address)
        self.assertFalse(self.httpd.allow_reuse_port)
        exclusive_option = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive_option is not None:
            self.assertEqual(self.httpd.socket.getsockopt(socket.SOL_SOCKET, exclusive_option), 1)
        with self.assertRaises(OSError):
            server.ThreadingHTTPServer(("127.0.0.1", self.httpd.server_port), QuietHandler)
        self.assertEqual(self.httpd.fileno(), descriptor)
        code, _, payload = self.json_request("GET", "/api/health")
        self.assertEqual(code, 200)
        self.assertEqual(payload["version"], server.VERSION)
        self.assertTrue(self.thread.is_alive())

    def test_preview_ranges_and_missing_approved_package(self):
        job = self.create_job()
        path = f"/api/jobs/{job['id']}/source"
        contents = self.video.read_bytes()
        code, headers, data = self.request("GET", path, headers={"Range": "bytes=3-12"})
        self.assertEqual(code, 206)
        self.assertEqual(data, contents[3:13])
        self.assertEqual(headers["Content-Range"], f"bytes 3-12/{len(contents)}")
        self.assertEqual(self.request("GET", path, headers={"Range": "bytes=-8"})[2], contents[-8:])
        self.assertEqual(self.request("GET", path, headers={"Range": "bytes=999999999-"})[0], 416)
        self.assertEqual(self.request("GET", path, headers={"Range": "bytes=-0"})[0], 416)
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/review-approved")[0], 409)

    def test_archive_is_recoverable_and_not_restored_as_active(self):
        job = self.create_job()
        directory = server.STORE.get(job["id"]).directory
        code, _, data = self.json_request("DELETE", f"/api/jobs/{job['id']}")
        self.assertEqual(code, 200)
        self.assertTrue(data["recoverable"])
        self.assertTrue((directory / "original.mp4").exists())
        self.assertTrue((directory / "pacote-colab.zip").exists())
        server.restore_jobs()
        self.assertIsNone(server.STORE.get(job["id"]))

    def test_delete_requires_confirmation_and_preserves_external_original_and_download(self):
        job = self.create_job()
        directory = server.STORE.get(job["id"]).directory
        external_original = self.root / "original-do-usuario.mp4"
        external_download = self.root / "download-do-usuario.mp4"
        external_original.write_bytes(self.video.read_bytes())
        external_download.write_bytes(b"existing downloaded result")
        originals = {path: path.read_bytes() for path in (external_original, external_download)}
        for payload in ({}, {"confirm": False}, {"confirm": "true"}):
            code, _, _ = self.json_request("POST", f"/api/jobs/{job['id']}/trash", payload)
            self.assertEqual(code, 400)
            self.assertTrue(directory.exists())
            self.assertIsNotNone(server.STORE.get(job["id"]))
        code, _, result = self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})
        self.assertEqual(code, 200, result)
        self.assertTrue(result["recoverable"])
        self.assertFalse(directory.exists())
        trashed = self.root / "trash" / job["id"]
        self.assertEqual((trashed / "original.mp4").read_bytes(), self.video.read_bytes())
        self.assertTrue((trashed / "pacote-colab.zip").is_file())
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/source")[0], 404)
        self.assertEqual(self.json_request("GET", "/api/jobs")[2]["jobs"], [])
        trash = self.json_request("GET", "/api/trash")[2]["jobs"]
        self.assertEqual([item["id"] for item in trash], [job["id"]])
        self.assertEqual(trash[0]["status"], "deleted")
        self.assertNotIn("outputs", trash[0]["metadata"])
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)

    def test_trash_restore_preserves_results_after_restart_without_external_execution(self):
        job = self.create_job()
        body, headers = self.multipart({}, {"result": ("result.zip", self.result_for(job))})
        self.assertEqual(self.request("POST", f"/api/jobs/{job['id']}/colab-result", body, headers)[0], 200)
        before = self.request("GET", f"/api/jobs/{job['id']}/outputs/pl-PL/video")[2]
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
        server.STORE = JobStore()
        server.restore_jobs()
        self.assertIsNone(server.STORE.get(job["id"]))
        with patch.object(server, "prepare_job", side_effect=AssertionError("restoring cannot execute")):
            code, _, restored = self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})
        self.assertEqual(code, 200, restored)
        self.assertEqual(restored["status"], "complete")
        self.assertTrue(restored["has_output"])
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/outputs/pl-PL/video")[2], before)
        self.assertEqual(self.json_request("GET", "/api/trash")[2]["jobs"], [])
        server.STORE = JobStore()
        server.restore_jobs()
        self.assertEqual(server.STORE.get(job["id"]).status, "complete")

    def test_trash_restore_keeps_text_reference_and_lipsync_audio_paths_valid(self):
        for operation, suffix in (("text_to_voice", "reference"), ("lipsync_video", "audio")):
            with self.subTest(operation=operation):
                job = self.create_job(operation)
                endpoint = f"/api/jobs/{job['id']}/{suffix}"
                before = self.request("GET", endpoint)[2]
                self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
                self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {})[0], 400)
                self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})[0], 200)
                self.assertEqual(self.request("GET", endpoint)[2], before)

    def test_trash_and_restore_collisions_never_overwrite_other_files(self):
        job = self.create_job()
        directory = server.STORE.get(job["id"]).directory
        collision = self.root / "trash" / job["id"]
        collision.mkdir(parents=True)
        sentinel = collision / "sentinel.txt"
        sentinel.write_bytes(b"preserve collision")
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 400)
        self.assertEqual(sentinel.read_bytes(), b"preserve collision")
        self.assertFalse((directory / server.TRASH_RECORD).exists())
        # Remove only our empty test collision after proving no overwrite.
        sentinel.unlink()
        collision.rmdir()
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
        directory.mkdir()
        sentinel = directory / "sentinel.txt"
        sentinel.write_bytes(b"active destination collision")
        self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})[0], 400)
        self.assertEqual(sentinel.read_bytes(), b"active destination collision")
        self.assertTrue((collision / "original.mp4").is_file())

    def test_failed_trash_move_leaves_project_active_and_originals_unchanged(self):
        job = self.create_job()
        directory = server.STORE.get(job["id"]).directory
        with patch.object(Path, "rename", side_effect=PermissionError("test locked directory")):
            code, _, _ = self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})
        self.assertEqual(code, 400)
        self.assertIsNotNone(server.STORE.get(job["id"]))
        self.assertFalse((directory / server.TRASH_RECORD).exists())
        self.assertEqual((directory / "original.mp4").read_bytes(), self.video.read_bytes())

    def test_interrupted_trash_move_is_hidden_at_startup_and_can_be_restored(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        server.atomic_json(internal.directory / server.TRASH_RECORD,
                           {"schema_version": 1, "id": job["id"], "deleted_at": time.time(), "job": internal.public()})
        server.STORE = JobStore()
        server.restore_jobs()
        self.assertIsNone(server.STORE.get(job["id"]))
        self.assertEqual(self.json_request("GET", "/api/trash")[2]["jobs"][0]["id"], job["id"])
        self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})[0], 200)
        self.assertFalse((internal.directory / server.TRASH_RECORD).exists())
        self.assertIsNotNone(server.STORE.get(job["id"]))

    def test_deletion_refuses_outside_paths_and_stale_workers_cannot_recreate_projects(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        correct_directory = internal.directory
        external = self.root / "not-a-project"
        external.mkdir()
        sentinel = external / "sentinel.txt"
        sentinel.write_bytes(b"do not move outside target")
        internal.directory = external
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 400)
        self.assertEqual(sentinel.read_bytes(), b"do not move outside target")
        self.assertTrue(correct_directory.exists())
        internal.directory = correct_directory
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
        with patch.object(server, "prepare_studio_package", side_effect=AssertionError("stale job must not execute")):
            server.prepare_job(internal)
        self.assertFalse(correct_directory.exists())
        self.assertEqual(self.request("POST", "/api/trash/../../outside/restore", b"{}")[0], 404)

    def test_rejected_post_small_body_delivers_status_without_connection_reset(self):
        body = b'{"confirm":true}'
        for endpoint, status in (
            ("/api/trash/../../outside/restore", 404),
            ("/api/trash/%2e%2e/%2e%2e/outside/restore", 404),
            ("/api/jobs/not-a-valid-id/trash", 404),
            ("/api/jobs/000000000000/external-started", 404),
            ("/api/jobs/000000000000/colab-result", 404),
            ("/api/runpod/retry", 410),
        ):
            with self.subTest(endpoint=endpoint):
                code, headers, response = self.request("POST", endpoint, body,
                    {"Content-Type": "application/json"})
                self.assertEqual(code, status, response)
                self.assertEqual(headers["Connection"], "close")
                self.assertIn("error", json.loads(response))
        for denied in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}):
            with self.subTest(denied=denied):
                self.assertEqual(self.request("POST", "/api/settings", body, denied)[0], 403)
        self.assertEqual(server.STORE.list(), [])
        self.assertFalse((self.root / "outside").exists())

    def test_rejected_post_body_drain_does_not_wait_for_incomplete_client(self):
        # The client deliberately announces more than it sends. No GPU, project
        # or file operation should happen while the rejection waits briefly.
        with patch.object(server, "REJECTED_BODY_TIMEOUT", 0.05):
            started = time.monotonic()
            code, _, response = self.request("POST", "/api/trash/../../outside/restore",
                b"{}", {"Content-Length": "64"})
            elapsed = time.monotonic() - started
        self.assertEqual(code, 404, response)
        self.assertLess(elapsed, 1.5, "A rejected incomplete upload must not hang the request")
        self.assertEqual(server.STORE.list(), [])

    def test_rejected_post_body_drain_refuses_large_or_ambiguous_framing(self):
        for raw in (
            f"Content-Length: {server.MAX_REJECTED_BODY + 1}\r\n",
            "Content-Length: -1\r\n", "Content-Length: invalid\r\n",
            "Content-Length: " + "9" * 5000 + "\r\n",
            "Content-Length: 2\r\nContent-Length: 3\r\n",
            "Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
        ):
            with self.subTest(headers=raw[:80]):
                handler = object.__new__(server.Handler)
                handler.headers = http.client.parse_headers(io.BytesIO((raw + "\r\n").encode()))
                handler.rfile, handler.connection = MagicMock(), MagicMock()
                self.assertFalse(handler.discard_rejected_body())
                handler.rfile.read1.assert_not_called()
                handler.connection.settimeout.assert_not_called()

    def test_rejected_post_body_drain_is_exact_and_restores_socket_timeout(self):
        handler = object.__new__(server.Handler)
        handler.headers = http.client.parse_headers(io.BytesIO(
            f"Content-Length: {server.MAX_REJECTED_BODY}\r\n\r\n".encode()))
        handler.rfile = io.BytesIO(b"x" * server.MAX_REJECTED_BODY + b"not-this-body")
        handler.connection = MagicMock()
        handler.connection.gettimeout.return_value = 7.0
        self.assertTrue(handler.discard_rejected_body())
        self.assertEqual(handler.rfile.read(), b"not-this-body")
        handler.connection.settimeout.assert_called_with(7.0)

    def test_rejected_multilanguage_result_in_trash_returns_http404_without_publishing(self):
        job = self.create_job(languages=["pl-PL", "es-ES"])
        result = self.result_for(job)
        internal = server.STORE.get(job["id"])
        input_bytes = internal.input_path.read_bytes()
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
        trashed = self.root / "trash" / job["id"]
        record = (trashed / server.TRASH_RECORD).read_bytes()
        body, headers = self.multipart({}, {"result": ("correct-result.zip", result)})
        self.assertGreater(len(body), server.MAX_REJECTED_BODY)
        self.assertLessEqual(len(body), server.MAX_UPLOAD)
        code, response_headers, response = self.request("POST", f"/api/jobs/{job['id']}/colab-result", body, headers)
        self.assertEqual(code, 404, response)
        self.assertEqual(response_headers["Connection"], "close")
        self.assertIn("Projeto", json.loads(response)["error"])
        self.assertIsNone(server.STORE.get(job["id"]))
        self.assertFalse((self.root / "jobs" / job["id"]).exists())
        self.assertFalse((trashed / "studio-results").exists())
        self.assertEqual((trashed / "original.mp4").read_bytes(), input_bytes)
        self.assertEqual((trashed / server.TRASH_RECORD).read_bytes(), record)

    def test_rejected_multipart_drain_is_limited_to_trusted_import_routes(self):
        import_path = "/api/jobs/000000000000/colab-result"
        cases = [
            (True, import_path, "multipart/form-data; boundary=x", server.MAX_UPLOAD + 1),
            (False, import_path, "multipart/form-data; boundary=x", server.MAX_REJECTED_BODY + 1),
            (True, "/api/settings", "multipart/form-data; boundary=x", server.MAX_REJECTED_BODY + 1),
            (True, "/api/jobs/invalid-id/colab-result", "multipart/form-data; boundary=x", server.MAX_REJECTED_BODY + 1),
            (True, import_path, "application/json", server.MAX_REJECTED_BODY + 1),
        ]
        for trusted, path, kind, size in cases:
            with self.subTest(trusted=trusted, path=path, kind=kind, size=size):
                handler = object.__new__(server.Handler)
                handler._trusted_local_request, handler.path = trusted, path
                handler.headers = http.client.parse_headers(io.BytesIO(
                    f"Content-Length: {size}\r\nContent-Type: {kind}\r\n\r\n".encode()))
                handler.rfile, handler.connection = MagicMock(), MagicMock()
                self.assertFalse(handler.discard_rejected_body())
                handler.rfile.read1.assert_not_called()
                handler.connection.settimeout.assert_not_called()

    def test_rejected_multipart_drain_streams_small_chunks_under_total_deadline(self):
        handler = object.__new__(server.Handler)
        size = server.MAX_REJECTED_BODY * 4
        handler._trusted_local_request, handler.path = True, "/api/jobs/000000000000/review"
        handler.headers = http.client.parse_headers(io.BytesIO(
            f"Content-Length: {size}\r\nContent-Type: multipart/form-data; boundary=x\r\n\r\n".encode()))
        stream = io.BytesIO(b"x" * size + b"outside-framed-body")
        handler.rfile, handler.connection = MagicMock(wraps=stream), MagicMock()
        handler.connection.gettimeout.return_value = 7.0
        self.assertTrue(handler.discard_rejected_body())
        self.assertEqual(stream.read(), b"outside-framed-body")
        self.assertTrue(all(call.args[0] <= 16 * 1024 for call in handler.rfile.read1.call_args_list))
        handler.connection.settimeout.assert_called_with(7.0)
        handler.rfile = MagicMock(wraps=io.BytesIO(b"x" * size))
        with patch.object(server.time, "monotonic", side_effect=[0.0, 0.1, 1.0]):
            self.assertFalse(handler.discard_rejected_body())
        self.assertEqual(handler.rfile.read1.call_count, 1)
        handler.connection.settimeout.assert_called_with(7.0)

    def test_rejected_incomplete_multipart_keeps_deadline_and_returns_status(self):
        with patch.object(server, "REJECTED_BODY_TIMEOUT", 0.05):
            started = time.monotonic()
            code, _, response = self.request("POST", "/api/jobs/000000000000/colab-result", b"{}",
                {"Content-Length": str(server.MAX_REJECTED_BODY * 4), "Content-Type": "multipart/form-data; boundary=x"})
            elapsed = time.monotonic() - started
        self.assertEqual(code, 404, response)
        self.assertLess(elapsed, 1.5)
        self.assertEqual(server.STORE.list(), [])
        self.assertFalse(self.root.joinpath("jobs").exists())

    def test_trash_does_not_cancel_processing_colab_and_local_busy_states_are_rejected(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        for status in ("queued", "preparing_colab", "validating"):
            internal.status = status
            self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 400)
        internal.status = "processing_colab"
        code, _, response = self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})
        self.assertEqual(code, 200, response)
        self.assertIn("não é cancelada", response["message"])
        self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})[2]["status"], "processing_colab")

    def test_trash_refuses_misconfigured_storage_and_malformed_restore_record(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        original_directory = internal.directory
        outside = self.root / "elsewhere"
        outside.mkdir()
        with patch.object(server, "JOBS_DIR", outside):
            self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 400)
        self.assertTrue(original_directory.exists())
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/trash", {"confirm": True})[0], 200)
        record = self.root / "trash" / job["id"] / server.TRASH_RECORD
        data = json.loads(record.read_text(encoding="utf-8"))
        data["id"] = "ffffffffffff"
        record.write_text(json.dumps(data), encoding="utf-8")
        self.assertEqual(self.json_request("POST", f"/api/trash/{job['id']}/restore", {"confirm": True})[0], 400)
        self.assertFalse(original_directory.exists())
        self.assertTrue((record.parent / "original.mp4").is_file())
        self.assertEqual(self.json_request("GET", "/api/trash")[2]["jobs"], [])

    def test_restore_preserves_ready_and_text_projects_without_external_execution(self):
        jobs = [self.create_job(), self.create_job("text_to_voice")]
        server.STORE = JobStore()
        with patch.object(server, "prepare_job", side_effect=AssertionError("must not prepare on restore")):
            server.restore_jobs()
        for job in jobs:
            restored = server.STORE.get(job["id"])
            self.assertIsNotNone(restored)
            self.assertEqual(restored.status, "awaiting_colab")

    def test_public_settings_do_not_return_legacy_secrets_or_commands(self):
        secrets_payload = {"runpod": {"api_key": "rp_secret", "api_key_protected": "protected_secret", "enabled": True}, "commands": {"tts": "hidden-command"}, "colab": {"notebook_url": ""}}
        (self.root / "config.json").write_text(json.dumps(secrets_payload), encoding="utf-8")
        code, _, body = self.request("GET", "/api/settings")
        self.assertEqual(code, 200)
        for secret in (b"rp_secret", b"protected_secret", b"hidden-command", b"runpod"):
            self.assertNotIn(secret, body)

    def test_media_paths_stay_confined_to_the_project(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        external = self.root / "external.wav"
        shutil.copy2(self.audio, external)
        internal.options["reference_path"] = str(external)
        internal.options["audio_path"] = str(external)
        for suffix in ("reference", "audio"):
            with self.subTest(suffix=suffix):
                self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/{suffix}")[0], 403)
        internal.input_path = external
        self.assertEqual(self.request("GET", f"/api/jobs/{job['id']}/source")[0], 403)
        self.assertEqual(self.request("GET", "/../studio_server.py")[0], 404)
        self.assertEqual(self.request("GET", "/%2e%2e/studio_server.py")[0], 404)

    def test_restore_rejects_id_mismatch_and_malformed_records(self):
        jobs = [self.create_job(), self.create_job()]
        for index, job in enumerate(jobs):
            record = server.STORE.get(job["id"]).directory / "job.json"
            content = json.loads(record.read_text(encoding="utf-8"))
            if index == 0:
                content["id"] = "000000000000"
            else:
                content["metadata"] = ["not-a-map"]
            record.write_text(json.dumps(content), encoding="utf-8")
        server.STORE = JobStore()
        server.restore_jobs()
        self.assertEqual(server.STORE.list(), [])

    def test_malformed_config_does_not_break_the_local_api(self):
        for value in ([], {"colab": [], "studio": "bad", "limits": None}):
            with self.subTest(value=value):
                (self.root / "config.json").write_text(json.dumps(value), encoding="utf-8")
                code, _, payload = self.json_request("GET", "/api/settings")
                self.assertEqual(code, 200)
                self.assertEqual(payload["config"]["colab"]["notebook_url"], server.DEFAULT_CONFIG["colab"]["notebook_url"])
                self.assertTrue(payload["config"]["studio"]["preserve_ambience"])

    def test_lipsync_option_is_explicit_for_dubbing_and_fixed_for_other_operations(self):
        for operation, options, expected in (
            ("dub_video", {}, True),
            ("dub_video", {"require_lipsync": "false"}, False),
            ("lipsync_video", {"require_lipsync": "false"}, True),
            ("text_to_voice", {"require_lipsync": "true"}, False),
        ):
            with self.subTest(operation=operation, options=options):
                job = self.create_job(operation, options=options)
                manifest = server.STORE.get(job["id"]).metadata["studio"]["manifest"]
                self.assertIs(manifest["options"]["require_lipsync"], expected)
                self.assertIs(job["options"]["require_lipsync"], expected)

    def test_settings_booleans_are_not_coerced_from_strings(self):
        for flag in ("preserve_ambience", "cache_on_drive"):
            with self.subTest(flag=flag):
                code, _, _ = self.json_request("POST", "/api/settings", {"studio": {flag: "false"}})
                self.assertEqual(code, 400)
                code, _, payload = self.json_request("POST", "/api/settings", {"studio": {flag: False}})
                self.assertEqual(code, 200)
                self.assertIs(payload["config"]["studio"][flag], False)

    def test_drive_cache_setting_is_carried_into_new_project_manifest(self):
        for preference in (True, False):
            with self.subTest(preference=preference):
                self.assertEqual(self.json_request("POST", "/api/settings", {"studio": {"cache_on_drive": preference}})[0], 200)
                job = self.create_job()
                manifest = server.STORE.get(job["id"]).metadata["studio"]["manifest"]
                self.assertIs(job["options"]["cache_on_drive"], preference)
                self.assertIs(manifest["options"]["cache_on_drive"], preference)

    def test_face_bbox_rejects_non_finite_negative_and_out_of_frame_coordinates(self):
        for bbox in ([float("nan"), 0.1, 0.5, 0.5], [-0.1, 0, 0.2, 0.2],
                     [0.1, 0.1, -0.2, 0.5], [0.8, 0, 0.5, 0.5], [0, 0.9, 0.3, 0.5],
                     [0, 0, 0, 0.4], ["0", 0, 0.2, 0.2]):
            with self.subTest(bbox=bbox):
                fields = {"operation": "dub_video", "voice_clone_consent": "true", "target_languages": '["pl-PL"]',
                          "face_selection": json.dumps({"bbox": bbox, "normalized": True})}
                body, headers = self.multipart(fields, {"video": ("video.mp4", self.video.read_bytes())})
                code, _, payload = self.request("POST", "/api/jobs", body, headers)
                self.assertEqual(code, 400, payload)
                self.assertEqual(server.STORE.list(), [])
        job = self.create_job(options={"face_selection": json.dumps({"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 0})})
        self.assertEqual(job["options"]["face_selection"]["bbox"], [0.1, 0.1, 0.5, 0.5])

    def test_face_correction_preserves_packages_inputs_and_exports_without_starting_execution(self):
        job = self.create_job("lipsync_video")
        internal = server.STORE.get(job["id"])
        internal.status, internal.error = "failed", "Ambiguous speaking face"
        sentinel = internal.directory / "exports" / "previous-report.json"
        sentinel.parent.mkdir()
        sentinel.write_bytes(b"Preserved export")
        (internal.directory / "pacote-revisado-colab.zip").write_bytes(b"Old reviewed package")
        (internal.directory / "review-approved.json").write_bytes(b"Old review")
        before_files = {name: (internal.directory / name).read_bytes() for name in server.FACE_PACKAGE_FILES}
        before_options = dict(internal.options)
        before_video = internal.input_path.read_bytes()
        audio_path = Path(internal.options["audio_path"])
        before_audio = audio_path.read_bytes()
        selection = {"bbox": [0.1, 0.2, 0.4, 0.5], "normalized": True, "frame_time": 1.5}
        with patch.object(server, "prepare_job", side_effect=AssertionError("Correction cannot start preparation")):
            code, _, result = self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {
                "face_selection": selection, "require_lipsync": False, "voice_clone_consent": False,
            })
        self.assertEqual(code, 200, result)
        self.assertEqual(result["status"], "awaiting_preparation")
        self.assertIs(result["face_selection_preparation_required"], True)
        self.assertIs(result["has_colab_package"], False)
        self.assertEqual(internal.options, {**before_options, "face_selection": selection})
        self.assertEqual(internal.input_path.read_bytes(), before_video)
        self.assertEqual(audio_path.read_bytes(), before_audio)
        self.assertEqual(sentinel.read_bytes(), b"Preserved export")
        correction = internal.metadata["face_selection_correction"]
        backup = internal.directory / correction["backup_directory"]
        for name, contents in before_files.items():
            self.assertEqual((internal.directory / name).read_bytes(), contents)
            self.assertEqual((backup / name).read_bytes(), contents)
        self.assertEqual(json.loads((backup / "backup.json").read_text(encoding="utf-8"))["previous_error"], "Ambiguous speaking face")
        for endpoint in ("colab-package", "review-approved"):
            self.assertEqual(self.json_request("GET", f"/api/jobs/{job['id']}/{endpoint}")[0], 409)
        for action in ("external-started", "awaiting-import", "colab-result", "review", "review-approved"):
            self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/{action}", {})[0], 400)
        with patch.object(server, "prepare_job", side_effect=AssertionError("Restart cannot prepare pending corrections")):
            restored = server.saved_job(json.loads((internal.directory / "job.json").read_text(encoding="utf-8")), internal.directory)
        self.assertEqual(restored.status, "awaiting_preparation")
        self.assertEqual(restored.options["face_selection"], selection)

    def test_face_correction_rejects_invalid_coordinates_and_times_before_creating_backups(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        before = (internal.directory / server.PACKAGE_NAME).read_bytes()
        valid = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        selections = [None, "auto", {}, {**valid, "normalized": "true"}]
        selections += [{**valid, "bbox": bbox} for bbox in (
            [float("nan"), 0, 0.1, 0.1], [float("inf"), 0, 0.1, 0.1],
            [True, 0, 0.1, 0.1], [10**1000, 0, 0.1, 0.1], [-0.1, 0, 0.1, 0.1], [0.9, 0, 0.5, 0.5], [0, 0, 0, 0.5],
        )]
        selections += [{**valid, "frame_time": value} for value in (True, -1, float("nan"), float("inf"), "1", 3, 4, 10**1000)]
        for selection in selections:
            with self.subTest(selection=selection):
                code, _, response = self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})
                self.assertEqual(code, 400, response)
                self.assertEqual(internal.status, "awaiting_colab")
                self.assertEqual(internal.options["face_selection"], "auto")
                self.assertEqual((internal.directory / server.PACKAGE_NAME).read_bytes(), before)
                self.assertFalse((internal.directory / "face-selection-history").exists())

    def test_face_correction_rejects_disabled_lipsync_busy_projects_and_approved_versions(self):
        selection = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        for operation, options in (("text_to_voice", {}), ("dub_video", {"require_lipsync": "false"})):
            job = self.create_job(operation, options=options)
            self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 400)
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        for status in ("queued", "preparing_colab", "processing_colab", "validating", "complete", "archived", "partial"):
            with self.subTest(status=status):
                internal.status = status
                self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 400)
        internal.status = "failed"
        for field, data in (
            ("studio_results", [{"status": "complete"}]),
            ("outputs", [{"status": "approved"}]),
            ("studio_result", {"outputs": [{"status": "verified"}]}),
            ("studio_result_history", [{"versions": [{"status": "validated"}]}]),
        ):
            with self.subTest(field=field):
                internal.metadata[field] = data
                self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 400)
                internal.metadata.pop(field)
        for field in ("output_path", "report_path"):
            with self.subTest(path=field):
                setattr(internal, field, internal.directory / "approved-output")
                self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 400)
                setattr(internal, field, None)
        self.assertFalse((internal.directory / "face-selection-history").exists())

    def test_face_correction_paths_remain_confined_and_failed_persistence_rolls_back_options(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        original_directory, original_input = internal.directory, internal.input_path
        outside = self.root / "outside"
        outside.mkdir()
        source = outside / "video.mp4"
        source.write_bytes(self.video.read_bytes())
        selection = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        endpoint = f"/api/jobs/{job['id']}/face-selection"
        internal.directory = outside
        self.assertEqual(self.json_request("POST", endpoint, {"face_selection": selection})[0], 400)
        internal.directory, internal.input_path = original_directory, source
        self.assertEqual(self.json_request("POST", endpoint, {"face_selection": selection})[0], 400)
        internal.input_path = original_input
        real_is_symlink = Path.is_symlink
        for linked_name in (server.PACKAGE_NAME, "studio-manifest.json", "face-selection-history"):
            with self.subTest(linked=linked_name), patch.object(Path, "is_symlink", lambda path: path.name == linked_name or real_is_symlink(path)):
                self.assertEqual(self.json_request("POST", endpoint, {"face_selection": selection})[0], 400)
        self.assertFalse((original_directory / "face-selection-history").exists())
        before_options, before_metadata = dict(internal.options), dict(internal.metadata)
        with patch.object(server, "persist_job", side_effect=OSError("Synthetic persistence failure")):
            code, _, response = self.json_request("POST", endpoint, {"face_selection": selection})
        self.assertEqual(code, 400, response)
        self.assertEqual(internal.status, "awaiting_colab")
        self.assertEqual(internal.options, before_options)
        self.assertEqual(internal.metadata, before_metadata)
        self.assertEqual(source.read_bytes(), self.video.read_bytes())
        self.assertTrue((original_directory / server.PACKAGE_NAME).is_file())

    def test_prepare_face_corrected_package_requires_confirmation_and_is_idempotent(self):
        job = self.create_job("lipsync_video")
        internal = server.STORE.get(job["id"])
        previous_manifest = internal.metadata["studio"]["manifest"]
        selection = {"bbox": [0.2, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 2.5}
        endpoint = f"/api/jobs/{job['id']}/prepare-colab"
        self.assertEqual(self.json_request("POST", endpoint, {"confirm": True})[0], 400)
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 200)
        for payload in ({}, {"confirm": False}, {"confirm": "true"}, {"confirm": 1}):
            self.assertEqual(self.json_request("POST", endpoint, payload)[0], 400)
            self.assertEqual(internal.status, "awaiting_preparation")
        with patch.object(server, "prepare_studio_package", wraps=server.prepare_studio_package) as preparation:
            code, _, preparing = self.json_request("POST", endpoint, {"confirm": True})
            self.assertEqual(code, 202, preparing)
            self.assertEqual(preparing["status"], "preparing_colab")
            result = self.wait_for_preparation(job)
            self.assertEqual(result["status"], "awaiting_colab", result)
            self.assertIs(result["face_selection_preparation_required"], False)
            self.assertIs(result["has_colab_package"], True)
            self.assertEqual(result["error"], "")
            self.assertEqual(self.json_request("POST", endpoint, {"confirm": True})[0], 200)
            self.assertEqual(preparation.call_count, 1)
        code, _, contents = self.request("GET", f"/api/jobs/{job['id']}/colab-package")
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(contents)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(manifest["options"]["face_selection"], selection)
            self.assertEqual(manifest["inputs"], previous_manifest["inputs"])
            self.assertEqual(manifest["authorization"], previous_manifest["authorization"])
            for entry in manifest["inputs"].values():
                self.assertEqual(hashlib.sha256(archive.read(entry["path"])).hexdigest(), entry["sha256"])
        self.assertTrue(internal.metadata["face_selection_correction"]["prepared_at"])

    def test_concurrent_face_preparation_requests_start_only_one_local_worker(self):
        job = self.create_job()
        selection = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 200)
        entered, release = threading.Event(), threading.Event()
        original_prepare = server.prepare_studio_package
        def delayed_prepare(value):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("Synthetic local worker timed out")
            return original_prepare(value)
        endpoint = f"/api/jobs/{job['id']}/prepare-colab"
        second_response = []
        with patch.object(server, "prepare_studio_package", side_effect=delayed_prepare) as preparation:
            self.assertEqual(self.json_request("POST", endpoint, {"confirm": True})[0], 202)
            self.assertTrue(entered.wait(5))
            second = threading.Thread(target=lambda: second_response.append(self.json_request("POST", endpoint, {"confirm": True})))
            second.start()
            try:
                time.sleep(0.05)
            finally:
                release.set()
            second.join(15)
            self.assertFalse(second.is_alive())
            self.assertEqual(second_response[0][0], 200, second_response)
            self.assertEqual(self.wait_for_preparation(job)["status"], "awaiting_colab")
            self.assertEqual(preparation.call_count, 1)

    def test_failed_face_preparation_restores_previous_bytes_and_keeps_partial_attempt_for_retry(self):
        job = self.create_job()
        internal = server.STORE.get(job["id"])
        old_studio = json.loads(json.dumps(internal.metadata["studio"]))
        before_files = {name: (internal.directory / name).read_bytes() for name in server.FACE_PACKAGE_FILES if (internal.directory / name).is_file()}
        before_input = internal.input_path.read_bytes()
        export = internal.directory / "exports" / "old-report.json"
        export.parent.mkdir()
        export.write_bytes(b"Existing report must remain")
        selection = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 200)
        def broken_prepare(value):
            (value.directory / server.PACKAGE_NAME).write_bytes(b"Incomplete corrected ZIP")
            (value.directory / "studio-manifest.json").write_bytes(b"Incomplete corrected manifest")
            (value.directory / "review-approved.json").write_bytes(b"Incomplete corrected review")
            value.metadata["studio"] = {"manifest": {"invalid": True}}
            raise OSError("Synthetic packaging failure")
        with patch.object(server, "prepare_studio_package", side_effect=broken_prepare):
            self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/prepare-colab", {"confirm": True})[0], 202)
            failed = self.wait_for_preparation(job)
        self.assertEqual(failed["status"], "failed", failed)
        self.assertIn("Synthetic packaging failure", failed["error"])
        self.assertIs(failed["face_selection_preparation_required"], True)
        self.assertIs(failed["has_colab_package"], False)
        self.assertEqual(internal.metadata["studio"], old_studio)
        correction = internal.metadata["face_selection_correction"]
        backup, attempt = (internal.directory / correction[key] for key in ("backup_directory", "last_failed_preparation"))
        for name, contents in before_files.items():
            self.assertEqual((internal.directory / name).read_bytes(), contents)
            self.assertEqual((backup / name).read_bytes(), contents)
        self.assertFalse((internal.directory / "review-approved.json").exists())
        self.assertEqual((attempt / server.PACKAGE_NAME).read_bytes(), b"Incomplete corrected ZIP")
        self.assertEqual((attempt / "review-approved.json").read_bytes(), b"Incomplete corrected review")
        self.assertEqual(internal.input_path.read_bytes(), before_input)
        self.assertEqual(export.read_bytes(), b"Existing report must remain")
        self.assertEqual(self.json_request("GET", f"/api/jobs/{job['id']}/colab-package")[0], 409)
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/prepare-colab", {"confirm": True})[0], 202)
        self.assertEqual(self.wait_for_preparation(job)["status"], "awaiting_colab")
        self.assertEqual((backup / server.PACKAGE_NAME).read_bytes(), before_files[server.PACKAGE_NAME])
        self.assertEqual(internal.options["face_selection"], selection)

    def test_reviewed_package_requires_reexport_after_face_correction(self):
        job = self.create_job()
        review = self.review_for(job)
        body, headers = self.multipart({}, {"review": ("review.zip", self.zip_bytes({"review.json": json.dumps(review).encode()}))})
        self.assertEqual(self.request("POST", f"/api/jobs/{job['id']}/review", body, headers)[0], 200)
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/review-approved", review)[0], 200)
        internal = server.STORE.get(job["id"])
        previous = (internal.directory / "pacote-revisado-colab.zip").read_bytes()
        selection = {"bbox": [0.1, 0.1, 0.5, 0.5], "normalized": True, "frame_time": 1}
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/face-selection", {"face_selection": selection})[0], 200)
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/prepare-colab", {"confirm": True})[0], 202)
        self.assertEqual(self.wait_for_preparation(job)["status"], "awaiting_colab")
        self.assertEqual((internal.directory / "pacote-revisado-colab.zip").read_bytes(), previous)
        self.assertEqual(self.json_request("GET", f"/api/jobs/{job['id']}/review-approved")[0], 409)
        self.assertEqual(self.json_request("POST", f"/api/jobs/{job['id']}/review-approved", review)[0], 200)
        code, _, corrected = self.request("GET", f"/api/jobs/{job['id']}/review-approved")
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(corrected)) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["options"]["face_selection"], selection)
            self.assertEqual(json.loads(archive.read("review-approved.json"))["translations"], review["translations"])
        backup = internal.directory / internal.metadata["face_selection_correction"]["backup_directory"]
        self.assertEqual((backup / "pacote-revisado-colab.zip").read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
