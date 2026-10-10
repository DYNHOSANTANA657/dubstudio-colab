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
from unittest.mock import patch

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
        subprocess.run([shutil.which("ffmpeg"), "-y", "-i", str(cls.video), "-vn", str(cls.audio)], check=True, capture_output=True)
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


if __name__ == "__main__":
    unittest.main()
