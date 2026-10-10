"""Studio Colab local: pacotes, revisão e validação; nunca inicia IA ou Pods."""
from __future__ import annotations
import argparse
import cgi
import json
import math
import mimetypes
import re
import secrets
import shutil
import socket
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from languages import PRESETS, languages_payload
from pipeline import Job, JobStore, PipelineError, probe_video
from studio_bridge import prepare_studio_package, import_studio_result, import_review, export_approved_review

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
RUNTIME = ROOT / "runtime"
JOBS_DIR = RUNTIME / "jobs"
CONFIG_PATH = RUNTIME / "config.json"
NOTEBOOK_PATH = ROOT / "DubStudio_Colab.ipynb"
PACKAGE_NAME = "pacote-colab.zip"
VERSION = "3.0.0"
PUBLIC_NOTEBOOK_URL = "https://colab.research.google.com/github/DYNHOSANTANA657/dubstudio-colab/blob/main/DubStudio_Colab.ipynb"
MAX_UPLOAD = 8 * 1024**3
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4"}
OPERATIONS = {"dub_video", "lipsync_video", "text_to_voice"}
DEFAULT_CONFIG = {
    "colab": {"notebook_url": PUBLIC_NOTEBOOK_URL},
    "studio": {"performance_profile": "fast_lipsync", "preserve_ambience": True,
               "review_mode": "before_voice", "cache_on_drive": True, "max_export_resolution": 1080},
    "limits": {"max_upload_gb": 8, "max_duration_minutes": 20, "recommended_duration_minutes": 2},
}
STORE = JobStore()
STATE_LOCK = threading.RLock()
JOB_LOCKS = {}


class ThreadingHTTPServer(_ThreadingHTTPServer):
    """Only one Studio may own a local port, including on Windows.

    Python's HTTPServer enables SO_REUSEADDR by default. On Windows that can
    admit another listener, leaving requests nondeterministically served by an
    old Studio. Exclusive binding must be set before bind, never combined with
    address/port reuse. The public class name stays compatible with our tests.
    """

    allow_reuse_address = False
    allow_reuse_port = False

    def server_bind(self):
        exclusive_option = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive_option is not None:
            self.socket.setsockopt(socket.SOL_SOCKET, exclusive_option, 1)
        super().server_bind()

def _raw_config():
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}

def load_config():
    """Só contrato público novo, sem segredos ou comandos legados."""
    raw = _raw_config()
    return {key: {**defaults, **{k: v for k, v in (raw.get(key) if isinstance(raw.get(key), dict) else {}).items() if k in defaults}}
            for key, defaults in DEFAULT_CONFIG.items()}

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)

def save_config(payload):
    current = load_config()
    colab = payload.get("colab", {})
    if not isinstance(colab, dict):
        raise ValueError("Configuração de notebook inválida.")
    url = str(colab.get("notebook_url", current["colab"]["notebook_url"])).strip()
    parsed = urlparse(url)
    if url and (parsed.scheme != "https" or parsed.netloc != "colab.research.google.com"):
        raise ValueError("Use a URL HTTPS do notebook em colab.research.google.com.")
    current["colab"]["notebook_url"] = url
    studio = payload.get("studio", {})
    if not isinstance(studio, dict):
        raise ValueError("Configurações do Studio inválidas.")
    for key in ("preserve_ambience", "cache_on_drive"):
        if key in studio:
            if not isinstance(studio[key], bool):
                raise ValueError(f"Use verdadeiro ou falso para {key}.")
            current["studio"][key] = studio[key]
    for key, accepted in (("performance_profile", {"fast_lipsync", "maximum_quality"}),
                          ("review_mode", {"automatic", "before_voice"})):
        if key in studio:
            if studio[key] not in accepted:
                raise ValueError(f"Opção inválida: {key}.")
            current["studio"][key] = studio[key]
    raw = _raw_config()
    previous_colab = dict(raw.get("colab", {})) if isinstance(raw.get("colab"), dict) else {}
    raw.update(current)
    if isinstance(raw.get("runpod"), dict):
        raw["runpod"]["enabled"] = False
    if raw.get("studio_schema_version") != 2:
        # Um notebook legado não aceita o manifesto novo. Preserve a configuração
        # antiga, mas só abra um notebook novo escolhido explicitamente.
        if previous_colab.get("notebook_url") and "notebook_url" not in colab:
            raw["legacy_colab"] = previous_colab
            current["colab"]["notebook_url"] = ""
            raw["colab"] = current["colab"]
        raw["studio_schema_version"] = 2
    RUNTIME.mkdir(parents=True, exist_ok=True)
    with STATE_LOCK:
        atomic_json(CONFIG_PATH, raw)
    return load_config()

def job_lock(job):
    with STATE_LOCK:
        return JOB_LOCKS.setdefault(job.id, threading.RLock())

def public_job(job):
    data = job.public()
    data["operation"] = job.options.get("operation", "legacy")
    data["execution_location"] = "colab" if data["operation"] in OPERATIONS else "legacy"
    data["progress_source"] = "local" if job.status in {"preparing_colab", "validating", "complete"} else "notebook"
    data["has_review"] = bool(job.metadata.get("review"))
    data["options"] = {k: v for k, v in job.options.items() if not k.endswith("_path")}
    outputs = job.metadata.get("studio_results", [])
    data["metadata"] = {**job.metadata, "outputs": outputs}
    data["has_output"] = any(item.get("status") == "complete" for item in outputs)
    data["has_report"] = data["has_output"]
    return data

def persist_job(job):
    with STATE_LOCK:
        atomic_json(job.directory / "job.json", job.public())

def restore_jobs():
    """Histórico preservado; nunca retome processamento externo silenciosamente."""
    if not JOBS_DIR.exists():
        return
    for record in JOBS_DIR.glob("*/job.json"):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not record.parent.resolve().is_relative_to(JOBS_DIR.resolve()):
                continue
            if data.get("status") == "archived" or not re.fullmatch(r"[a-f0-9]{12}", str(data.get("id", ""))):
                continue
            if record.parent.name != data["id"] or not isinstance(data.get("options", {}), dict) or not isinstance(data.get("metadata", {}), dict):
                continue
            directory = record.parent
            candidates = [p for p in directory.glob("original.*") if p.is_file()]
            input_path = candidates[0] if candidates else directory / "texto.txt"
            if not input_path.exists():
                continue
            if not input_path.resolve().is_relative_to(directory.resolve()):
                continue
            job = Job(id=data["id"], directory=directory, input_path=input_path,
                      options=dict(data.get("options", {})), status=str(data.get("status", "failed")),
                      progress=int(data.get("progress", 0)), stage=str(data.get("stage", "Histórico")),
                      message=str(data.get("message", "")), metadata=dict(data.get("metadata", {})),
                      error=str(data.get("error", "")), created_at=float(data.get("created_at", time.time())),
                      updated_at=float(data.get("updated_at", time.time())))
            if job.options.get("operation") not in OPERATIONS:
                job.metadata["legacy"] = True
                job.status, job.stage = "legacy", "Histórico preservado"
                job.message = "Projeto anterior à migração. Arquivos preservados; crie um projeto Colab para executar."
            elif job.status in {"queued", "preparing_colab", "validating"}:
                job.status = "awaiting_colab" if (directory / PACKAGE_NAME).exists() else "failed"
                job.progress = 0
                job.stage = "Aguardando Colab" if job.status == "awaiting_colab" else "Preparação interrompida"
                job.message = "O Studio foi reiniciado; nenhum trabalho externo foi reenviado."
            STORE.add(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue

def readiness():
    return {"ffmpeg": {"ready": bool(shutil.which("ffmpeg"))},
            "ffprobe": {"ready": bool(shutil.which("ffprobe"))},
            "colab_notebook_ready": NOTEBOOK_PATH.is_file(),
            "notebook_published": bool(load_config()["colab"]["notebook_url"]),
            "execution_location": "colab", "local_ai_required": False,
            "validation_status": "pending_gpu_validation"}

def as_bool(value, default=False):
    if value is None:
        return default
    text = str(value).lower()
    if text in {"true", "1", "on", "yes", "sim"}:
        return True
    if text in {"false", "0", "off", "no", "nao", "não"}:
        return False
    raise ValueError("Opção booleana inválida; use verdadeiro ou falso.")

def parse_job_options(fields):
    operation = fields.getfirst("operation", "dub_video")
    if operation not in OPERATIONS:
        raise ValueError("Operação inválida.")
    if not as_bool(fields.getfirst("voice_clone_consent")):
        raise ValueError("Confirme a autorização de uso do vídeo e da voz antes de preparar o pacote.")
    try:
        languages = json.loads(fields.getfirst("target_languages", "[\"pl-PL\"]"))
    except ValueError as exc:
        raise ValueError("Selecione os idiomas de destino.") from exc
    if operation == "lipsync_video" and languages == []:
        languages = ["und"]  # Áudio enviado, sem tradução nem promessa de idioma.
    if not isinstance(languages, list) or not 1 <= len(languages) <= 23 or not all(isinstance(v, str) for v in languages):
        raise ValueError("Selecione de 1 a 23 idiomas válidos.")
    if len(languages) != len(set(languages)):
        raise ValueError("Há idiomas duplicados.")
    available = {item["code"]: item for item in languages_payload()}
    for language in languages:
        if operation == "lipsync_video" and language == "und":
            continue
        if language not in available or not available[language].get("clone_supported", False):
            raise ValueError(f"O timbre original não está disponível para {language}; não usaremos voz genérica.")
    profile = fields.getfirst("performance_profile", "fast_lipsync")
    if profile not in {"fast_lipsync", "maximum_quality"}:
        raise ValueError("Perfil de qualidade inválido.")
    review_mode = fields.getfirst("review_mode", "before_voice")
    if review_mode not in {"automatic", "before_voice"}:
        raise ValueError("Modo de revisão inválido.")
    subtitle = fields.getfirst("subtitle_mode", "srt")
    if subtitle not in {"none", "srt", "burn"}:
        raise ValueError("Modo de legendas inválido.")
    require_lipsync = operation == "lipsync_video" or (
        operation == "dub_video" and as_bool(fields.getfirst("require_lipsync"), True))
    face = fields.getfirst("face_selection", "auto") if require_lipsync else "auto"
    if face != "auto":
        try:
            face = json.loads(face)
        except ValueError as exc:
            raise ValueError("Seleção de rosto inválida.") from exc
        if not isinstance(face, dict) or not isinstance(face.get("bbox"), list) or len(face["bbox"]) != 4:
            raise ValueError("Selecione um rosto válido ou use seleção automática.")
        coordinates = face["bbox"]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in coordinates):
            raise ValueError("Coordenadas do rosto inválidas.")
        x, y, width, height = coordinates
        frame_time = face.get("frame_time", 0)
        if (face.get("normalized") is not True or x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > 1 or y + height > 1
                or not isinstance(frame_time, (int, float)) or isinstance(frame_time, bool)
                or not math.isfinite(frame_time) or frame_time < 0):
            raise ValueError("Selecione um rosto dentro da imagem e um tempo válido.")
    text = fields.getfirst("text", "").strip()
    if operation == "text_to_voice" and not 1 <= len(text) <= 12000:
        raise ValueError("Informe um texto com até 12.000 caracteres.")
    speaker_count = fields.getfirst("speaker_count", "auto")
    if speaker_count not in {"auto", "1", "2", "3", "4"}:
        raise ValueError("Quantidade de locutores inválida.")
    return {"operation": operation, "mode": "colab", "target_languages": languages,
            "target_language": languages[0], "source_language": fields.getfirst("source_language", "auto"),
            "performance_profile": profile, "review_mode": review_mode,
            "cache_on_drive": as_bool(fields.getfirst("cache_on_drive"), load_config()["studio"]["cache_on_drive"]),
            "preset": "original", "require_lipsync": require_lipsync,
            "preserve_ambience": as_bool(fields.getfirst("preserve_ambience"), True),
            "voice_clone_consent": True, "voice_clone_confirmed": True,
            "text": text, "speaker_count": speaker_count, "subtitle_mode": subtitle,
            "face_selection": face, "max_duration_minutes": 20}

def prepare_job(job):
    with job_lock(job):
        try:
            job.status, job.stage, job.progress = "preparing_colab", "Preparando pacote", 0
            persist_job(job)
            prepare_studio_package(job)
            job.status, job.progress, job.stage = "awaiting_colab", 0, "Pacote pronto"
            job.message = "Abra o notebook, conecte uma GPU e envie o ZIP. Acompanhe a execução no Colab."
        except Exception as exc:
            job.status, job.stage, job.error = "failed", "Falha na preparação", str(exc)
        job.updated_at = time.time()
        persist_job(job)

class Handler(BaseHTTPRequestHandler):
    server_version = "DubStudio/3.0"
    protocol_version = "HTTP/1.1"
    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

    def trusted_request(self):
        host = self.headers.get("Host", "")
        allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        origin = self.headers.get("Origin")
        if host not in allowed or (origin and origin != f"http://{host}"):
            self.send_json({"error": "O Studio aceita somente solicitações da interface local."}, 403)
            return False
        return True

    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_file(self, path, download_name=None):
        if not path or not path.is_file():
            self.send_json({"error": "Arquivo não disponível ou resultado ainda não validado."}, 409)
            return
        size = path.stat().st_size
        start, end, partial = 0, size - 1, False
        raw_range = self.headers.get("Range", "")
        match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw_range)
        if raw_range and not download_name:
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            a, b = match.groups()
            if a:
                start, end = int(a), min(int(b), size - 1) if b else size - 1
            else:
                start = max(size - int(b), 0)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            partial = True
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(max(0, end - start + 1)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("X-Content-Type-Options", "nosniff")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        try:
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = stream.read(min(1024**2, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 <= length <= 4 * 1024**2:
            raise ValueError("Solicitação JSON muito grande.")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("Envie um objeto JSON.")
        return payload

    def read_form(self):
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_UPLOAD:
            raise ValueError("Arquivo vazio ou envio maior que 8 GB.")
        if not self.headers.get("Content-Type", "").startswith("multipart/form-data;"):
            raise ValueError("Envie os arquivos como multipart/form-data.")
        return cgi.FieldStorage(fp=self.rfile, headers=self.headers,
                                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": self.headers.get("Content-Type"),
                                         "CONTENT_LENGTH": str(length)})

    def save_upload(self, fields, key, directory, stem, extensions, required=True):
        if key not in fields:
            if required:
                raise ValueError(f"Selecione o arquivo: {key}.")
            return None
        item = fields[key]
        if isinstance(item, list) or not item.filename or item.file is None:
            raise ValueError(f"Arquivo inválido: {key}.")
        extension = Path(item.filename).suffix.lower()
        if extension not in extensions:
            raise ValueError(f"Formato não suportado para {key}.")
        destination = directory / (stem + extension)
        with destination.open("wb") as stream:
            shutil.copyfileobj(item.file, stream, 1024**2)
        if destination.stat().st_size == 0:
            raise ValueError(f"Arquivo vazio: {key}.")
        return destination

    def do_GET(self):
        if not self.trusted_request():
            return
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/health":
            self.send_json({"ok": True, "version": VERSION, "engines": readiness(),
                            "validation": {"pipeline_validation": "pending_gpu_validation"}})
            return
        if path == "/api/catalog":
            self.send_json({"languages": languages_payload(), "presets": PRESETS})
            return
        if path == "/api/settings":
            self.send_json({"config": load_config(), "readiness": readiness()})
            return
        if path == "/api/colab/notebook":
            self.send_file(NOTEBOOK_PATH, "DubStudio_Colab.ipynb")
            return
        if path == "/api/jobs":
            with STATE_LOCK:
                jobs = [public_job(STORE.get(item["id"])) for item in STORE.list()]
            self.send_json({"jobs": jobs})
            return
        if path.startswith("/api/runpod"):
            self.send_json({"error": "RunPod foi retirado deste Studio. Use o fluxo Colab."}, 410)
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})(?:/(.*))?", path)
        if match:
            job = STORE.get(match.group(1))
            if not job:
                self.send_json({"error": "Projeto não encontrado."}, 404)
                return
            suffix = match.group(2)
            if not suffix:
                self.send_json(public_job(job))
                return
            if suffix == "review":
                self.send_json(job.metadata.get("review", {"error": "Não há revisão importada."}))
                return
            if suffix in {"source", "colab-package", "review-approved", "reference", "audio"}:
                target = {"source": job.input_path, "colab-package": job.directory / PACKAGE_NAME,
                          "review-approved": job.directory / "pacote-revisado-colab.zip",
                          "reference": Path(job.options.get("reference_path", "__missing__")),
                          "audio": Path(job.options.get("audio_path", "__missing__"))}[suffix]
                download = None if suffix in {"source", "reference", "audio"} else target.name
                if not target.resolve().is_relative_to(job.directory.resolve()):
                    self.send_json({"error": "Arquivo fora do projeto local."}, 403)
                    return
                self.send_file(target, download)
                return
            output = re.fullmatch(r"outputs/([A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*)/(video|wav|mp3|report|segments|original_srt|translation_srt)", suffix)
            if output:
                language, kind = output.groups()
                item = next((v for v in job.metadata.get("studio_results", []) if v.get("language") == language and v.get("status") == "complete"), None)
                relative = item.get("files", {}).get(kind) if item else None
                target = (job.directory / relative).resolve() if relative else None
                if not target or not target.is_relative_to(job.directory.resolve()):
                    self.send_json({"error": "Versão não validada ou arquivo ausente."}, 409)
                    return
                preview = parse_qs(parsed.query).get("preview") == ["1"]
                self.send_file(target, None if preview else f"DubStudio-{language}-{target.name}")
                return
            self.send_json({"error": "Arquivo ou ação não encontrada."}, 404)
            return
        target = (STATIC / ("index.html" if path == "/" else path.lstrip("/"))).resolve()
        if not target.is_relative_to(STATIC.resolve()) or not target.is_file():
            self.send_error(404)
            return
        self.send_file(target)

    def do_DELETE(self):
        self.close_connection = True
        if not self.trusted_request():
            return
        match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})", urlparse(self.path).path)
        job = STORE.get(match.group(1)) if match else None
        if not job:
            self.send_json({"error": "Projeto não encontrado."}, 404)
            return
        with job_lock(job):
            job.status, job.stage = "archived", "Arquivado"
            job.updated_at = time.time()
            persist_job(job)
            with STATE_LOCK:
                STORE.remove(job.id)
        self.send_json({"ok": True, "recoverable": True, "message": "Projeto arquivado; arquivos preservados no computador."})

    def do_POST(self):
        # Erros anteriores à leitura do corpo não podem deixar bytes pendentes
        # para serem interpretados como outra solicitação HTTP.
        self.close_connection = True
        if not self.trusted_request():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/settings":
                self.send_json({"config": save_config(self.read_json()), "readiness": readiness()})
                return
            if path.startswith("/api/runpod") or path.endswith("/retry-runpod"):
                self.send_json({"error": "Execução RunPod desativada. Nenhum Pod será iniciado."}, 410)
                return
            if path == "/api/jobs":
                form = self.read_form()
                options = parse_job_options(form)
                job_id = secrets.token_hex(6)
                directory = JOBS_DIR / job_id
                directory.mkdir(parents=True, exist_ok=False)
                video = self.save_upload(form, "video", directory, "original", VIDEO_EXTENSIONS, options["operation"] != "text_to_voice")
                audio = self.save_upload(form, "audio", directory, "audio", AUDIO_EXTENSIONS, options["operation"] == "lipsync_video")
                reference = self.save_upload(form, "reference", directory, "referencia", AUDIO_EXTENSIONS, options["operation"] == "text_to_voice")
                if audio:
                    options["audio_path"] = str(audio)
                if reference:
                    options["reference_path"] = str(reference)
                if video:
                    media = probe_video(video)
                    if not media.get("video") or media["duration"] <= 0:
                        raise ValueError("O arquivo não contém uma faixa de vídeo válida.")
                    if options["operation"] == "dub_video" and not media.get("audio"):
                        raise ValueError("A dublagem precisa de áudio no vídeo original.")
                    if media["duration"] > 20 * 60 + 1:
                        raise ValueError("O limite inicial é de 20 minutos por vídeo.")
                    input_path = video
                else:
                    input_path = directory / "texto.txt"
                    input_path.write_text(options["text"], encoding="utf-8")
                    media = {}
                for sample in (audio, reference):
                    if sample and not probe_video(sample).get("audio"):
                        raise ValueError("O áudio enviado não possui faixa de áudio válida.")
                job = Job(id=job_id, directory=directory, input_path=input_path, options=options)
                job.metadata["original_name"] = str(form["video"].filename) if video else "Texto para voz"
                job.metadata["media"] = media
                job.metadata["validation_status"] = "pending_gpu_validation"
                STORE.add(job)
                persist_job(job)
                threading.Thread(target=prepare_job, args=(job,), daemon=True).start()
                self.send_json(public_job(job), 201)
                return
            match = re.fullmatch(r"/api/jobs/([a-f0-9]{12})/(external-started|awaiting-import|colab-result|review|review-approved)", path)
            if not match:
                self.send_json({"error": "Ação não encontrada."}, 404)
                return
            job = STORE.get(match.group(1))
            if not job or job.options.get("operation") not in OPERATIONS:
                self.send_json({"error": "Projeto Colab não encontrado."}, 404)
                return
            action = match.group(2)
            with job_lock(job):
                if action in {"external-started", "awaiting-import"}:
                    self.read_json()
                    if job.status not in {"awaiting_colab", "processing_colab", "awaiting_import", "review_approved"}:
                        raise ValueError("O pacote não está pronto para execução externa.")
                    job.status = "processing_colab" if action == "external-started" else "awaiting_import"
                    job.progress = 0
                    job.stage = "Execução no Colab" if action == "external-started" else "Importe o resultado"
                    job.message = "Acompanhe as etapas e o progresso real pelo notebook."
                elif action in {"colab-result", "review"}:
                    form = self.read_form()
                    key = "result" if action == "colab-result" else "review"
                    with tempfile.TemporaryDirectory(prefix="dubstudio-import-") as temporary:
                        archive = self.save_upload(form, key, Path(temporary), "resultado", {".zip"})
                        if action == "review":
                            import_review(job, archive)
                            job.status, job.stage = "review_required", "Revise a tradução"
                            job.message = "Aprove os textos e envie o pacote revisado ao mesmo notebook."
                        else:
                            import_studio_result(job, archive)
                            outputs = job.metadata.get("studio_results", [])
                            completed = sum(v.get("status") == "complete" for v in outputs)
                            job.status = "complete" if outputs and completed == len(outputs) else "partial" if completed else "failed"
                            job.progress = 100 if job.status == "complete" else 0
                            job.stage = "Arquivos validados" if completed else "Falha informada pelo notebook"
                            job.message = f"{completed} de {len(outputs)} versões aprovadas na validação de arquivos. Confira a reprodução."
                elif action == "review-approved":
                    export_approved_review(job, self.read_json())
                    job.status, job.stage = "review_approved", "Revisão aprovada"
                    job.message = "Baixe o pacote revisado e retome o notebook; etapas anteriores serão reaproveitadas."
                job.updated_at, job.error = time.time(), ""
                persist_job(job)
                payload = public_job(job)
                if action == "review-approved":
                    payload["download_url"] = f"/api/jobs/{job.id}/review-approved"
                self.send_json(payload)
        except (ValueError, PipelineError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.log_message("Falha local: %s", type(exc).__name__)
            self.send_json({"error": "Falha local inesperada. Os arquivos anteriores foram preservados."}, 500)

def main():
    parser = argparse.ArgumentParser(description="DubStudio Colab")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    save_config({})
    restore_jobs()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"DubStudio Colab disponível em http://{args.host}:{args.port}", flush=True)
    print("IA somente no notebook; CTRL+C encerra a interface local.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
