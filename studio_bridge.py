"""Versioned, offline projects between the local Studio and a manual Colab run.

Returned file paths are relative to the job directory. Imported artifacts are
published only after every claimed complete language passes local validation.
The legacy Colab bridge and old job archives remain independent of this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from colab_bridge import MAX_DURATION_SECONDS, _package_source, _tail_max_volume, sha256_file, target_dimensions
from languages import supports_original_voice
from pipeline import Job, PipelineError, probe_video

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = "2.0"
PIPELINE_VERSION = "3.0.0"
PACKAGE_NAME = "pacote-colab.zip"
MANIFEST_NAME = "studio-manifest.json"
MAX_ZIP_BYTES = 12 * 1024**3
MAX_JSON_BYTES = 4 * 1024**2
MAX_MEMBERS = 2048
OPERATIONS = {"dub_video", "lipsync_video", "text_to_voice"}
FILE_EXTENSIONS = {
    "video": {".mp4"}, "wav": {".wav"}, "mp3": {".mp3"}, "report": {".json"},
    "segments": {".json"}, "original_srt": {".srt"}, "translation_srt": {".srt"},
}
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _language_list(job: Job) -> list[str]:
    values = job.options.get("target_languages", [job.options.get("language", "pl-PL")])
    if not isinstance(values, list) or not values or len(values) > 40:
        raise PipelineError("Escolha entre 1 e 40 idiomas de destino.")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not _LANGUAGE_RE.fullmatch(value):
            raise PipelineError("O idioma de destino precisa usar um código BCP-47 válido.")
        if value in result:
            raise PipelineError("O projeto contém idiomas de destino duplicados.")
        if job.options.get("operation", "dub_video") != "lipsync_video" and not supports_original_voice(value):
            raise PipelineError(f"O motor atual não oferece timbre original para {value}.")
        result.append(value)
    return result


def _input_path(job: Job, role: str) -> Path | None:
    entry = (job.metadata.get("studio_inputs") or {}).get(role)
    if isinstance(entry, dict):
        entry = entry.get("path")
    value = entry or job.options.get(f"{role}_path")
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = job.directory / path
        path = path.resolve()
        if not path.is_relative_to(job.directory.resolve()):
            raise PipelineError("Um arquivo do projeto aponta para fora da pasta do trabalho.")
        if not path.is_file() or path.stat().st_size == 0:
            raise PipelineError(f"O arquivo de {role} não foi encontrado ou está vazio.")
        return path
    return None


def _check_identity(job: Job) -> None:
    if not re.fullmatch(r"[0-9a-f]{12}", job.id):
        raise PipelineError("O identificador do projeto não é válido.")


def prepare_studio_package(job: Job) -> Path:
    """Create manifest v2 and an offline ZIP; no cloud side effects occur here."""
    _check_identity(job)
    operation = job.options.get("operation", "dub_video")
    if operation not in OPERATIONS:
        raise PipelineError("A operação solicitada não existe no Studio.")
    targets = _language_list(job)
    consent = job.options.get("voice_clone_consent") is True
    if operation in {"dub_video", "text_to_voice"} and not consent:
        raise PipelineError("Confirme a autorização para preservar o timbre da voz original.")
    source = job.input_path.resolve()
    if not source.is_relative_to(job.directory.resolve()) or not source.is_file() or not source.stat().st_size:
        raise PipelineError("O arquivo original do projeto não é válido.")
    if operation == "text_to_voice":
        # The HTTP app stores text in texto.txt; only the vocal reference is media.
        source = _input_path(job, "reference") or source
    inputs: dict[str, dict[str, str]] = {}
    input_files: dict[str, Path] = {}
    metadata = probe_video(source)
    proxied = False
    if operation in {"dub_video", "lipsync_video"}:
        if not metadata.get("video"):
            raise PipelineError("Escolha um arquivo com uma faixa de vídeo válida.")
        if operation == "dub_video" and not metadata.get("audio"):
            raise PipelineError("O vídeo precisa conter a voz original para a dublagem.")
        if not 0 < float(metadata.get("duration", 0)) <= MAX_DURATION_SECONDS:
            raise PipelineError("O Studio aceita vídeos com duração máxima de 20 minutos.")
        source, proxied = _package_source(job, metadata)
        input_files["video"] = source
    else:
        text = job.options.get("text", "")
        if not isinstance(text, str) or not text.strip() or len(text) > 40000:
            raise PipelineError("Informe um texto de até 40 mil caracteres para gerar a voz.")
        reference = _input_path(job, "reference") or source
        reference_metadata = probe_video(reference)
        if not reference_metadata.get("audio"):
            raise PipelineError("A referência vocal precisa conter uma faixa de áudio.")
        source = reference
        metadata = reference_metadata
        input_files["reference"] = source
    for role in ("audio", "reference"):
        extra = _input_path(job, role)
        if extra and role not in input_files:
            if not probe_video(extra).get("audio"):
                raise PipelineError(f"O arquivo de {role} precisa conter áudio.")
            input_files[role] = extra
    if operation == "lipsync_video" and "audio" not in input_files:
        raise PipelineError("Envie o áudio que será sincronizado com o vídeo.")
    for role, path in input_files.items():
        # Fixed role names keep same-name uploads from colliding in the archive.
        inputs[role] = {"path": f"input/{role}{path.suffix.lower()}", "sha256": sha256_file(path)}
    primary_role = "reference" if operation == "text_to_voice" else "video"
    options = {key: job.options.get(key, default) for key, default in {
        "source_language": "auto", "require_lipsync": True, "preserve_ambience": True, "cache_on_drive": True,
        "review_mode": "automatic_optional", "performance_profile": "fast_lipsync",
        "speaker_count": "auto", "text": "", "face_selection": "auto", "subtitle_mode": "none",
    }.items()}
    if operation == "text_to_voice":
        options["require_lipsync"] = False
    if operation == "lipsync_video":
        options["require_lipsync"] = True
    if not isinstance(options["require_lipsync"], bool):
        raise PipelineError("A opção de lip-sync precisa ser verdadeira ou falsa, sem valores implícitos.")
    if not isinstance(options["cache_on_drive"], bool):
        raise PipelineError("A preferência de cache no Drive precisa ser verdadeira ou falsa.")
    manifest = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "job_id": job.id, "created_at": int(time.time()), "operation": operation,
        "target_languages": targets, "input_file": inputs[primary_role]["path"],
        "input_sha256": inputs[primary_role]["sha256"], "inputs": inputs,
        "options": options, "authorization": {"voice_clone_confirmed": consent},
        "input_metadata": metadata, "proxy_created": proxied,
        "original_name": job.options.get("original_name", job.input_path.name),
        "output_contract": {"max_resolution": "1080p", "duration_tolerance_seconds": 1.0},
    }
    manifest_path = job.directory / MANIFEST_NAME
    _write_json(manifest_path, manifest)
    package = job.directory / PACKAGE_NAME
    staged = job.directory / f".package-{uuid.uuid4().hex}.zip"
    try:
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(manifest_path, "manifest.json")
            for role, path in input_files.items():
                archive.write(path, inputs[role]["path"])
            # Bundle only public worker source; never settings or runtime files.
            for name in ("dubstudio_bootstrap.py", "dubstudio_runner_v2.py", "dubstudio_colab_worker.py", "dubstudio_translate.py"):
                path = ROOT / "colab" / name
                archive.write(path, f"worker/{path.name}")
            for name in ("requirements-app.txt", "requirements-musetalk.txt"):
                path = ROOT / "colab" / name
                archive.write(path, f"worker/{path.name}")
        os.replace(staged, package)
    finally:
        if staged.exists():
            staged.unlink()
    job.metadata["studio"] = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "manifest": manifest, "package_bytes": package.stat().st_size,
        "input_sha256": manifest["input_sha256"], "proxy_created": proxied,
    }
    return package


def _manifest(job: Job) -> dict[str, Any]:
    _check_identity(job)
    path = job.directory / MANIFEST_NAME
    if not path.exists():
        raise PipelineError("Prepare o pacote deste projeto antes de importar o resultado.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError("O manifesto local do projeto não é válido.") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION or payload.get("job_id") != job.id:
        raise PipelineError("O manifesto não corresponde a este projeto.")
    return payload


def _safe_path(value: Any) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 32 or char in '\\:<>"|?*' for char in value):
        raise PipelineError("O ZIP contém um caminho inseguro.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or any(part in {"", "."} for part in value.split("/")):
        raise PipelineError("O ZIP contém um caminho inseguro.")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.endswith((" ", ".")) or part.split(".")[0].upper() in reserved for part in path.parts):
        raise PipelineError("O ZIP contém um nome de arquivo inseguro no Windows.")
    return str(path)


def _safe_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    seen: set[str] = set()
    total = 0
    if len(archive.infolist()) > MAX_MEMBERS:
        raise PipelineError("O ZIP excede o limite seguro de arquivos.")
    for item in archive.infolist():
        raw_name = item.filename.rstrip("/") if item.is_dir() else item.filename
        name = _safe_path(raw_name)
        folded = name.casefold()
        if folded in seen:
            raise PipelineError("O ZIP contém nomes de arquivo duplicados.")
        seen.add(folded)
        if stat.S_IFMT(item.external_attr >> 16) not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise PipelineError("O ZIP contém um vínculo ou tipo de arquivo inseguro.")
        if item.flag_bits & 1:
            raise PipelineError("O ZIP de resultado não pode ser criptografado.")
        total += item.file_size
        if total > MAX_ZIP_BYTES or (item.file_size > 1024**2 and item.file_size > max(1, item.compress_size) * 1000):
            raise PipelineError("O ZIP excede o limite seguro de descompactação.")
        if not item.is_dir():
            members[name] = item
    file_names = {value.casefold() for value in members}
    for name in members:
        parent = PurePosixPath(name).parent
        while str(parent) != ".":
            if str(parent).casefold() in file_names:
                raise PipelineError("O ZIP contém arquivos que conflitam com pastas.")
            parent = parent.parent
    return members


def _read_json(archive: zipfile.ZipFile, members: dict[str, zipfile.ZipInfo], name: str) -> dict[str, Any]:
    item = members.get(name)
    if item is None or item.file_size > MAX_JSON_BYTES:
        raise PipelineError(f"O arquivo {name} está ausente ou excede o limite de 4 MB.")
    try:
        payload = json.loads(archive.read(item).decode("utf-8"))
    except (UnicodeError, ValueError, zipfile.BadZipFile) as exc:
        raise PipelineError(f"O arquivo {name} não contém JSON válido.") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"O arquivo {name} precisa conter um objeto JSON.")
    return payload


def _validate_identity(payload: dict[str, Any], job: Job, manifest: dict[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PipelineError("A versão do resultado não é compatível com o Studio.")
    if payload.get("job_id") != job.id:
        raise PipelineError("O resultado pertence a outro trabalho.")
    if payload.get("input_sha256") != manifest.get("input_sha256"):
        raise PipelineError("O resultado pertence a outro vídeo ou referência vocal.")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PipelineError(f"{label} possui um valor inválido.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PipelineError(f"{label} possui um valor inválido.") from None
    if not math.isfinite(number):
        raise PipelineError(f"{label} possui um valor inválido.")
    return number


def _segments(values: Any, duration: float | None = None) -> list[dict[str, Any]]:
    if not isinstance(values, list) or not values or len(values) > 20000:
        raise PipelineError("A revisão não contém trechos de fala válidos.")
    seen: set[str] = set()
    previous = -1.0
    for index, segment in enumerate(values):
        if not isinstance(segment, dict):
            raise PipelineError("Um trecho de fala não é válido.")
        start, end = _number(segment.get("start"), "O início do trecho"), _number(segment.get("end"), "O fim do trecho")
        if start < 0 or end <= start or start < previous or (duration and end > duration + 1.0):
            raise PipelineError("Os tempos da revisão não são válidos para este projeto.")
        previous = start
        identity = str(segment.get("id", index))
        if identity in seen:
            raise PipelineError("A revisão contém trechos duplicados.")
        seen.add(identity)
        if not isinstance(segment.get("text"), str) or not segment["text"].strip() or not isinstance(segment.get("translated_text"), str) or not segment["translated_text"].strip():
            raise PipelineError("Cada trecho precisa conter texto original e tradução.")
    return values


def _validate_lipsync(report: dict[str, Any], duration: float) -> None:
    proof = report.get("lipsync")
    if not isinstance(proof, dict) or proof.get("required") is not True or proof.get("confirmed") is not True:
        raise PipelineError("O resultado não confirma o lip-sync obrigatório.")
    ambiguity = proof.get("ambiguity", report.get("ambiguous_face_segments", []))
    if ambiguity != []:
        raise PipelineError("Há rostos falantes ambíguos no relatório de lip-sync.")
    skipped = report.get("no_lipsync_segments", [])
    if skipped:
        raise PipelineError("O resultado deixou trechos de rosto falante sem lip-sync.")
    segments = proof.get("confirmed_segments", report.get("lipsync_confirmed_segments", []))
    if not isinstance(segments, list):
        raise PipelineError("Os trechos confirmados de lip-sync não são válidos.")
    if not segments:
        # Voice-over with no visible speaking face may legitimately preserve every frame.
        if not (proof.get("no_visible_speaking_face") is True and proof.get("scene_analysis_complete") is True):
            raise PipelineError("Não há evidência de trechos faciais confirmados para o lip-sync.")
    previous_end = 0.0
    for segment in segments:
        if not isinstance(segment, dict):
            raise PipelineError("Um trecho confirmado de lip-sync não é válido.")
        start, end = _number(segment.get("start"), "O início do lip-sync"), _number(segment.get("end"), "O fim do lip-sync")
        if start < previous_end - 0.05 or start < 0 or end <= start or end > duration + 1.0:
            raise PipelineError("Os trechos faciais confirmados possuem tempos inseguros.")
        if not str(segment.get("coordinate_source") or proof.get("coordinate_source") or "").strip():
            raise PipelineError("A evidência de lip-sync não informa a origem das coordenadas faciais.")
        previous_end = end


def _validate_report(report: dict[str, Any], job: Job, manifest: dict[str, Any], language: str) -> None:
    _validate_identity(report, job, manifest)
    if report.get("pipeline_version") != PIPELINE_VERSION or report.get("target_language") != language:
        raise PipelineError("O relatório não corresponde ao idioma ou à versão do pipeline solicitado.")
    operation = manifest["operation"]
    if operation in {"dub_video", "text_to_voice"}:
        proof = report.get("voice_clone")
        if manifest.get("authorization", {}).get("voice_clone_confirmed") is not True or not isinstance(proof, dict) or proof.get("required") is not True or proof.get("confirmed") is not True or proof.get("authorized") is not True:
            raise PipelineError("O resultado não confirma a preservação autorizada do timbre original.")
    if manifest["options"].get("require_lipsync") is True:
        _validate_lipsync(report, float(manifest["input_metadata"]["duration"]))
    else:
        proof = report.get("lipsync")
        if not isinstance(proof, dict) or proof.get("required") is not False or proof.get("confirmed") is not False:
            raise PipelineError("O relatório precisa declarar que o lip-sync foi desativado, sem confirmar uma etapa não solicitada.")


def _probe_streams(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if not executable:
        raise PipelineError("FFprobe é necessário para validar o resultado recebido.")
    result = subprocess.run([
        executable, "-v", "error", "-show_entries",
        "format=duration,format_name:stream=codec_type,codec_name,duration,width,height,sample_rate,channels", "-of", "json", str(path),
    ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
    try:
        payload = json.loads(result.stdout)
    except ValueError:
        raise PipelineError("O resultado recebido não é uma mídia válida.") from None
    if result.returncode:
        raise PipelineError("O resultado recebido não é uma mídia válida.")
    streams = payload.get("streams", [])
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    # An audio-only container may omit stream duration (WAV). Video cannot use
    # container duration: long audio can conceal an image ending many seconds early.
    video_duration = _number(video.get("duration"), "A duração da faixa de vídeo") if video else None
    audio_duration = _number(audio.get("duration", payload.get("format", {}).get("duration")), "A duração da faixa de áudio") if audio else None
    return {"video": video, "audio": audio, "video_duration": video_duration, "audio_duration": audio_duration,
            "format_name": payload.get("format", {}).get("format_name", "")}


def _validate_media(path: Path, role: str, job: Job, manifest: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    media = _probe_streams(path)
    if not media["audio"] or not media["audio_duration"] or media["audio_duration"] <= 0:
        raise PipelineError("O resultado não contém uma faixa de áudio válida.")
    if role == "video":
        expected = float(manifest["input_metadata"]["duration"])
        if not media["video"] or abs(media["video_duration"] - expected) > 1.0:
            raise PipelineError("A duração da faixa de vídeo ficou fora da tolerância de 1 segundo.")
        if abs(media["audio_duration"] - expected) > 1.0 or media["audio_duration"] < media["video_duration"] - 0.5:
            raise PipelineError("A duração da faixa de áudio ficou fora da tolerância ou termina antes do vídeo.")
        stream = media["video"]
        original = manifest["input_metadata"]["video"]
        width, height = int(stream.get("width", 0)), int(stream.get("height", 0))
        cap_width, cap_height = target_dimensions(int(original.get("width", 0)), int(original.get("height", 0)))
        if not 0 < width <= cap_width or not 0 < height <= cap_height:
            raise PipelineError("O resultado foi ampliado ou excede a resolução máxima de 1080p.")
        if abs(width / height - int(original["width"]) / int(original["height"])) > 0.03:
            raise PipelineError("O resultado alterou a proporção do vídeo original.")
        if stream.get("codec_name") != "h264" or media["audio"].get("codec_name") != "aac":
            raise PipelineError("O vídeo final precisa usar H.264 com áudio AAC.")
        if "mp4" not in media["format_name"].split(","):
            raise PipelineError("O vídeo final precisa ser um MP4 real, não apenas um arquivo renomeado.")
        original_audio = job.input_path
        if manifest["operation"] == "lipsync_video":
            original_audio = _input_path(job, "audio") or original_audio
        if probe_video(original_audio).get("audio"):
            tail = min(5.0, max(0.5, expected * 0.1))
            if _tail_max_volume(original_audio, tail) > -45 and _tail_max_volume(path, tail) < -55:
                raise PipelineError("O áudio ficou silencioso no final apesar de existir fala ou áudio no original.")
    else:
        if media["video"] or media["format_name"] != role or (role == "mp3" and media["audio"].get("codec_name") != "mp3"):
            raise PipelineError(f"O áudio {role.upper()} precisa ter o formato real solicitado, sem faixa de vídeo.")
        duration = report.get("summary", {}).get("duration")
        if duration is None or abs(_number(duration, "A duração declarada") - media["audio_duration"]) > 1.0:
            raise PipelineError("A duração do áudio não corresponde ao relatório.")
    return media


def _current_approved_review(job: Job, manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Bind imports to the latest local approval, not just the media/job ID."""
    path = job.directory / "review-approved.json"
    if manifest["operation"] != "dub_video":
        return None
    if not path.exists():
        if job.metadata.get("review_approved"):
            raise PipelineError("A revisão aprovada atual não foi encontrada; reexporte-a antes de importar.")
        return None
    try:
        approved = _validate_review(json.loads(path.read_text(encoding="utf-8")), job, manifest)
        original = _validate_review(json.loads((job.directory / "studio-review.json").read_text(encoding="utf-8")), job, manifest)
    except (OSError, ValueError) as exc:
        raise PipelineError("Os arquivos locais da revisão aprovada não estão íntegros.") from exc
    if approved["source_language"] != original["source_language"]:
        raise PipelineError("A aprovação atual não corresponde ao idioma original da revisão.")
    for language in manifest["target_languages"]:
        _match_approved_segments(approved["translations"][language], original["translations"][language], translated=False)
    snapshot_hash = approved.get("source_snapshot_sha256")
    if snapshot_hash is not None:
        try:
            # Text-mode newline normalization preserves the JSON digest used in
            # ZIP packages when this local checkpoint was written on Windows.
            snapshot_text = (job.directory / "review-source.json").read_text(encoding="utf-8")
            snapshot = json.loads(snapshot_text)
        except (OSError, ValueError) as exc:
            raise PipelineError("O snapshot original aprovado não foi encontrado ou está inválido.") from exc
        if hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest() != snapshot_hash:
            raise PipelineError("O snapshot original aprovado foi alterado.")
        _validate_identity(snapshot, job, manifest)
        if snapshot.get("pipeline_version") != PIPELINE_VERSION or snapshot.get("source_language") != approved["source_language"]:
            raise PipelineError("O snapshot aprovado não corresponde ao pipeline ou idioma original.")
        _match_approved_segments(snapshot.get("segments"), original["translations"][manifest["target_languages"][0]], translated=False)
    return approved


def _match_approved_segments(received: Any, approved: list[dict], *, translated: bool = True) -> None:
    if not isinstance(received, list) or len(received) != len(approved):
        raise PipelineError("O resultado não contém todos os trechos da revisão aprovada atual.")
    keys = ("id", "start", "end", "speaker", "text", "translated_text") if translated else ("id", "start", "end", "speaker", "text")
    for actual, expected in zip(received, approved):
        if not isinstance(actual, dict) or type(actual.get("id")) is not type(expected["id"]) or any(isinstance(actual.get(key), bool) for key in ("start", "end")) or any(actual.get(key) != expected.get(key) for key in keys):
            raise PipelineError("O resultado diverge da revisão aprovada atual; retome o pacote revisado para gerar a versão correta.")


def import_studio_result(job: Job, result_zip: Path) -> dict[str, Any]:
    """Validate an offline result atomically, with independent A/V measurements."""
    manifest = _manifest(job)
    if not zipfile.is_zipfile(result_zip):
        raise PipelineError("O arquivo importado não é um ZIP válido.")
    approved_review = _current_approved_review(job, manifest)
    output_records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="studio-validate-", dir=job.directory) as name:
        staging = Path(name)
        with zipfile.ZipFile(result_zip) as archive:
            members = _safe_members(archive)
            result = _read_json(archive, members, "result.json")
            _validate_identity(result, job, manifest)
            if result.get("pipeline_version") != PIPELINE_VERSION:
                raise PipelineError("A versão do pipeline do resultado é incompatível.")
            outputs = result.get("outputs")
            if not isinstance(outputs, list) or len(outputs) != len(manifest["target_languages"]):
                raise PipelineError("O resultado não contém um estado para cada idioma solicitado.")
            seen_languages: set[str] = set()
            for output in outputs:
                if not isinstance(output, dict) or output.get("language") not in manifest["target_languages"] or output["language"] in seen_languages:
                    raise PipelineError("O resultado contém um idioma inesperado ou duplicado.")
                language = output["language"]
                seen_languages.add(language)
                status = output.get("status")
                if status == "failed":
                    if not isinstance(output.get("error"), str) or not output["error"].strip():
                        raise PipelineError("Um idioma falhou sem explicar o motivo.")
                    output_records.append({"language": language, "status": "failed", "error": output["error"][:4000], "files": {}})
                    continue
                if status != "complete":
                    raise PipelineError("O resultado contém um estado de idioma inválido.")
                files, checksums = output.get("files"), output.get("checksums")
                if not isinstance(files, dict) or not isinstance(checksums, dict):
                    raise PipelineError("Faltam arquivos ou checksums do resultado.")
                required = {"video", "report"} if manifest["operation"] != "text_to_voice" else {"wav", "mp3", "report", "segments"}
                if manifest["operation"] == "dub_video":
                    required.update({"segments", "original_srt", "translation_srt"})
                if not required.issubset(files):
                    raise PipelineError("O resultado está incompleto para a operação solicitada.")
                paths: dict[str, str] = {}
                for role, value in files.items():
                    if role not in FILE_EXTENSIONS:
                        raise PipelineError("O resultado contém um tipo de arquivo inesperado.")
                    path = _safe_path(value)
                    if not path.startswith(f"outputs/{language}/") or PurePosixPath(path).suffix.lower() not in FILE_EXTENSIONS[role] or path not in members or path in paths.values() or members[path].file_size == 0:
                        raise PipelineError("Um arquivo de resultado não corresponde ao idioma ou tipo esperado.")
                    checksum = checksums.get(path)
                    if not isinstance(checksum, str) or not _SHA_RE.fullmatch(checksum):
                        raise PipelineError("Um arquivo de resultado não tem checksum SHA-256 válido.")
                    target = staging / path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(members[path]) as incoming, target.open("wb") as destination:
                        digest = hashlib.sha256()
                        for chunk in iter(lambda: incoming.read(1024**2), b""):
                            digest.update(chunk)
                            destination.write(chunk)
                    if digest.hexdigest() != checksum:
                        raise PipelineError("Um arquivo de resultado falhou na verificação de integridade SHA-256.")
                    paths[role] = path
                if set(checksums) != set(paths.values()):
                    raise PipelineError("Os checksums não correspondem exatamente aos arquivos declarados.")
                report = _read_json(archive, members, paths["report"])
                _validate_report(report, job, manifest, language)
                evidence = None
                if "segments" in paths:
                    evidence = _read_json(archive, members, paths["segments"])
                    if evidence.get("target_language") != language:
                        raise PipelineError("Os segmentos traduzidos pertencem a outro idioma.")
                    evidence_duration = (_number(report.get("summary", {}).get("duration"), "A duração da voz gerada")
                                         if manifest["operation"] == "text_to_voice" else float(manifest["input_metadata"]["duration"]))
                    _segments(evidence.get("segments"), evidence_duration)
                    if approved_review:
                        _match_approved_segments(evidence["segments"], approved_review["translations"][language])
                for role in ("original_srt", "translation_srt"):
                    if role in paths:
                        try:
                            if "-->" not in (staging / paths[role]).read_text(encoding="utf-8"):
                                raise ValueError("empty subtitle")
                        except (UnicodeError, ValueError) as exc:
                            raise PipelineError("Uma evidência de legendas está vazia ou não é um SRT válido.") from exc
                media = {role: _validate_media(staging / paths[role], role, job, manifest, report) for role in ("video", "wav", "mp3") if role in paths}
                if manifest["operation"] == "text_to_voice" and evidence:
                    actual_duration = media["wav"]["audio_duration"]
                    if any(_number(item["end"], "O fim da fala gerada") > actual_duration + 0.01 for item in evidence["segments"]):
                        raise PipelineError("Os tempos da voz gerada excedem a duração real do áudio.")
                output_records.append({"language": language, "status": "complete", "files": paths, "checksums": checksums, "summary": report.get("summary", {}), "validation": {
                    "validated_at": int(time.time()), "media": media,
                    "authorized_clone": manifest["operation"] in {"dub_video", "text_to_voice"},
                    "lipsync_requested": manifest["options"].get("require_lipsync") is True,
                    "lipsync_confirmed": manifest["options"].get("require_lipsync") is True,
                    "duration_tolerance_seconds": 1.0,
                }})
        successes = [record for record in output_records if record["status"] == "complete"]
        destination = job.directory / "studio-results" / uuid.uuid4().hex
        if successes:
            destination.parent.mkdir(exist_ok=True)
            shutil.move(str(staging), str(destination))
            prefix = destination.relative_to(job.directory).as_posix()
            for record in successes:
                record["files"] = {role: f"{prefix}/{path}" for role, path in record["files"].items()}
            first = successes[0]
            job.output_path = job.directory / (first["files"].get("video") or first["files"]["wav"])
            job.report_path = job.directory / first["files"]["report"]
    imported = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "job_id": job.id, "outputs": output_records, "success_count": len(successes), "failed_count": len(output_records) - len(successes), "imported_at": int(time.time())}
    previous = job.metadata.get("studio_result")
    if previous:
        job.metadata.setdefault("studio_result_history", []).append(previous)
    job.metadata["studio_result"] = imported
    job.metadata["studio_results"] = output_records
    return imported


def _validate_review(payload: dict[str, Any], job: Job, manifest: dict[str, Any]) -> dict[str, Any]:
    _validate_identity(payload, job, manifest)
    if payload.get("pipeline_version") != PIPELINE_VERSION:
        raise PipelineError("A revisão pertence a uma versão de pipeline incompatível.")
    translations = payload.get("translations")
    if not isinstance(translations, dict) or set(translations) != set(manifest["target_languages"]):
        raise PipelineError("A revisão precisa conter os idiomas solicitados no projeto.")
    if not isinstance(payload.get("source_language"), str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", payload["source_language"]):
        raise PipelineError("A revisão não informa o idioma original.")
    duration = float(manifest["input_metadata"]["duration"])
    original_segments = None
    for segments in translations.values():
        _segments(segments, duration)
        for segment in segments:
            if any(isinstance(segment.get(key), bool) or not isinstance(segment.get(key), (int, float)) for key in ("start", "end")):
                raise PipelineError("A revisão não possui tempos numéricos válidos.")
            if isinstance(segment.get("id"), bool) or not isinstance(segment.get("id"), (int, str)):
                raise PipelineError("A revisão não possui identificadores de trecho válidos.")
            if not isinstance(segment.get("speaker"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", segment["speaker"]):
                raise PipelineError("A revisão não possui locutores válidos.")
        immutable = [{key: segment[key] for key in ("id", "start", "end", "speaker", "text")} for segment in segments]
        if original_segments is not None and immutable != original_segments:
            raise PipelineError("As versões por idioma não compartilham a mesma transcrição original.")
        original_segments = immutable
    return payload


def import_review(job: Job, review_zip: Path) -> dict[str, Any]:
    manifest = _manifest(job)
    if manifest["operation"] != "dub_video":
        raise PipelineError("Esta operação não possui revisão de tradução.")
    if not zipfile.is_zipfile(review_zip):
        raise PipelineError("A revisão precisa ser um ZIP válido.")
    with zipfile.ZipFile(review_zip) as archive:
        members = _safe_members(archive)
        payload = _validate_review(_read_json(archive, members, "review.json"), job, manifest)
    _write_json(job.directory / "studio-review.json", payload)
    job.metadata["review"] = payload
    return payload


def export_approved_review(job: Job, payload: dict[str, Any]) -> Path:
    manifest = _manifest(job)
    path = job.directory / "studio-review.json"
    if not path.exists():
        raise PipelineError("Importe a revisão do notebook antes de aprovar os textos.")
    original = _validate_review(json.loads(path.read_text(encoding="utf-8")), job, manifest)
    approved = _validate_review(payload, job, manifest)
    if approved["source_language"] != original["source_language"]:
        raise PipelineError("A revisão aprovada alterou o idioma original.")
    for language, segments in approved["translations"].items():
        source_segments = original["translations"][language]
        if len(segments) != len(source_segments):
            raise PipelineError("A revisão alterou a quantidade de trechos originais.")
        for segment, source in zip(segments, source_segments):
            if any(segment.get(key) != source.get(key) for key in ("id", "start", "end", "speaker", "text")):
                raise PipelineError("Altere somente a tradução; os tempos, locutores e textos originais devem permanecer iguais.")
    source_segments = original["translations"][manifest["target_languages"][0]]
    snapshot = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
        "job_id": job.id, "input_sha256": manifest["input_sha256"],
        "source_language": original["source_language"],
        "segments": [{key: segment[key] for key in ("id", "start", "end", "speaker", "text")} for segment in source_segments],
    }
    source_encoded = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)
    source_checksum = hashlib.sha256(source_encoded.encode("utf-8")).hexdigest()
    approved = {**approved, "approved_at": int(time.time()), "pipeline_version": PIPELINE_VERSION,
                "source_snapshot_sha256": source_checksum}
    encoded = json.dumps(approved, ensure_ascii=False, indent=2, allow_nan=False)
    if max(len(encoded.encode("utf-8")), len(source_encoded.encode("utf-8"))) > MAX_JSON_BYTES:
        raise PipelineError("A revisão aprovada excede o limite de 4 MB.")
    resume_manifest = {**manifest, "review_snapshot": {
        "path": "review-source.json", "sha256": source_checksum,
        "approved_path": "review-approved.json", "approved_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    }}
    public_workers = {f"worker/{name}": ROOT / "colab" / name for name in (
        "dubstudio_bootstrap.py", "dubstudio_runner_v2.py", "dubstudio_colab_worker.py", "dubstudio_translate.py",
        "requirements-app.txt", "requirements-musetalk.txt",
    )}
    base_package = job.directory / PACKAGE_NAME
    if not zipfile.is_zipfile(base_package):
        raise PipelineError("O pacote original deste projeto não foi encontrado para a retomada.")
    output = job.directory / "pacote-revisado-colab.zip"
    staged = job.directory / f".approved-package-{uuid.uuid4().hex}.zip"
    try:
        with zipfile.ZipFile(base_package) as original_archive:
            members = _safe_members(original_archive)
            packaged_manifest = _read_json(original_archive, members, "manifest.json")
            if packaged_manifest != manifest:
                raise PipelineError("O pacote original foi alterado e não corresponde a esta revisão.")
            for entry in manifest["inputs"].values():
                item = members.get(_safe_path(entry["path"]))
                if item is None:
                    raise PipelineError("Um arquivo original está ausente no pacote de retomada.")
                digest = hashlib.sha256()
                with original_archive.open(item) as source:
                    for chunk in iter(lambda: source.read(1024**2), b""):
                        digest.update(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise PipelineError("Um arquivo original foi alterado no pacote de retomada.")
            with zipfile.ZipFile(staged, "w", zipfile.ZIP_STORED, allowZip64=True) as destination_archive:
                for name, member in members.items():
                    if name in {"manifest.json", "review-approved.json", "review-source.json"} or name in public_workers:
                        continue
                    with original_archive.open(member) as source, destination_archive.open(name, "w", force_zip64=True) as destination:
                        shutil.copyfileobj(source, destination, 1024**2)
                destination_archive.writestr("review-approved.json", encoded)
                destination_archive.writestr("review-source.json", source_encoded)
                destination_archive.writestr("manifest.json", json.dumps(resume_manifest, ensure_ascii=False, indent=2, allow_nan=False))
                for name, file in public_workers.items():
                    destination_archive.write(file, name)
        os.replace(staged, output)
    finally:
        if staged.exists():
            staged.unlink()
    _write_json(job.directory / "review-approved.json", approved)
    _write_json(job.directory / "review-source.json", snapshot)
    job.metadata["review_approved"] = {"approved_at": approved["approved_at"], "languages": list(approved["translations"]),
                                       "source_snapshot_sha256": source_checksum}
    return output
