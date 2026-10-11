#!/usr/bin/env python3
"""Orquestrador Colab v2: etapas isoladas, retomada íntegra e saídas por idioma.

Modelos vivem em processos diferentes, portanto VRAM é liberada entre etapas.
Computação ocorre em /content; Drive recebe somente cache e checkpoints.
"""
from __future__ import annotations

import argparse
import ast
from array import array
import copy
import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import textwrap
import time
import uuid
import wave
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "2.0"
PIPELINE_VERSION = "3.0.0"
OPERATIONS = {"dub_video", "lipsync_video", "text_to_voice"}
CLONE_LANGUAGES = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nb", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}
RAW_SYNTHESIS_CACHE_REVISION = "chatterbox-raw-v1-asset-source-reference-sha256"
MAX_VOICE_TEMPO_FACTOR = 2.5


class PipelineError(RuntimeError):
    def __init__(self, message: str, *, diagnostic: dict | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or {}


def safe_error_text(value: object, limit: int = 1200) -> str:
    """Export one bounded error, never an arbitrary private subprocess log tail."""
    text = str(value)[:8192]
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = re.sub(r"https?://[^\s\"'<>]+", "[URL omitida]", text, flags=re.I)
    text = re.sub(r"\b(?:Bearer|Basic)\s+[^\s,;]+", "[credencial omitida]", text, flags=re.I)
    text = re.sub(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-|AIza)[A-Za-z0-9_-]+", "[credencial omitida]", text)
    text = re.sub(r"(?i)([\"']?\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|authorization|cookie|credential)[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)", r"\1[credencial omitida]", text)
    text = re.sub(r"\b[A-Za-z]:[\\/][^\s\"'<>]+", "[caminho local omitido]", text)
    text = re.sub(r"(?<!\w)/(?:content|home|root|tmp|workspace|Users)/[^\s\"'<>]+", "[caminho local omitido]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email omitido]", text)
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = " ".join(text.split())
    return text[:limit] if text else "A etapa falhou sem diagnóstico público seguro."


def failure_diagnostic(exc: Exception) -> dict:
    """Keep only our small public diagnostic schema, including safe face intervals."""
    raw = getattr(exc, "diagnostic", {})
    if not isinstance(raw, dict):
        raw = {}
    result = {"reason": safe_error_text(raw.get("reason", exc))}
    for key in ("stage", "stage_label", "log_file"):
        if isinstance(raw.get(key), str):
            result[key] = safe_error_text(raw[key], 160)
    if type(raw.get("exit_code")) is int:
        result["exit_code"] = raw["exit_code"]
    if isinstance(raw.get("ambiguity"), list):
        intervals = []
        for item in raw["ambiguity"][:64]:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 1200:
                continue
            intervals.append({"start": round(start, 3), "end": round(end, 3), "reason": safe_error_text(item.get("reason", "Ambiguidade facial detectada"), 240)})
        if intervals:
            result["ambiguity"] = intervals
            total = raw.get("ambiguity_total_intervals")
            result["ambiguity_total_intervals"] = max(len(intervals), total) if type(total) is int and 0 <= total <= 30000 else len(intervals)
            result["ambiguity_truncated"] = result["ambiguity_total_intervals"] > len(intervals)
    return result


def analysis_ambiguity_diagnostic(config: dict) -> dict:
    """Read only bounded analysis JSON, excluding boxes, pickles and private logs."""
    if config.get("stage") != "analysis" or not config.get("output"):
        return {}
    path = Path(config["output"])
    directory = Path(config["directory"])
    try:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(directory.resolve()) or path.stat().st_size > 2 * 1024**2:
            return {}
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    raw = payload.get("ambiguity") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return {}
    allowed_reasons = {"múltiplos rostos candidatos", "trecho facial curto demais para confirmar", "uma trilha facial contém mais de um locutor; revise a associação", "seleção facial inválida", "a área selecionada não identifica um único rosto detectado; ajuste a seleção"}
    intervals = []
    for item in raw[:30000]:
        if not isinstance(item, dict):
            continue
        start, end = item.get("start"), item.get("end")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= 1200:
            continue
        reason = item.get("reason")
        intervals.append({"start": float(start), "end": float(end), "reason": reason if isinstance(reason, str) and reason in allowed_reasons else "Ambiguidade facial detectada"})
    intervals.sort(key=lambda item: (item["start"], item["end"]))
    merged = []
    for item in intervals:
        if merged and item["reason"] == merged[-1]["reason"] and item["start"] <= merged[-1]["end"] + 0.001:
            merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        else:
            merged.append(dict(item))
    return {"ambiguity": merged[:64], "ambiguity_total_intervals": len(merged)} if merged else {}


def stage_failure(exc: Exception, key: str, label: str, config: dict) -> PipelineError:
    diagnostic = failure_diagnostic(exc)
    diagnostic.update(stage=safe_error_text(key, 80), stage_label=safe_error_text(label, 160))
    diagnostic.update(analysis_ambiguity_diagnostic(config))
    diagnostic = failure_diagnostic(PipelineError(diagnostic["reason"], diagnostic=diagnostic))
    message = f"{diagnostic['stage_label']}: {diagnostic['reason']}"
    if diagnostic.get("ambiguity"):
        windows = ", ".join(f"{item['start']:.2f}–{item['end']:.2f}s" for item in diagnostic["ambiguity"][:8])
        message += f" Intervalos para revisar: {windows}."
        if len(diagnostic["ambiguity"]) > 8 or diagnostic.get("ambiguity_truncated"):
            message += " Veja os demais intervalos no relatório."
    return PipelineError(safe_error_text(message, 1800), diagnostic=diagnostic)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def worker_fingerprint() -> str:
    """A repaired worker cannot silently reuse checkpoints from older code."""
    files = [Path(__file__), Path(__file__).with_name("dubstudio_colab_worker.py"), Path(__file__).with_name("dubstudio_translate.py"), Path(__file__).with_name("dubstudio_audio_master.py")]
    return fingerprint({file.name: sha256(file) for file in files if file.is_file()})


def needs_lipsync(manifest: dict) -> bool:
    operation = manifest.get("operation")
    return operation == "lipsync_video" or (operation == "dub_video" and manifest.get("options", {}).get("require_lipsync", True))


def needs_face_analysis(manifest: dict) -> bool:
    return needs_lipsync(manifest) or (manifest.get("operation") == "dub_video" and manifest.get("options", {}).get("timing_mode") == "audio_master")


def public_spatial_composition(value: object, expected_segments: int) -> dict | None:
    """Publish only the renderer's bounded composition contract, not private logs."""
    if value is None:
        return None  # Legacy proof has no spatial claim; never invent one.
    policy = value.get('policy') if isinstance(value, dict) else None
    mapped = policy == "perioral-mapped-source-dwpose68-inner-feather-v1"
    clock = "mapped-source-frame-clock" if mapped else "original-frame-clock"
    if (not isinstance(value, dict) or policy not in {"tracked-face-box-only-inner-feather-v1", "perioral-source-dwpose68-inner-feather-v1", "perioral-mapped-source-dwpose68-inner-feather-v1"}
            or value.get("outside_face_source") != clock
            or type(value.get("mask_outside_tracked_box")) is not int
            or value["mask_outside_tracked_box"] != 0
            or value.get("mask_expands_box") is not False
            or type(value.get("segments")) is not int
            or type(expected_segments) is not int
            or not 1 <= expected_segments <= 30000
            or value["segments"] != expected_segments):
        raise PipelineError("A composição espacial do lip-sync não possui um contrato facial válido.")
    result = {"policy": policy, "outside_face_source": clock,
            "mask_outside_tracked_box": 0, "mask_expands_box": False,
            "segments": expected_segments,
            "encoding_note": "H.264 lossy encoding may alter decoded pixels; generated content is spatially confined."}
    if mapped:
        for key in ("source_video_sha256", "timeline_sha256", "output_base_sha256", "audio_master_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(key, ""))):
                raise PipelineError("A composição remapeada está sem vínculo com a fonte e a montagem.")
            result[key] = value[key]
    if policy in {'perioral-source-dwpose68-inner-feather-v1', 'perioral-mapped-source-dwpose68-inner-feather-v1'}:
        if (value.get('region') != 'mouth-and-lower-nose' or value.get('source_landmarks') != 'dwpose68-pinned'
                or type(value.get('eyes_forehead_alpha')) is not int or value['eyes_forehead_alpha'] != 0
                or value.get('human_visual_review') != 'pending'):
            raise PipelineError('A composição espacial perioral não confirma landmarks e proteção de olhos/testa.')
        records = value.get('evidence')
        hashes = ('source_sha256', 'mask_sha256', 'output_sha256', 'model_sha256', 'landmarks_sha256')
        if not isinstance(records, list) or len(records) != expected_segments:
            raise PipelineError('A composição espacial perioral está sem evidências por janela.')
        public_records = []
        for index, record in enumerate(records, 1):
            if (not isinstance(record, dict) or type(record.get('index')) is not int or record['index'] != index
                    or type(record.get('frame_count')) is not int or not 1 <= record['frame_count'] <= 30000
                    or any(not re.fullmatch(r'[0-9a-f]{64}', str(record.get(key, ''))) for key in hashes)):
                raise PipelineError('A composição espacial perioral possui evidência incompleta ou trocada.')
            public_records.append({'index': index, 'frame_count': record['frame_count'],
                                   **{key: record[key] for key in hashes}})
        if len({record['source_sha256'] for record in public_records}) != 1:
            raise PipelineError('A composição espacial perioral mistura vídeos de origem.')
        result.update(region='mouth-and-lower-nose', source_landmarks='dwpose68-pinned',
                      eyes_forehead_alpha=0, human_visual_review='pending', evidence=public_records)
    return result


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def safe_path(root: Path, relative: str) -> Path:
    path = PurePosixPath(str(relative).replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise PipelineError("O pacote contém caminho inseguro.")
    resolved = (root / str(path)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PipelineError("O arquivo está fora do pacote.")
    return resolved


def unpack_package(archive_path: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) > 10000 or sum(info.file_size for info in infos) > 15 * 1024**3:
            raise PipelineError("O pacote é maior que o limite seguro do notebook.")
        names: set[str] = set()
        for info in infos:
            safe_path(destination, info.filename)
            if info.filename in names or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise PipelineError("O pacote contém arquivo repetido ou link simbólico.")
            names.add(info.filename)
        archive.extractall(destination)
    return validate_manifest(destination)


def validate_manifest(package: Path) -> dict:
    manifest = read_json(package / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("pipeline_version") != PIPELINE_VERSION:
        raise PipelineError("Pacote incompatível. Exporte novamente pelo Studio atualizado.")
    if not re.fullmatch(r"[0-9a-f]{12}", str(manifest.get("job_id", ""))):
        raise PipelineError("Identificador de projeto inválido.")
    if manifest.get("operation") not in OPERATIONS:
        raise PipelineError("Operação não suportada.")
    if manifest.get("authorization", {}).get("voice_clone_confirmed") is not True:
        raise PipelineError("A autorização de uso da voz deve ser confirmada no Studio.")
    targets = manifest.get("target_languages")
    if not isinstance(targets, list) or not targets or len(targets) > 32 or len(targets) != len(set(targets)):
        raise PipelineError("Selecione pelo menos um idioma de destino sem repetições.")
    for target in targets:
        if not isinstance(target, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", target) or (manifest["operation"] != "lipsync_video" and target.split("-")[0] not in CLONE_LANGUAGES):
            raise PipelineError(f"Preservação de timbre não disponível para {target}. Nenhuma voz genérica será usada.")
    inputs = manifest.get("inputs", {})
    required = {"dub_video": {"video"}, "lipsync_video": {"video", "audio"}, "text_to_voice": {"reference"}}[manifest["operation"]]
    if not isinstance(inputs, dict) or not required.issubset(inputs):
        raise PipelineError("O pacote não possui todas as mídias da operação.")
    for item in inputs.values():
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise PipelineError("Uma mídia está sem caminho/checksum.")
        file = safe_path(package, item["path"])
        if not file.is_file() or sha256(file) != item.get("sha256"):
            raise PipelineError(f"Checksum divergente na mídia {file.name}.")
    primary = safe_path(package, manifest.get("input_file", ""))
    if not primary.is_file() or sha256(primary) != manifest.get("input_sha256"):
        raise PipelineError("O checksum principal não corresponde ao projeto.")
    if manifest["operation"] == "text_to_voice" and not str(manifest.get("options", {}).get("text", "")).strip():
        raise PipelineError("Digite o texto para gerar a voz.")
    if "require_lipsync" in manifest.get("options", {}) and not isinstance(manifest["options"]["require_lipsync"], bool):
        raise PipelineError("A opção de lip-sync precisa estar ativada ou desativada explicitamente.")
    if manifest.get("options", {}).get("timing_mode", "source_video") not in {"source_video", "audio_master"}:
        raise PipelineError("O modo temporal do pacote não é suportado.")
    return manifest


def validate_review(review: dict, manifest: dict, source_segments: list[dict]) -> dict[str, list[dict]]:
    if review.get("schema_version") != SCHEMA_VERSION or review.get("job_id") != manifest["job_id"] or review.get("input_sha256") != manifest["input_sha256"]:
        raise PipelineError("A revisão pertence a outro projeto ou vídeo.")
    translations = review.get("translations", {})
    if not isinstance(translations, dict) or set(translations) != set(manifest["target_languages"]):
        raise PipelineError("A revisão não corresponde aos idiomas solicitados.")
    for language in manifest["target_languages"]:
        items = translations.get(language)
        if not isinstance(items, list) or len(items) != len(source_segments):
            raise PipelineError(f"A revisão de {language} não contém todos os segmentos.")
        for original, revised in zip(source_segments, items):
            try:
                if not isinstance(revised, dict) or any(isinstance(revised.get(key), bool) for key in ("start", "end")):
                    raise ValueError()
                revised_times = [float(revised.get(key, -1)) for key in ("start", "end")]
            except (TypeError, ValueError) as exc:
                raise PipelineError("A revisão alterou identificadores ou tempos da transcrição.") from exc
            if type(revised.get("id")) is not type(original.get("id")) or revised.get("id") != original.get("id") or not all(math.isfinite(value) for value in revised_times) or any(abs(value - float(original[key])) > 0.002 for value, key in zip(revised_times, ("start", "end"))):
                raise PipelineError("A revisão alterou identificadores ou tempos da transcrição.")
            if not isinstance(revised.get("translated_text"), str) or not revised["translated_text"].strip():
                raise PipelineError("A revisão contém fala traduzida vazia.")
            if any(revised.get(key) != original.get(key) for key in ("speaker", "text")):
                raise PipelineError("A revisão alterou locutores ou o texto original.")
    return translations


def load_approved_snapshot(package: Path, manifest: dict, duration: float) -> dict | None:
    """An approved transcript is frozen input, not a fresh ASR prediction."""
    review_path = package / "review-approved.json"
    contract = manifest.get("review_snapshot")
    if not review_path.exists() and contract is None:
        return None
    if manifest["operation"] != "dub_video" or not isinstance(contract, dict):
        raise PipelineError("Reexporte a revisão pelo Studio atualizado: falta o snapshot verificável da transcrição original.")
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError("A duração real da mídia não pôde ser confirmada para esta revisão.")
    payloads = {}
    for role, path_key, hash_key in (("source", "path", "sha256"), ("approved", "approved_path", "approved_sha256")):
        expected_path = "review-source.json" if role == "source" else "review-approved.json"
        if contract.get(path_key) != expected_path or not re.fullmatch(r"[0-9a-f]{64}", str(contract.get(hash_key, ""))):
            raise PipelineError("O contrato do snapshot da revisão é inválido.")
        file = safe_path(package, expected_path)
        if not file.is_file() or file.stat().st_size > 4 * 1024**2 or sha256(file) != contract[hash_key]:
            raise PipelineError("Checksum divergente no snapshot ou na revisão aprovada.")
        payload = read_json(file)
        if not isinstance(payload, dict) or any(payload.get(key) != manifest.get(key) for key in ("schema_version", "pipeline_version", "job_id", "input_sha256")):
            raise PipelineError("O snapshot da revisão pertence a outro projeto, mídia ou pipeline.")
        payloads[role] = payload
    snapshot, review = payloads["source"], payloads["approved"]
    source_language = snapshot.get("source_language")
    if not isinstance(source_language, str) or not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", source_language) or review.get("source_language") != source_language:
        raise PipelineError("O idioma original do snapshot e da revisão é divergente.")
    requested_source = manifest.get("options", {}).get("source_language", "auto")
    if requested_source != "auto" and str(requested_source).split("-")[0].lower() != source_language.split("-")[0].lower():
        raise PipelineError("O snapshot não corresponde ao idioma original configurado.")
    if review.get("source_snapshot_sha256") != contract["sha256"]:
        raise PipelineError("A revisão aprovada não está vinculada ao snapshot original.")
    source_segments = snapshot.get("segments")
    if not isinstance(source_segments, list) or not source_segments or len(source_segments) > 20000:
        raise PipelineError("O snapshot não contém todos os trechos originais.")
    previous, seen = -1.0, set()
    for segment in source_segments:
        if not isinstance(segment, dict):
            raise PipelineError("Um trecho do snapshot é inválido.")
        if isinstance(segment.get("id"), bool) or not isinstance(segment.get("id"), (int, str)):
            raise PipelineError("Identificador de trecho do snapshot inválido.")
        identity = (type(segment["id"]).__name__, str(segment["id"]))
        if identity in seen:
            raise PipelineError("O snapshot contém identificadores de trecho repetidos.")
        seen.add(identity)
        try:
            if any(isinstance(segment.get(key), bool) or not isinstance(segment.get(key), (int, float)) for key in ("start", "end")):
                raise ValueError()
            start, end = float(segment["start"]), float(segment["end"])
        except (ValueError, TypeError, KeyError) as exc:
            raise PipelineError("Os tempos do snapshot são inválidos.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < previous or start < 0 or end <= start or end > duration + 1.0:
            raise PipelineError("Os tempos do snapshot não correspondem aos limites da mídia.")
        previous = start
        if not isinstance(segment.get("speaker"), str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", segment["speaker"]):
            raise PipelineError("O snapshot possui locutor inválido.")
        if not isinstance(segment.get("text"), str) or not segment["text"].strip():
            raise PipelineError("O snapshot possui texto original vazio.")
    translations = validate_review(review, manifest, source_segments)
    for items in translations.values():
        for original, revised in zip(source_segments, items):
            if any(isinstance(revised.get(key), bool) or not isinstance(revised.get(key), (int, float)) for key in ("start", "end")) or any(revised.get(key) != original.get(key) for key in ("id", "start", "end", "speaker", "text")):
                raise PipelineError("A revisão alterou a transcrição congelada; somente a tradução pode ser editada.")
    return {"source": snapshot, "translations": translations, "source_sha256": contract["sha256"]}


def run_command(command: list[str], log: Path | None = None) -> None:
    if log:
        public_error = None
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\n$ " + " ".join(command) + "\n")
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1, env={**os.environ, "PYTHONIOENCODING": "utf-8"}) as process:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                    if line.startswith("ERRO_ETAPA="):
                        public_error = safe_error_text(line[len("ERRO_ETAPA="):])
                exit_code = process.wait()
            if exit_code:
                reason = public_error or f"A etapa terminou com código {exit_code}; confira o log preservado."
                diagnostic = {"reason": reason, "exit_code": exit_code, "log_file": log.name}
                if "--stage" in command:
                    position = command.index("--stage") + 1
                    if position < len(command):
                        diagnostic["stage"] = safe_error_text(command[position], 80)
                raise PipelineError(reason, diagnostic=diagnostic)
    else:
        subprocess.run(command, check=True)


def media_info(file: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,format_name:stream=index,codec_type,duration,width,height,channels,sample_rate,avg_frame_rate,r_frame_rate,nb_frames", "-of", "json", str(file)], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    payload["duration"] = float(payload.get("format", {}).get("duration", 0))
    payload["container_duration"] = payload["duration"]
    # Private probe context, not proof from a result report. Missing stream
    # durations can be measured against this exact file, never its container.
    payload["_media_path"] = str(file)
    kind = "video" if any(item.get("codec_type") == "video" for item in payload.get("streams", [])) else "audio"
    payload["duration"] = input_stream_duration(payload, kind)
    selected = next(item for item in payload["streams"] if item.get("codec_type") == kind)
    selected["duration"] = payload["duration"]
    return payload


def verify_final_media(video: Path, expected_duration: float, *, audio_master: bool = False) -> dict:
    try:
        expected_duration = float(expected_duration)
    except (TypeError, ValueError):
        raise PipelineError("A duração esperada do resultado não é válida.") from None
    if not math.isfinite(expected_duration) or expected_duration <= 0:
        raise PipelineError("A duração esperada do resultado não é finita e positiva.")
    info = media_info(video)
    types = {stream.get("codec_type"): stream for stream in info["streams"]}
    if not {"video", "audio"}.issubset(types):
        raise PipelineError("O resultado está sem faixa de vídeo ou áudio.")
    try:
        durations = {kind: input_stream_duration(info, kind) for kind in ("video", "audio")}
        container_duration = float(info.get("container_duration", info.get("duration", 0)))
    except PipelineError as exc:
        raise PipelineError(f"Resultado incompleto: duração de vídeo/áudio inválida. {exc}") from exc
    except (TypeError, ValueError):
        raise PipelineError("O resultado possui duração de vídeo/áudio inválida.") from None
    if not math.isfinite(container_duration) or container_duration <= 0:
        raise PipelineError("O contêiner do resultado não possui duração finita e positiva.")
    tolerance = 1.0
    if audio_master:
        from fractions import Fraction
        try:
            fps = float(Fraction(types["video"].get("avg_frame_rate") or types["video"].get("r_frame_rate")))
            rate = int(types["audio"]["sample_rate"])
            if not math.isfinite(fps) or fps <= 0 or rate <= 0:
                raise ValueError()
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            raise PipelineError("O resultado não informa relógios de quadro/áudio válidos.") from None
        tolerance = 1 / fps + 1024 / rate + 0.002
    if any(not math.isfinite(duration) or duration <= 0 or abs(duration - expected_duration) > tolerance for duration in durations.values()):
        raise PipelineError(f"Resultado incompleto: vídeo/áudio {durations}, esperado {expected_duration:.3f}s.")
    return {"duration": expected_duration if audio_master else container_duration, "container_duration": container_duration, "video_duration": durations["video"], "audio_duration": durations["audio"], "expected_duration": expected_duration, "allowed_timing_tolerance_seconds": tolerance, "timing_tolerance_seconds": max(abs(value - expected_duration) for value in durations.values()), "audio_channels": int(types["audio"].get("channels", 0))}


def _stream_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stream_timeline_duration(path: Path, kind: str, sample_rate: object = None) -> float:
    """Measure v:0/a:0 from actual frame timestamps and positive durations.

    Matroska/WebM commonly omit stream.duration. The last image's own frame
    or matching packet duration is required: audio padding, average FPS and
    differences between timestamps cannot prove how long that image lasts.
    """
    if kind not in {"video", "audio"}:
        raise PipelineError("A faixa solicitada não é de vídeo ou áudio.")
    selector = "v:0" if kind == "video" else "a:0"

    def probe(entries: str, collection: str) -> list[dict]:
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", selector,
                f"-show_{collection}", "-show_entries", entries, "-of", "json", str(path),
            ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
            payload = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            raise PipelineError(f"Não foi possível medir os quadros reais da faixa de {kind}.") from exc
        items = payload.get(collection) if isinstance(payload, dict) else None
        if result.returncode or not isinstance(items, list) or not items:
            raise PipelineError(f"A faixa de {kind} não possui quadros reais mensuráveis.")
        return items

    frames = probe("frame=best_effort_timestamp_time,pts_time,duration_time,pkt_duration_time,nb_samples,sample_rate", "frames")
    rate = _stream_number(sample_rate)
    intervals = []
    missing_duration = []
    for frame in frames:
        if not isinstance(frame, dict):
            raise PipelineError(f"A faixa de {kind} possui um quadro inválido.")
        timestamp = _stream_number(frame.get("best_effort_timestamp_time", frame.get("pts_time")))
        if timestamp is None:
            raise PipelineError(f"A faixa de {kind} possui quadro sem timestamp finito.")
        lengths = [_stream_number(frame.get(key)) for key in ("duration_time", "pkt_duration_time")]
        length = next((value for value in lengths if value is not None and value > 0), None)
        if length is None and kind == "audio":
            samples = _stream_number(frame.get("nb_samples"))
            frame_rate = _stream_number(frame.get("sample_rate")) or rate
            if samples is not None and samples > 0 and frame_rate is not None and frame_rate > 0:
                length = samples / frame_rate
        intervals.append([timestamp, length])
        if length is None:
            missing_duration.append(len(intervals) - 1)
    if missing_duration:
        packets = probe("packet=pts_time,duration_time", "packets")
        packet_lengths = {}
        for packet in packets:
            if not isinstance(packet, dict):
                continue
            timestamp = _stream_number(packet.get("pts_time"))
            length = _stream_number(packet.get("duration_time"))
            if timestamp is not None and length is not None and length > 0:
                packet_lengths.setdefault(round(timestamp, 6), []).append(length)
        for index in missing_duration:
            timestamp = intervals[index][0]
            candidates = packet_lengths.get(round(timestamp, 6), [])
            if not candidates or max(candidates) - min(candidates) > 0.000002:
                raise PipelineError(f"A faixa de {kind} possui quadro sem duração positiva verificável.")
            intervals[index][1] = candidates[0]
    start = min(item[0] for item in intervals)
    end = max(timestamp + length for timestamp, length in intervals)
    duration = end - start
    if not math.isfinite(duration) or duration <= 0:
        raise PipelineError(f"A duração real da faixa de {kind} não é finita e positiva.")
    return duration


def input_stream_duration(info: dict, kind: str) -> float:
    stream = next((item for item in info.get("streams", []) if item.get("codec_type") == kind), None)
    if stream is None:
        raise PipelineError(f"A mídia não contém uma faixa de {kind} mensurável.")
    value = stream.get("duration")
    duration = _stream_number(value)
    formats = str(info.get("format", {}).get("format_name", "")).split(",")
    if duration is not None and duration > 0:
        # Matroska demuxers may synthesize a video stream.duration from the
        # overall Segment duration (including much longer audio). Trust real
        # frames for these actual formats even when that field looks valid.
        if kind == "video" and {"matroska", "webm"}.intersection(formats):
            if not info.get("_media_path"):
                raise PipelineError("A faixa de vídeo Matroska/WebM exige medição dos quadros reais.")
            return stream_timeline_duration(Path(info["_media_path"]), kind)
        return duration
    # Only absent/N/A metadata may use a fresh frame probe. NaN, infinity and
    # nonpositive declared durations are corruption, not a container fallback.
    if value not in (None, "N/A", "") or not info.get("_media_path"):
        raise PipelineError(f"A duração da faixa de {kind} não é válida.")
    return stream_timeline_duration(Path(info["_media_path"]), kind, stream.get("sample_rate"))


def validate_uploaded_audio_duration(audio_duration: float, video_duration: float) -> None:
    if not math.isfinite(audio_duration) or audio_duration <= 0 or audio_duration > video_duration + 0.000001:
        raise PipelineError("O áudio enviado excede a duração da faixa de vídeo e seria cortado. Ajuste os arquivos antes de continuar; nenhuma fala será descartada automaticamente.")


def validate_lipsync_timeline(report: dict, source_duration: float) -> None:
    """Reject incomplete/out-of-source evidence before exporting a completed result."""
    proof = report.get("lipsync", {})
    if proof.get("required") is not True:
        return
    if proof.get("ambiguity") != []:
        raise PipelineError("Lip-sync com ambiguidade: corrija a seleção do rosto antes de exportar.")
    confirmed = proof.get("confirmed_segments")
    preserved = report.get("preserved_no_speaker_visual_segments")
    if not isinstance(confirmed, list) or not isinstance(preserved, list):
        raise PipelineError("A evidência temporal do lip-sync está incompleta.")
    faceless = (report.get("operation") == "dub_video" and
                proof.get("no_visible_speaking_face") is True and
                proof.get("scene_analysis_complete") is True and
                proof.get("confirmed") is False and proof.get("applied") is False)
    if (not confirmed and not faceless) or (confirmed and
            (proof.get("confirmed") is not True or proof.get("applied") is not True)):
        raise PipelineError("O lip-sync solicitado não possui trechos realmente confirmados.")
    windows = []
    for intervals in (confirmed, preserved):
        previous_end = 0.0
        for interval in intervals:
            if not isinstance(interval, dict):
                raise PipelineError("A evidência temporal do lip-sync está malformada.")
            start, end = interval.get("start"), interval.get("end")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
                   not math.isfinite(value) for value in (start, end)):
                raise PipelineError("O lip-sync possui tempo inválido no relatório.")
            if start < 0 or end <= start or end > source_duration + 0.05 or start < previous_end - 0.001:
                raise PipelineError("O lip-sync possui trecho fora da duração real do vídeo ou sobreposto.")
            previous_end = end
            windows.append((start, end))
    cursor = 0.0
    for start, end in sorted(windows):
        if start < cursor - 0.001 or start > cursor + 0.05:
            raise PipelineError("Os trechos sincronizados e preservados não cobrem o vídeo sem lacunas ou sobreposições.")
        cursor = end
    if abs(cursor - source_duration) > 0.05:
        raise PipelineError("A evidência do lip-sync não cobre o vídeo até o fim.")


class StageCache:
    def __init__(self, root: Path, storage: Path):
        self.root, self.storage = root, storage
        self.seconds: dict[str, float] = {}
        self.used: dict[str, bool] = {}
        self.persisted: dict[str, tuple[int, int]] = {}
        self.gpu_activity: dict[str, dict] = {}
        self.worker_code = worker_fingerprint()

    def persist(self) -> None:
        source_root, storage_root = self.root.resolve(), self.storage.resolve()
        if source_root == storage_root:
            return  # No Drive: the local job is already its own checkpoint.
        if source_root.is_relative_to(storage_root) or storage_root.is_relative_to(source_root):
            raise PipelineError("Origem e destino dos checkpoints não podem ficar um dentro do outro.")
        # Checkpoint files may change without changing size (edited translations).
        # Track local stat changes instead of rehashing GBs of video after every stage.
        for file in self.root.rglob("*"):
            if not file.is_file() or file.name.endswith((".writing", ".copying")):
                continue
            relative = file.relative_to(self.root).as_posix()
            stat = file.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            target = self.storage / relative
            if self.persisted.get(relative) == signature and target.exists() and target.stat().st_size == stat.st_size:
                continue
            if not target.exists() or target.stat().st_size != stat.st_size or sha256(file) != sha256(target):
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".copying")
                shutil.copy2(file, temporary)
                temporary.replace(target)
            self.persisted[relative] = signature

    def stage(self, key: str, label: str, config: dict, required: list[Path]) -> None:
        signature = fingerprint({"pipeline": PIPELINE_VERSION, "worker": self.worker_code, "config": config})
        marker = self.root / "checkpoints" / f"{key}.json"
        cached = False
        if marker.exists():
            try:
                record = read_json(marker)
            except (ValueError, OSError):
                record = {}  # Interrupted marker writes require rebuilding this stage.
            # Valid JSON can still be an interrupted/incompatible marker.
            # Treat malformed schemas as a cache miss, never completion, and
            # leave the previous marker untouched until rebuilding succeeds.
            if not isinstance(record, dict) or not isinstance(record.get("checksums"), dict):
                record = {}
            checksums = record.get("checksums", {})
            cached = record.get("signature") == signature and all(file.is_file() and sha256(file) == checksums.get(file.relative_to(self.root).as_posix()) for file in required)
        self.used[key] = cached
        if cached:
            print(f"OK {label} — checkpoint íntegro reutilizado", flush=True)
            self.seconds[key] = 0.0
            return
        started = time.monotonic()
        print(f"INICIANDO {label}", flush=True)
        request = self.root / "requests" / f"{key}.json"
        write_json(request, config)
        samples = []
        stop_sampling = threading.Event()
        def sample_gpu():
            while not stop_sampling.is_set():
                try:
                    state = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                    if state.returncode == 0:
                        values = state.stdout.strip().splitlines()[0].split(",")
                        samples.append((int(values[0].strip()), int(values[1].strip())))
                except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                    pass
                stop_sampling.wait(2)
        observer = threading.Thread(target=sample_gpu, daemon=True)
        observer.start()
        try:
            run_command([sys.executable, "-u", str(Path(__file__).resolve()), "--stage", config["stage"], "--stage-config", str(request)], self.root / "pipeline.log")
            if not all(file.is_file() and file.stat().st_size > 0 for file in required):
                raise PipelineError(f"{label} não produziu todos os arquivos esperados.")
            write_json(marker, {"signature": signature, "checksums": {file.relative_to(self.root).as_posix(): sha256(file) for file in required}, "finished_at": int(time.time())})
            self.seconds[key] = round(time.monotonic() - started, 3)
            print(f"OK {label} — {self.seconds[key]:.1f}s", flush=True)
        except Exception as exc:
            raise stage_failure(exc, key, label, config) from exc
        finally:
            stop_sampling.set()
            observer.join(timeout=5)
            self.gpu_activity[key] = {"sample_count": len(samples), "utilization_max_percent": max((sample[0] for sample in samples), default=None), "memory_peak_mib": max((sample[1] for sample in samples), default=None)}
            self.persist()  # Preserve diagnostics/checkpoints even on disconnect/error.
            runtime = Path(os.environ.get("DUBSTUDIO_RUNTIME_ROOT", "/content/dubstudio"))
            storage = Path(os.environ.get("DUBSTUDIO_STORAGE_ROOT", str(runtime)))
            if runtime.exists():
                from dubstudio_bootstrap import persist_cache
                persist_cache(runtime, storage)


def duration_slot(segments: list[dict], index: int, total_duration: float) -> float:
    item = segments[index]
    next_start = float(segments[index + 1]["start"]) if index + 1 < len(segments) else total_duration
    available = min(total_duration, max(float(item["end"]), next_start)) - float(item["start"])
    if available <= 0:
        raise PipelineError("Os segmentos possuem tempos incompatíveis com a duração do vídeo.")
    return available


def tempo_filters(ratio: float) -> str:
    if ratio <= 0 or not math.isfinite(ratio):
        raise PipelineError("Fator de tempo inválido.")
    factors = []
    while ratio > 2:
        factors.append(2.0)
        ratio /= 2
    while ratio < 0.5:
        factors.append(0.5)
        ratio /= 0.5
    factors.append(ratio)
    return ",".join(f"atempo={value:.8f}" for value in factors)


def fit_voice_to_frames(raw: Path, adjusted: Path, ratio: float, available_frames: int, sample_rate: int = 48000) -> float:
    """Re-encode the entire utterance; never trim an overrun to fit a window."""
    if available_frames <= 0:
        raise PipelineError("Não há espaço na linha do tempo para a fala completa.")
    initial_ratio = ratio
    for attempt in range(3):
        run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
                     "-af", tempo_filters(ratio), "-ac", "2", "-ar", str(sample_rate),
                     "-c:a", "pcm_s16le", str(adjusted)])
        with wave.open(str(adjusted), "rb") as rendered:
            if rendered.getframerate() != sample_rate:
                raise PipelineError("A fala renderizada está em uma taxa de áudio inesperada.")
            frames = rendered.getnframes()
        if 0 < frames <= available_frames:
            return ratio
        if frames <= 0:
            raise PipelineError("A fala ajustada está vazia.")
        # FFmpeg atempo may round a few samples beyond its requested duration.
        # Retry from the untouched full utterance, with a bounded correction.
        ratio *= frames / available_frames * 1.002
        if ratio > initial_ratio * 1.05:
            break
    raise PipelineError("A fala completa ainda excede a janela; exportação recusada para não cortar seu final.")


def tts_segments_for_frames(segments: list[dict], frame_counts: list[int], sample_rate: int = 48000, pause_frames: int = 8640) -> list[dict]:
    """Derive TTS timestamps from rendered samples, including inter-phrase pauses."""
    if not segments or len(segments) != len(frame_counts) or sample_rate <= 0 or pause_frames < 0:
        raise PipelineError("A linha do tempo da voz gerada está incompleta.")
    rendered, cursor = [], 0
    for index, (segment, frames) in enumerate(zip(segments, frame_counts)):
        if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
            raise PipelineError("Uma fala gerada não possui amostras válidas.")
        if index:
            cursor += pause_frames
        emitted = copy.deepcopy(segment)
        emitted.update(start=cursor / sample_rate, end=(cursor + frames) / sample_rate)
        rendered.append(emitted)
        cursor += frames
    return rendered


def validate_voice_reference(file: Path) -> dict:
    """Reject silence/music-only prompts before downloading the synthesis model.

    Silero VAD is bundled with Faster-Whisper; this small CPU check does not load
    an ASR or synthesis model and never claims perceptual timbre verification.
    """
    decoded = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(file), "-t", "120", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"], check=True, capture_output=True)
    samples = array("h", decoded.stdout)
    if sys.byteorder != "little":
        samples.byteswap()
    seconds = len(samples) / 16000.0
    rms = math.sqrt(sum((sample / 32768.0) ** 2 for sample in samples) / max(1, len(samples)))
    if seconds < 0.8 or rms < 0.0001:
        raise PipelineError("Referência sem fala audível: envie uma amostra autorizada com voz clara, sem silêncio.")
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    speech = get_speech_timestamps(np.asarray(samples, dtype="float32") / 32768.0, vad_options=VadOptions(min_speech_duration_ms=150, min_silence_duration_ms=200, speech_pad_ms=0), sampling_rate=16000)
    speech_seconds = sum(max(0, item["end"] - item["start"]) for item in speech) / 16000.0
    if speech_seconds < 0.6:
        raise PipelineError("Referência sem fala confirmada: envie uma amostra de voz clara em vez de silêncio, música ou ruído.")
    return {"speech_seconds": round(speech_seconds, 3), "checked_seconds": round(seconds, 3), "rms_dbfs": round(20 * math.log10(max(rms, 1e-12)), 2), "method": "Silero VAD e energia; timbre exige revisão humana"}


def synthesis_preflight(segments: list[dict], references: dict[str, Path], duration: float) -> None:
    """Require revision, never silently discard dubious micro-speech before GPU load."""
    if not math.isfinite(duration) or duration < 0 or not isinstance(segments, list) or not segments:
        raise PipelineError("A linha do tempo da síntese está incompleta ou inválida.")
    if not references or any(not file.is_file() or file.stat().st_size == 0 for file in references.values()):
        raise PipelineError("As referências autorizadas de voz não estão disponíveis.")
    def words(item, field):
        return tuple(re.findall(r"[^\W_]+", str(item.get(field, "")).casefold(), re.UNICODE))
    for index, item in enumerate(segments):
        if not isinstance(item, dict) or not isinstance(item.get("translated_text"), str) or not words(item, "translated_text"):
            raise PipelineError(f"O segmento {index + 1} não possui texto falado. Revise a transcrição/tradução; nenhuma fala foi removida.")
        speaker = item.get("speaker", "SPEAKER_00")
        if speaker not in references:
            raise PipelineError(f"Locutor {speaker} sem referência vocal autorizada.")
        if duration == 0:  # TTS has no fixed video window; timestamps follow generated samples.
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PipelineError(f"O segmento {index + 1} possui tempos inválidos; revise a transcrição.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration + 0.001:
            raise PipelineError(f"O segmento {index + 1} não possui janela de fala válida; revise a transcrição, sem descartar fala automaticamente.")
        if end - start > 0.2 + 1e-9:
            continue
        neighbors = segments[max(0, index - 1):index] + segments[index + 1:index + 2]
        def nearby(other):
            try:
                other_start, other_end = float(other["start"]), float(other["end"])
            except (KeyError, TypeError, ValueError):
                return False
            return math.isfinite(other_start) and math.isfinite(other_end) and max(0.0, max(start, other_start) - min(end, other_end)) <= 0.2 + 1e-9
        duplicated = any(isinstance(other, dict) and nearby(other) and other.get("speaker", "SPEAKER_00") == speaker and any(
            words(item, field) and words(item, field) == words(other, field) for field in ("text", "translated_text")
        ) for other in neighbors)
        if duplicated:
            raise PipelineError(f"O segmento {item.get('id', index + 1)} repete uma fala adjacente em apenas {end - start:.3f}s. Revise a transcrição e seus tempos antes da síntese; nenhuma fala foi removida.")


def chatterbox_asset_names(module) -> list[str]:
    """Read the installed loader's explicit download contract, not guessed filenames."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(module.ChatterboxMultilingualTTS.from_pretrained)))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "snapshot_download"]
    if len(calls) != 1:
        raise PipelineError("O loader vocal instalado não possui contrato de pesos compatível com o cache seguro.")
    pattern_node = next((kw.value for kw in calls[0].keywords if kw.arg == "allow_patterns"), None)
    if not isinstance(pattern_node, (ast.List, ast.Tuple)):
        raise PipelineError("O loader vocal não informa uma lista verificável de pesos e configurações.")
    names = []
    for node in pattern_node.elts:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            name = node.value
        elif isinstance(node, ast.Name) and node.id == "t3_model" and callable(getattr(module, "_resolve_multilingual_t3_model", None)):
            name = module._resolve_multilingual_t3_model(None)
        else:
            raise PipelineError("O contrato de pesos vocais mudou; o cache não será usado sem conferência do loader.")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in {".", ".."}:
            raise PipelineError("Nome de peso/configuração vocal inválido.")
        names.append(name)
    if len(names) != len(set(names)) or not {"ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"}.issubset(names) or sum(name.startswith("t3") and name.endswith(".safetensors") for name in names) != 1:
        raise PipelineError("O contrato de pesos vocais está incompleto.")
    return sorted(names)


def chatterbox_model_fingerprint(module, snapshot: Path, assets: list[str]) -> dict:
    """Fingerprint actual immutable assets plus installed model/config code and defaults."""
    version = importlib.metadata.version("chatterbox-tts")
    if version != "0.1.7":
        raise PipelineError("A versão vocal instalada não é a 0.1.7 validada pelo pipeline.")
    if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[a-fA-F0-9]{40}", snapshot.name):
        raise PipelineError("Os pesos vocais não têm revisão imutável confirmada; cache RAW recusado.")
    mandatory = {"ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"} | {name for name in assets if name.startswith("t3") and name.endswith(".safetensors")}
    if "Cangjie5_TC.json" in assets:
        mandatory.add("Cangjie5_TC.json")
    hashes = {}
    for name in assets:
        file = snapshot / name
        if not file.is_file() or file.stat().st_size == 0:
            if name in mandatory:
                raise PipelineError(f"Peso/configuração vocal obrigatório ausente ou vazio: {name}.")
            hashes[name] = None  # Absence of optional conditionals/config is part of provenance.
        else:
            hashes[name] = sha256(file)
    module_file = Path(inspect.getfile(module))
    source_root = module_file.parent
    sources = {str(file.relative_to(source_root)).replace("\\", "/"): sha256(file) for file in sorted(source_root.rglob("*.py")) if file.is_file()}
    if module_file.name not in sources:
        raise PipelineError("O código/configuração do modelo vocal não pôde ser conferido.")
    defaults = {}
    for name, parameter in inspect.signature(module.ChatterboxMultilingualTTS.generate).parameters.items():
        if name in {"self", "text", "language_id", "audio_prompt_path"}:
            continue
        default = parameter.default
        if default is inspect.Parameter.empty or not (default is None or type(default) in {str, int, float, bool}) or (isinstance(default, float) and not math.isfinite(default)):
            raise PipelineError("Os parâmetros de síntese vocal mudaram e precisam de conferência.")
        defaults[name] = default
    dependencies = {name: importlib.metadata.version(name) for name in ("torch", "torchaudio", "numpy", "librosa", "safetensors", "s3tokenizer", "resemble-perth", "tokenizers", "pykakasi", "spacy-pkuseg")}
    return {"engine": "Chatterbox Multilingual", "version": version, "repo": module.REPO_ID, "revision": snapshot.name.lower(), "assets": hashes,
            "installed_sources": sources, "dependencies": dependencies, "generation_defaults": defaults, "device": "cuda", "wav_encoding": "PCM_16"}


def prepare_chatterbox_model() -> tuple[dict, object]:
    """Resolve and hash disk weights without loading a model; lazy-load that exact snapshot."""
    import chatterbox.mtl_tts as module
    from huggingface_hub import snapshot_download
    assets = chatterbox_asset_names(module)
    if module.REPO_ID != "ResembleAI/chatterbox" or not callable(getattr(module.ChatterboxMultilingualTTS, "from_local", None)):
        raise PipelineError("O modelo vocal instalado não possui origem/loader compatível com a validação.")
    kwargs = {"repo_id": module.REPO_ID, "repo_type": "model", "allow_patterns": assets, "token": os.getenv("HF_TOKEN")}
    try:
        snapshot = Path(snapshot_download(**kwargs, local_files_only=True))
    except Exception:
        snapshot = Path(snapshot_download(**kwargs, revision="main"))
    # A warm cache may have a partial snapshot. Download only missing mandatory assets
    # from its same immutable revision, never silently replace a pinned set with main.
    mandatory = {"ve.pt", "s3gen.pt", "grapheme_mtl_merged_expanded_v1.json"} | {name for name in assets if name.startswith("t3") and name.endswith(".safetensors")}
    if "Cangjie5_TC.json" in assets:
        mandatory.add("Cangjie5_TC.json")
    if any(not (snapshot / name).is_file() or (snapshot / name).stat().st_size == 0 for name in mandatory):
        if not re.fullmatch(r"[a-fA-F0-9]{40}", snapshot.name):
            raise PipelineError("Revisão vocal inválida; download e síntese interrompidos.")
        snapshot = Path(snapshot_download(**kwargs, revision=snapshot.name))
    fingerprint = chatterbox_model_fingerprint(module, snapshot, assets)
    def load():
        if chatterbox_model_fingerprint(module, snapshot, assets) != fingerprint:
            raise PipelineError("Pesos/configurações vocais mudaram após a conferência; síntese interrompida.")
        tokenizer_module = inspect.getmodule(module.MTLTokenizer)
        download = getattr(tokenizer_module, "hf_hub_download", None)
        unexpected_download = []
        def pinned_tokenizer_download(*args, **kwargs):
            repo = kwargs.get("repo_id", args[0] if args else None)
            filename = kwargs.get("filename", args[1] if len(args) > 1 else None)
            if repo != module.REPO_ID or filename != "Cangjie5_TC.json" or filename not in mandatory:
                unexpected_download.append(True)
                raise PipelineError("O tokenizer solicitou uma configuração não vinculada aos pesos conferidos.")
            return str(snapshot / filename)
        # 0.1.7 otherwise re-downloads Cangjie from main in a nested cache even
        # when the main model was loaded from a pinned local snapshot.
        if callable(download):
            tokenizer_module.hf_hub_download = pinned_tokenizer_download
        try:
            model = module.ChatterboxMultilingualTTS.from_local(snapshot, device="cuda")
        finally:
            if callable(download):
                tokenizer_module.hf_hub_download = download
        if unexpected_download:
            del model
            raise PipelineError("Uma configuração vocal externa não conferida impediu a síntese segura.")
        if chatterbox_model_fingerprint(module, snapshot, assets) != fingerprint:
            del model
            raise PipelineError("Pesos/configurações vocais mudaram durante o carregamento; síntese interrompida.")
        return model
    return fingerprint, load


def raw_voice_identity(item: dict, language: str, reference: Path, model_fingerprint: dict) -> tuple[str, dict]:
    identity = {"revision": RAW_SYNTHESIS_CACHE_REVISION, "language": language, "speaker": item.get("speaker", "SPEAKER_00"),
                "text": item["translated_text"], "reference_sha256": sha256(reference), "model": model_fingerprint}
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), identity


def raw_voice_metadata(file: Path) -> dict:
    """Check the entire PCM WAV independently, including truncation and audible energy."""
    try:
        with wave.open(str(file), "rb") as audio:
            channels, rate, width, frames = audio.getnchannels(), audio.getframerate(), audio.getsampwidth(), audio.getnframes()
            if channels not in {1, 2} or width != 2 or not 8000 <= rate <= 192000 or not frames > 0 or audio.getcomptype() != "NONE":
                raise PipelineError("O áudio RAW vocal possui formato/amostras inválidos.")
            count, energy = 0, 0
            while chunk := audio.readframes(65536):
                samples = array("h", chunk)
                if sys.byteorder != "little":
                    samples.byteswap()
                count += len(samples)
                energy += sum(sample * sample for sample in samples)
            if count != frames * channels or math.sqrt(energy / count) / 32768.0 < 1e-6:
                raise PipelineError("O áudio RAW vocal está truncado ou silencioso.")
    except (OSError, wave.Error, EOFError, ValueError) as exc:
        raise PipelineError("O áudio RAW vocal não é um WAV íntegro.") from exc
    return {"frames": frames, "sample_rate": rate, "channels": channels, "duration": frames / rate, "sha256": sha256(file)}


class RawVoiceCache:
    """An ordinal WAV without complete provenance is never an approved cached utterance."""
    def __init__(self, directory: Path):
        self.directory = directory / "voice-raw-cache"

    def get_or_generate(self, identity_key: str, identity: dict, generate, *, accept=None, attempts: int = 1) -> tuple[Path, dict, bool]:
        if not re.fullmatch(r"[a-f0-9]{64}", identity_key):
            raise PipelineError("Identificador de cache vocal inválido.")
        if type(attempts) is not int or not 1 <= attempts <= 2:
            raise PipelineError("A síntese permite no máximo duas tentativas completas por trecho.")
        raw, marker = self.directory / f"{identity_key}.wav", self.directory / f"{identity_key}.json"
        cached_metadata = None
        try:
            record = read_json(marker)
            if isinstance(record, dict) and record.get("identity") == identity and record.get("identity_key") == identity_key:
                metadata = raw_voice_metadata(raw)
                if record.get("audio") == metadata:
                    cached_metadata = metadata
        except (OSError, ValueError, PipelineError):
            pass
        if cached_metadata is not None and (accept is None or accept(raw, cached_metadata)):
            return raw, cached_metadata, True
        self.directory.mkdir(parents=True, exist_ok=True)
        pending = raw.with_suffix(".pending.wav")
        for attempt in range(1, attempts + 1):
            generate(pending)
            metadata = raw_voice_metadata(pending)
            if accept is not None and not accept(pending, metadata):
                rejected = self.directory / f"{identity_key}.attempt-{attempt}.rejected.wav"
                if rejected.exists():
                    rejected = self.directory / f"{identity_key}.attempt-{attempt}-{metadata['sha256']}.rejected.wav"
                os.replace(pending, rejected)
                print(f"Geração vocal {attempt}/{attempts} recusada para a janela; candidato preservado sem checkpoint de sucesso.", flush=True)
                continue
            # Decode and hash the complete candidate again before committing it.
            # This pipeline does not trim speech or outside silence automatically.
            metadata = raw_voice_metadata(pending)
            if raw.exists():
                previous = self.directory / f"{identity_key}.{sha256(raw)}.previous.wav"
                if not previous.exists():
                    shutil.copy2(raw, previous)
            os.replace(pending, raw)
            write_json(marker, {"identity_key": identity_key, "identity": identity, "audio": metadata})
            return raw, metadata, False
        raise PipelineError("A fala completa permanece longa demais após duas tentativas. Revise/encurte o texto ou corrija os tempos; nenhuma fala foi cortada e os checkpoints anteriores foram preservados.")


def synthesize_stage(config: dict, legacy) -> None:
    directory = Path(config["directory"])
    language = config["language"].split("-")[0]
    language = "no" if language == "nb" else language
    if language not in CLONE_LANGUAGES:
        raise PipelineError("Idioma sem clonagem de timbre. Não há substituição por voz genérica.")
    payload = read_json(Path(config["segments"]))
    segments = payload["segments"]
    audio_master = config.get("timing_mode") == "audio_master" and float(config.get("duration", 0)) > 0
    if audio_master:
        # Stable source-turn order, including A -> B -> A and overlaps. Never
        # mutate the approved translation or group all speech by one identity.
        segments = sorted(copy.deepcopy(segments), key=lambda item: float(item["start"]))
    references = {key: Path(value) for key, value in config["references"].items()}
    duration = float(config.get("duration", 0))
    synthesis_preflight(segments, references, duration)
    import numpy as np
    import soundfile as sf
    import torch
    from pydub import AudioSegment
    reference_validation = {speaker: validate_voice_reference(file) for speaker, file in references.items()}
    model_fingerprint, load_model = prepare_chatterbox_model()
    model = None
    raw_cache = RawVoiceCache(directory)
    timeline = AudioSegment.silent(duration=math.ceil(duration * 1000), frame_rate=48000).set_channels(2) if duration > 0 and not audio_master else None
    generated = []
    adjustments = []
    raw_proof = []
    try:
        for index, item in enumerate(segments):
            speaker = item.get("speaker", "SPEAKER_00")
            if speaker not in references:
                raise PipelineError(f"Locutor {speaker} sem referência vocal autorizada.")
            print(f"Voz {config['language']}: segmento {index + 1}/{len(segments)}", flush=True)
            identity_key, identity = raw_voice_identity(item, language, references[speaker], model_fingerprint)
            fitted = {}
            if timeline is not None:
                slot = duration_slot(segments, index, duration)
                speaking_time = max(0.15, float(item["end"]) - float(item["start"]))
                position_ms = round(float(item["start"]) * 1000)
                position_frames = position_ms * 48
                available_frames = min(int(slot * 48000), int(timeline.frame_count()) - position_frames)
                budget = min(slot, speaking_time, available_frames / 48000)
                if not math.isfinite(budget) or budget <= 0:
                    raise PipelineError(f"Não há janela de fala para o segmento {item['id']}; corrija os tempos antes da síntese.")
                adjusted = directory / f"voice-fitted-{index:04d}.wav"
                adjusted_pending = directory / f"voice-fitted-{index:04d}.pending.wav"
            def accept_candidate(candidate, metadata):
                ratio = max(1.0, metadata["duration"] / budget)
                if ratio > MAX_VOICE_TEMPO_FACTOR:
                    return False
                try:
                    ratio = fit_voice_to_frames(candidate, adjusted_pending, ratio, available_frames)
                except PipelineError as exc:
                    if "ainda excede a janela" in str(exc):
                        return False
                    raise
                clip = AudioSegment.from_wav(adjusted_pending)
                if not math.isfinite(ratio) or ratio > MAX_VOICE_TEMPO_FACTOR or int(clip.frame_count()) > available_frames:
                    return False
                fitted.update(ratio=ratio, clip=clip)
                return True
            def generate_raw(pending):
                nonlocal model
                if sha256(references[speaker]) != identity["reference_sha256"]:
                    raise PipelineError("A referência vocal mudou durante a síntese; cache recusado.")
                if model is None:
                    model = load_model()
                utterance = model.generate(item["translated_text"], language_id=language, audio_prompt_path=str(references[speaker]))
                samples = utterance.detach().float().cpu().numpy().squeeze()
                if samples.size == 0 or not np.isfinite(samples).all() or float(np.sqrt(np.mean(np.square(samples)))) < 1e-6:
                    raise PipelineError(f"Síntese silenciosa/inválida no segmento {item['id']}.")
                sf.write(pending, samples, model.sr, subtype="PCM_16")
                if sha256(references[speaker]) != identity["reference_sha256"]:
                    raise PipelineError("A referência vocal mudou durante a síntese; cache recusado.")
            raw, metadata, reused = raw_cache.get_or_generate(identity_key, identity, generate_raw, accept=accept_candidate if timeline is not None else None, attempts=2 if timeline is not None else 1)
            if sha256(references[speaker]) != identity["reference_sha256"]:
                raise PipelineError("A referência vocal mudou durante a conferência; síntese interrompida.")
            print(f"Voz RAW {index + 1}: {'cache íntegro reutilizado' if reused else 'gerada e conferida'}", flush=True)
            raw_proof.append({"segment_id": item["id"], "identity_key": identity_key, "raw_sha256": metadata["sha256"], "reused": reused})
            actual = metadata["duration"]
            if timeline is None:
                generated.append(AudioSegment.from_wav(raw).set_frame_rate(48000).set_channels(2))
                continue
            # The entire utterance was fitted and capped before any RAW marker.
            ratio, clip = fitted["ratio"], fitted["clip"]
            os.replace(adjusted_pending, adjusted)
            timeline = timeline.overlay(clip, position=position_ms)
            adjustments.append({"segment_id": item["id"], "factor": round(ratio, 4), "speech_seconds": actual, "rendered_seconds": len(clip) / 1000, "slot_seconds": slot})
            item["speed"] = ratio
        if timeline is None:
            if not generated:
                raise PipelineError("Nenhuma fala gerada.")
            timeline = generated[0]
            for clip in generated[1:]:
                # Dubbing keeps all RAW samples/pauses exactly; only legacy
                # standalone TTS adds its documented inter-phrase pause.
                timeline += clip if audio_master else AudioSegment.silent(duration=180, frame_rate=48000).set_channels(2) + clip
            rendered_segments = tts_segments_for_frames(segments, [int(clip.frame_count()) for clip in generated], pause_frames=0 if audio_master else 8640)
            write_json(directory / "voice-segments.json", {"target_language": config["language"], "segments": rendered_segments})
            if not audio_master:
                segments = rendered_segments
        timeline.export(directory / "dialogue.wav", format="wav")
        if audio_master:
            from dubstudio_audio_master import build_timeline
            master = build_timeline(segments, [int(clip.frame_count()) for clip in generated], sample_rate=48000)
            master.update(timing_mode="audio_master", target_language=config["language"],
                          input_sha256=config["identity"]["input_sha256"], source_video_sha256=config["source_video_sha256"],
                          audio_master={"path": "dialogue.wav", "sha256": sha256(directory / "dialogue.wav"), "sample_rate": 48000, "sample_count": int(timeline.frame_count())})
            write_json(directory / "audio-master.json", master)
        write_json(directory / "voice-proof.json", {"voice_engine": "Chatterbox Multilingual 0.1.7", "voice_clone": {"required": True, "confirmed": True, "authorized": True}, "references": {key: sha256(file) for key, file in references.items()}, "reference_validation": reference_validation, "timing_adjustments": adjustments, "raw_cache": raw_proof, "model_fingerprint_sha256": hashlib.sha256(json.dumps(model_fingerprint, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(), "segments": segments, **({"timing_mode": "audio_master", "voice_tempo_policy": "untouched-raw-samples", "overlap_policy": "serialize-source-turn-order"} if audio_master else {})})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


ASR_CONFIDENCE_THRESHOLD = 0.6
ASR_TIME_TOLERANCE = 0.002


def _asr_finite_number(value: object) -> bool:
    try:
        return type(value) in {int, float} and math.isfinite(value)
    except OverflowError:
        return False


def _asr_observed_number(value: object) -> float | str:
    # Keep invalid observations readable in diagnostics, never JSON NaN.
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    return number if not isinstance(value, bool) and math.isfinite(number) else repr(value)


def assess_asr_transcript(segments: list[dict], duration: object, language_probability: object) -> dict:
    """Confidence is a retry/review gate, never permission to discard a word."""
    issues, segment_confidence, valid_probabilities, weighted = [], [], [], []
    if not _asr_finite_number(duration) or duration <= 0:
        issues.append("duração ASR inválida")
    if not _asr_finite_number(language_probability) or not 0 <= language_probability <= 1:
        issues.append("confiança de detecção de idioma inválida")
    if not segments:
        issues.append("nenhuma fala reconhecida")
    for item in segments:
        start, end = item.get("start"), item.get("end")
        prefix = f"trecho {item.get('id')} ({start}–{end} s)"
        span_valid = _asr_finite_number(start) and _asr_finite_number(end) and 0 <= start < end
        if not span_valid or (_asr_finite_number(duration) and span_valid and end > duration + ASR_TIME_TOLERANCE):
            issues.append(f"{prefix}: tempos do bloco inválidos")
        words = item.get("words")
        if not isinstance(words, list) or not words:
            issues.append(f"{prefix}: nenhuma evidência de palavra disponível")
            continue
        previous_end, block_weighted = None, []
        for index, word in enumerate(words, 1):
            probability = word.get("probability")
            probability_valid = _asr_finite_number(probability) and 0 <= probability <= 1
            word_start, word_end = word.get("start"), word.get("end")
            word_valid = _asr_finite_number(word_start) and _asr_finite_number(word_end) and 0 <= word_start < word_end
            if not probability_valid:
                issues.append(f"{prefix}, palavra {index}: confiança inválida")
            else:
                valid_probabilities.append(probability)
            if (not word_valid or not span_valid
                    or (word_valid and span_valid and (word_start < start - ASR_TIME_TOLERANCE or word_end > end + ASR_TIME_TOLERANCE))
                    or (word_valid and previous_end is not None and word_start < previous_end - ASR_TIME_TOLERANCE)):
                issues.append(f"{prefix}, palavra {index} ({word_start}–{word_end} s): tempos de palavra inválidos")
            if word_valid:
                previous_end = word_end
            if probability_valid and word_valid:
                # Invalid spans already cause an independent blocker; zero
                # weight cannot silently forgive them in the average.
                block_weighted.append((probability, word_end - word_start))
        weight = sum(span for _, span in block_weighted)
        score = sum(probability * span for probability, span in block_weighted) / weight if weight else 0.0
        segment_confidence.append({"id": item.get("id"), "start": start, "end": end,
                                   "duration_weighted_probability": score,
                                   "positive_word_seconds": weight, "word_count": len(words)})
        weighted.extend(block_weighted)
        if not weight or score < ASR_CONFIDENCE_THRESHOLD:
            issues.append(f"{prefix}: confiança ponderada insuficiente ({score:.4f} < {ASR_CONFIDENCE_THRESHOLD})")
    weight = sum(span for _, span in weighted)
    weighted_average = sum(probability * span for probability, span in weighted) / weight if weight else 0.0
    average = sum(valid_probabilities) / len(valid_probabilities) if valid_probabilities else 0.0
    if not weight or weighted_average < ASR_CONFIDENCE_THRESHOLD:
        issues.append("confiança global ponderada insuficiente ou inválida")
    return {"uncertainties": issues, "approved": not issues, "word_probability": average,
            "duration_weighted_word_probability": weighted_average,
            "confidence_threshold": ASR_CONFIDENCE_THRESHOLD, "confidence_policy": "duration-weighted-per-segment-v1",
            "segment_confidence": segment_confidence}


def asr_stage(config: dict) -> None:
    from faster_whisper import WhisperModel
    source = config.get("source_language", "auto")
    model_name = "large-v3" if config.get("profile") in {"quality", "maximum_quality"} else "large-v3-turbo"
    output_path = Path(config["output"])
    candidate_path = output_path.with_name("asr-quality-candidate.json" if model_name == "large-v3" else "asr-fast-candidate.json")
    output, model, segments, info = [], None, None, None
    try:
        model = WhisperModel(model_name, device="cuda", compute_type="int8_float16", download_root=os.environ.get("DUBSTUDIO_WHISPER_CACHE"))
        segments, info = model.transcribe(config["vocals"], language=None if source == "auto" else source.split("-")[0], word_timestamps=True, vad_filter=True, beam_size=5)
        for item in segments:
            if not item.text.strip():
                continue
            # Do not round a tiny but positive segment into zero duration.
            output.append({"id": len(output) + 1, "start": _asr_observed_number(item.start), "end": _asr_observed_number(item.end),
                           "text": item.text.strip(), "words": [{"start": _asr_observed_number(word.start), "end": _asr_observed_number(word.end),
                           "word": word.word, "probability": _asr_observed_number(word.probability)} for word in (item.words or [])]})
            print(f"Transcrição: {_asr_observed_number(item.end)}s de {_asr_observed_number(info.duration)}s", flush=True)
    except Exception as error:
        write_json(candidate_path, {"segments": output, "model": model_name, "failure": safe_error_text(error), "approved": False})
        raise PipelineError("A transcrição ASR falhou; o candidato disponível foi preservado para revisão. Nenhuma palavra será descartada.", diagnostic={"stage": "asr"}) from error
    finally:
        # Faster-Whisper uses CTranslate2. Its iterator can also retain the
        # native model; release both before starting the quality subprocess.
        segments = model = None
        gc.collect()
    duration, probability = _asr_observed_number(info.duration), _asr_observed_number(info.language_probability)
    assessment = assess_asr_transcript(output, duration, probability)
    candidate = {"segments": output, "language": info.language, "probability": probability,
                 "model": model_name, "duration": duration, "asr_request_id": config.get("asr_request_id"), **assessment}
    write_json(candidate_path, candidate)
    issues = assessment["uncertainties"]
    if issues and model_name != "large-v3":
        # A unique unpublished destination detects exit=0 without a fresh
        # transcript. The previous approved output remains untouched.
        retry_id = uuid.uuid4().hex
        pending_path = output_path.with_name(f"asr-quality-approved-pending-{retry_id}.json")
        request = dict(config, profile="quality", output=str(pending_path), asr_request_id=retry_id)
        request_file = output_path.with_name("asr-quality-request.json")
        write_json(request_file, request)
        print("ASR incerta por trecho; retranscrevendo toda a mídia com large-v3 em processo separado.", flush=True)
        try:
            run_command([sys.executable, str(Path(__file__).resolve()), "--stage", "asr", "--stage-config", str(request_file)])
            if not pending_path.is_file():
                raise PipelineError("O processo de revisão terminou sem produzir uma nova transcrição aprovada.")
            approved = json.loads(pending_path.read_text(encoding="utf-8"))
            quality = json.loads(output_path.with_name("asr-quality-candidate.json").read_text(encoding="utf-8"))
            if (approved.get("model") != "large-v3" or quality.get("model") != "large-v3" or quality.get("approved") is not True
                    or approved.get("asr_request_id") != retry_id or quality.get("asr_request_id") != retry_id
                    or approved.get("segments") != quality.get("segments") or approved.get("language") != quality.get("language")
                    or (_asr_finite_number(duration) and _asr_finite_number(quality.get("duration")) and abs(duration - quality["duration"]) > ASR_TIME_TOLERANCE)):
                raise PipelineError("A transcrição produzida não corresponde ao candidato large-v3 aprovado desta revisão.")
            verified = assess_asr_transcript(approved.get("segments", []), quality.get("duration"), approved.get("probability"))
            if not verified["approved"]:
                raise PipelineError("O candidato de revisão ainda possui tempos ou confiança inválidos.")
            os.replace(pending_path, output_path)
        except Exception as error:
            raise PipelineError(f"A revisão ASR com large-v3 não foi aprovada. {issues[0]}. {safe_error_text(error)} Os candidatos e a transcrição anterior foram preservados; revise os trechos indicados antes de gerar vozes.", diagnostic={"stage": "asr"}) from error
        return
    if issues:
        raise PipelineError(f"Transcrição ASR ainda incerta com large-v3: {'; '.join(issues[:4])}. Revise os trechos indicados; candidatos preservados, sem descarte automático de palavras.", diagnostic={"stage": "asr"})
    write_json(output_path, {"segments": output, "language": info.language, "probability": probability,
                            "word_probability": assessment["word_probability"],
                            "duration_weighted_word_probability": assessment["duration_weighted_word_probability"],
                            "model": model_name, "asr_request_id": config.get("asr_request_id")})


def assign_diarized_speakers(segments: list[dict], mapping: dict[int, int]) -> list[dict]:
    """Never guess an unembedded turn from one neighbor or nearest start."""
    times = []
    for index, item in enumerate(segments):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as error:
            raise PipelineError(f"Diarização incerta: trecho {item.get('id', index + 1)} com tempos inválidos; revise o áudio.") from error
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise PipelineError(f"Diarização incerta: trecho {item.get('id', index + 1)} com tempos inválidos; revise o áudio.")
        times.append((start, end))
    if (not mapping or any(type(index) is not int or not 0 <= index < len(segments)
                           or type(label) is not int or not 0 <= label < 12
                           for index, label in mapping.items())):
        raise PipelineError("Diarização incerta: identidades vocais sem evidência válida; revise os locutores.")
    assignments = {}
    epsilon = 1e-7
    maximum_gap = 0.20
    for index, item in enumerate(segments):
        if index in mapping:
            assignments[index] = (mapping[index], "ecapa_clustering")
            continue
        start, end = times[index]
        reason = "sem dois turnos vizinhos confirmados"
        # Use interval ends/starts, not list order or nearest speech START.
        # Every turn participates: an intervening unembedded turn cannot be
        # skipped to manufacture an otherwise convenient speaker match.
        previous = [candidate for candidate, (_, other_end) in enumerate(times)
                    if candidate != index and other_end <= start + epsilon]
        following = [candidate for candidate, (other_start, _) in enumerate(times)
                     if candidate != index and other_start >= end - epsilon]
        overlapping = [candidate for candidate, (other_start, other_end) in enumerate(times)
                       if candidate != index and min(end, other_end) - max(start, other_start) > epsilon]
        if overlapping:
            reason = "fala sobreposta sem identidade própria confirmada"
        elif previous and following:
            nearest_end = max(times[candidate][1] for candidate in previous)
            nearest_start = min(times[candidate][0] for candidate in following)
            left = [candidate for candidate in previous if abs(times[candidate][1] - nearest_end) <= epsilon]
            right = [candidate for candidate in following if abs(times[candidate][0] - nearest_start) <= epsilon]
            if len(left) != 1 or len(right) != 1:
                reason = "mais de um turno vocal vizinho candidato"
            elif left[0] not in mapping or right[0] not in mapping:
                reason = "turno vocal vizinho também sem identidade confirmada"
            elif mapping[left[0]] != mapping[right[0]]:
                reason = "vozes diferentes antes e depois do trecho curto"
            elif start - nearest_end > maximum_gap + epsilon or nearest_start - end > maximum_gap + epsilon:
                reason = "intervalo maior que 0,20 s até um dos turnos confirmados"
            else:
                assignments[index] = (mapping[left[0]], "bilateral_same_speaker_context")
                continue
        raise PipelineError(f"Diarização incerta: trecho {item.get('id', index + 1)} "
                            f"({start:.3f}–{end:.3f} s), {reason}. "
                            "Revise a associação de locutor ou a divisão da fala; nenhum timbre será adivinhado.")
    # An error never leaves partially rewritten identities in the caller's
    # ASR checkpoint. Successful changes keep the original speech untouched.
    result = [dict(item) for item in segments]
    for index, (label, evidence) in assignments.items():
        result[index].update(speaker=f"SPEAKER_{label:02d}", speaker_assignment=evidence)
    return result


def diarize_stage(config: dict, segments: list[dict]) -> list[dict]:
    requested = str(config.get("speaker_count", "auto"))
    requested_count = None
    if requested != "auto":
        try:
            requested_count = int(requested)
        except (TypeError, ValueError) as error:
            raise PipelineError("A quantidade de locutores deve ser automática ou um inteiro de 1 a 12.") from error
        if not 1 <= requested_count <= 12:
            raise PipelineError("A quantidade de locutores deve ser automática ou um inteiro de 1 a 12.")
    import numpy as np
    import torch
    import torchaudio
    import huggingface_hub
    from sklearn.cluster import AgglomerativeClustering
    # SpeechBrain 1.0.3 predates Hub 1.x. Adapt only obsolete public parameters;
    # downloads still use the official client/cache and normal authentication.
    original_download = huggingface_hub.hf_hub_download
    def compatible_download(*args, **kwargs):
        if "use_auth_token" in kwargs:
            token = kwargs.pop("use_auth_token")
            if token:
                kwargs["token"] = token
        kwargs.pop("local_dir_use_symlinks", None)
        kwargs.pop("force_filename", None)
        return original_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = compatible_download
    from speechbrain.inference.speaker import EncoderClassifier

    waveform, sample_rate = torchaudio.load(config["vocals"])
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    sample_rate = 16000
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.environ.get("DUBSTUDIO_SPEECHBRAIN_CACHE", str(Path(config["directory"]) / "speaker-model")),
        run_opts={"device": "cuda"},
    )
    vectors, valid_indices = [], []
    try:
        for index, item in enumerate(segments):
            clip = waveform[:, int(float(item["start"]) * sample_rate):int(float(item["end"]) * sample_rate)]
            if clip.shape[1] < int(0.7 * sample_rate):
                continue
            with torch.inference_mode():
                vector = encoder.encode_batch(clip.to("cuda")).detach().float().cpu().numpy().reshape(-1)
            vectors.append(vector / max(1e-8, float(np.linalg.norm(vector))))
            valid_indices.append(index)
            print(f"Locutores: {index + 1}/{len(segments)} segmentos", flush=True)
        if requested_count is not None and len(vectors) < requested_count:
            raise PipelineError(
                f"Diarização incerta: você informou {requested_count} locutores, mas há somente {len(vectors)} "
                "trechos vocais válidos com duração suficiente para gerar embeddings. "
                "A quantidade solicitada não será reduzida silenciosamente. Revise os turnos ou forneça fala mais clara; "
                "os checkpoints anteriores foram preservados. Isso não confirma quantas vozes existem no áudio."
            )
        if not vectors:
            raise PipelineError("Não há referência de fala longa o suficiente para identificar locutores. Selecione um locutor explicitamente ou use uma gravação mais clara.")
        matrix = np.stack(vectors)
        if len(matrix) == 1:
            labels = [0]
        elif requested_count is not None:
            labels = AgglomerativeClustering(n_clusters=requested_count, metric="cosine", linkage="average").fit_predict(matrix)
        else:
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.55, metric="cosine", linkage="average").fit_predict(matrix)
            if len(set(labels)) > 12:
                raise PipelineError("A identificação encontrou mais de doze vozes. Revise o áudio ou indique a quantidade de locutores.")
        mapping = {index: int(label) for index, label in zip(valid_indices, labels)}
        return assign_diarized_speakers(segments, mapping)
    finally:
        del encoder
        huggingface_hub.hf_hub_download = original_download
        gc.collect()
        torch.cuda.empty_cache()


def identity_segments(tracks: list, scores: list, frame_count: int, width: int, height: int, transcript: list[dict], selection: object, operation: str) -> tuple[list[dict], list[dict]]:
    """Strict face/voice association; detected ambiguity is never preserved silently."""
    by_frame: dict[int, list[tuple[int, list[int]]]] = {}
    neutral_by_frame: dict[int, list[tuple[int, list[int]]]] = {}
    neutral_boundaries: set[int] = set()
    score_by_track_frame: dict[tuple[int, int], float | None] = {}
    selected_track = selection.get("track_id") if isinstance(selection, dict) else None
    selection_source = "track_id" if selected_track is not None else None
    if isinstance(selection, dict) and selection.get("bbox"):
        bbox = selection["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return [], [{"start": 0, "end": frame_count / 25, "reason": "seleção facial inválida"}]
        x, y, box_width, box_height = (float(value) for value in bbox)
        if selection.get("normalized"):
            x, y, box_width, box_height = x * width, y * height, box_width * width, box_height * height
        chosen_box = [x, y, x + box_width, y + box_height]
        at_frame = round(float(selection.get("frame_time", 0)) * 25)
        candidates = []
        for index, track in enumerate(tracks):
            frames, boxes = list(track.get("track", {}).get("frame", [])), list(track.get("track", {}).get("bbox", []))
            if at_frame not in frames:
                continue
            box = boxes[frames.index(at_frame)]
            intersection = max(0, min(chosen_box[2], box[2]) - max(chosen_box[0], box[0])) * max(0, min(chosen_box[3], box[3]) - max(chosen_box[1], box[1]))
            area = max(0, box_width) * max(0, box_height) + max(0, box[2] - box[0]) * max(0, box[3] - box[1]) - intersection
            candidates.append((intersection / max(area, 1.0), index))
        candidates.sort(reverse=True)
        if not candidates or candidates[0][0] < 0.25 or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.1):
            return [], [{"start": at_frame / 25, "end": (at_frame + 1) / 25, "reason": "a área selecionada não identifica um único rosto detectado; ajuste a seleção"}]
        selected_track = candidates[0][1]
        selection_source = "bbox"
    for track_index, (track, score_list) in enumerate(zip(tracks, scores)):
        if selected_track is not None and track_index != int(selected_track) and operation == "lipsync_video":
            continue
        raw = track.get("track", {})
        frames, boxes = list(raw.get("frame", [])), list(raw.get("bbox", []))
        values = list(score_list)
        for position, (frame, box) in enumerate(zip(frames, boxes)):
            index = int(frame)
            # Zero is the decision boundary, not positive speaking evidence.
            # Otherwise a silent peripheral face scoring -0.0 creates a false
            # tie with the actual speaking face; genuinely positive ties still
            # fail below and require explicit correction.
            score = float(values[min(len(values) - 1, position * len(values) // max(1, len(frames)))]) if values else None
            active = score is not None and math.isfinite(score) and score > 0
            if selected_track is not None and track_index == int(selected_track):
                active = True  # Explicit user choice still requires a real detected track.
            bbox = [max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))]
            if 0 <= index < frame_count and bbox[2] - bbox[0] >= 24 and bbox[3] - bbox[1] >= 24:
                score_by_track_frame[(track_index, index)] = score
                if operation == "dub_video" and not active:
                    if score == 0:
                        neutral_by_frame.setdefault(index, []).append((track_index, bbox))
                    continue
                by_frame.setdefault(index, []).append((track_index, bbox))
    # A single neutral sample inside the same uniquely positive detected track
    # is not a new scene/identity. Never fill negative runs, multi-frame gaps,
    # missing detections, competing candidates, or a change of face identity.
    for index, neutral in neutral_by_frame.items():
        previous, following = by_frame.get(index - 1, []), by_frame.get(index + 1, [])
        if index in by_frame or len(neutral) != 1 or len(previous) != 1 or len(following) != 1:
            continue
        if previous[0][0] == neutral[0][0] == following[0][0]:
            by_frame[index] = neutral
            neutral_boundaries.add(index)
    if selected_track is not None:
        for index, candidates in by_frame.items():
            chosen = [candidate for candidate in candidates if candidate[0] == int(selected_track)]
            if chosen:
                by_frame[index] = chosen
    ambiguity = [{"start": index / 25, "end": (index + 1) / 25, "reason": "múltiplos rostos candidatos"} for index, candidates in by_frame.items() if len(candidates) != 1]
    if ambiguity:
        return [], ambiguity
    groups = []
    for frame in sorted(by_frame):
        track, box = by_frame[frame][0]
        if not groups or frame != groups[-1]["end_frame"] or track != groups[-1]["track_id"]:
            selected = selected_track is not None and track == int(selected_track)
            external_audio = operation == "lipsync_video"
            coordinate_source = ("S3FD: rosto detectado com seleção do usuário" if selected else "S3FD: rosto detectado para áudio enviado") if external_audio else ("TalkNet com seleção do usuário" if selected else "TalkNet: rosto falante confirmado")
            groups.append({"start_frame": frame, "end_frame": frame + 1, "fps": 25.0, "track_id": track, "boxes": [box], "coordinate_source": coordinate_source,
                           "face_confirmation": "user_selected" if selected else "detected",
                           "speaker_confirmation": "external_audio" if external_audio else ("user_selected" if selected else "active_speaker"),
                           "neutral_boundary_frames": int(frame in neutral_boundaries),
                           "selection_source": selection_source if selected else None})
        else:
            groups[-1]["end_frame"] += 1
            groups[-1]["boxes"].append(box)
            groups[-1]["neutral_boundary_frames"] += int(frame in neutral_boundaries)
    for group in groups:
        if len(group["boxes"]) < 20:
            count = len(group["boxes"])
            selected = group["face_confirmation"] == "user_selected"
            external_audio = operation == "lipsync_video"
            group_scores = [score_by_track_frame.get((group["track_id"], frame)) for frame in range(group["start_frame"], group["end_frame"])]
            finite = all(value is not None and math.isfinite(value) for value in group_scores)
            positive = sum(value is not None and math.isfinite(value) and value > 0 for value in group_scores)
            peak = max(group_scores) if finite else None
            speaking_confirmed = finite and positive * 5 >= count * 4 and peak is not None and peak >= 0.8
            # Only the user-selected, actually detected identity can use a
            # short inference window. Automatic windows remain >=20 frames.
            # External audio confirms geometry, never the original speaker.
            if not selected or count < 12 or (not external_audio and not speaking_confirmed):
                ambiguity.append({"start": group["start_frame"] / 25, "end": group["end_frame"] / 25, "reason": "trecho facial curto demais para confirmar"})
                continue
            group["explicit_selected_short_track"] = {
                "schema_version": 1,
                "policy": "explicit-user-selected-short-track-v1",
                "operation": operation,
                "selection_source": selection_source,
                "track_id": group["track_id"],
                "start_frame": group["start_frame"],
                "end_frame": group["end_frame"],
                "fps": group["fps"],
                "frame_count": count,
                "minimum_frames": 12,
                "boxes_sha256": hashlib.sha256(json.dumps(group["boxes"], separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
                "score_evidence_required": not external_audio,
                "scores": group_scores if not external_audio else [],
                "finite_score_count": count if not external_audio else None,
                "positive_score_count": positive if not external_audio else None,
                "positive_score_ratio": positive / count if not external_audio else None,
                "peak_score": peak if not external_audio else None,
            }
        overlap: dict[str, int] = {}
        start, end = group["start_frame"] / 25, group["end_frame"] / 25
        for item in transcript:
            # Associate identities on the same canonical frame clock as
            # TalkNet. ASR timing itself is not modified. A half-frame rounding
            # difference at a cut must not assign the preceding voice to the
            # next person's face; real overlapping frame intervals still fail.
            # Frame centers are included in [speech_start, speech_end). Ties
            # consistently belong to the following interval, not even/odd
            # Python rounding; tiny float noise cannot change an identity.
            speech_start = math.ceil(float(item["start"]) * 25 - 0.5 - 1e-7)
            speech_end = math.ceil(float(item["end"]) * 25 - 0.5 - 1e-7)
            overlap_frames = max(0, min(group["end_frame"], speech_end) - max(group["start_frame"], speech_start))
            if overlap_frames > 0:
                speaker = item.get("speaker")
                if operation == "dub_video" and (not isinstance(speaker, str) or not speaker.strip()):
                    ambiguity.append({"start": start, "end": end, "reason": "fala transcrita sem identidade vocal confirmada; revise a diarização"})
                    continue
                if operation == "lipsync_video":
                    continue  # User audio confirms geometry, not an original vocal identity.
                overlap[speaker] = overlap.get(speaker, 0) + overlap_frames
        if len(overlap) > 1:
            ambiguity.append({"start": start, "end": end, "reason": "uma trilha facial contém mais de um locutor; revise a associação"})
        if operation == "dub_video" and not overlap:
            ambiguity.append({"start": start, "end": end, "reason": "rosto candidato sem fala transcrita correspondente; revise a associação de rosto e voz"})
        else:
            group["speaker"] = "USER_AUDIO" if operation == "lipsync_video" else next(iter(overlap))
    return ([] if ambiguity else groups), ambiguity


def stage_main(name: str, config: dict) -> None:
    import dubstudio_colab_worker as legacy
    directory = Path(config["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "stage.log"
    original_progress = legacy.write_progress
    def notebook_progress(job_dir, progress, stage, message):
        original_progress(job_dir, progress, stage, message)
        print(f"{stage}: {message}", flush=True)
    legacy.write_progress = notebook_progress
    if name == "separation":
        vocals, background = legacy.separate_audio(Path(config["video"]), directory, log)
        shutil.copy2(vocals, directory / "vocals.wav")
        shutil.copy2(background, directory / "background.wav")
    elif name == "asr":
        asr_stage(config)
    elif name == "diarization":
        payload = read_json(Path(config["transcript"]))
        segments = payload["segments"]
        if str(config.get("speaker_count")) == "1":
            # A declared single speaker skips acoustic clustering; it is not
            # a model confidence score and must not inherit an old score.
            segments = [dict(item) for item in segments]
            for item in segments:
                item.pop("speaker_confidence", None)
                item.update(speaker="SPEAKER_00", speaker_assignment="user_declared_count")
            payload = {**payload, "speaker_count_evidence": {
                "method": "user_declared_count", "requested": 1, "acoustic_validation": False,
            }}
        else:
            segments = diarize_stage(config, segments)
        write_json(Path(config["output"]), {**payload, "segments": segments})
    elif name == "references":
        segments = read_json(Path(config["segments"]))["segments"]
        if config.get("reference"):
            if len({item["speaker"] for item in segments}) != 1:
                raise PipelineError("Uma referência única não pode substituir vários locutores diferentes.")
            reference = directory / f"reference-{segments[0]['speaker']}.wav"
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["reference"], "-ac", "1", "-ar", "24000", str(reference)])
            refs = {segments[0]["speaker"]: reference}
        else:
            refs = legacy.voice_references(Path(config["vocals"]), segments, directory)
        if any(media_info(file)["duration"] < 0.8 for file in refs.values()):
            raise PipelineError("A referência vocal é curta demais para preservar o timbre. Envie uma amostra mais longa ou revise os locutores.")
        write_json(Path(config["output"]), {speaker: str(file) for speaker, file in refs.items()})
    elif name == "translation":
        run_command([os.environ.get("DUBSTUDIO_TRANSLATE_PYTHON", sys.executable), str(Path(__file__).with_name("dubstudio_translate.py")), "--input", config["segments"], "--output", config["output"], "--source", config["source_language"], "--target", config["language"]], log)
    elif name == "synthesis":
        synthesize_stage(config, legacy)
    elif name == "analysis":
        analysis_video = Path(config["video"])
        if config.get("audio"):
            analysis_video = directory / "speaker-analysis-input.mp4"
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["video"], "-i", config["audio"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-t", str(config["duration"]), str(analysis_video)])
        talknet = Path(os.environ["DUBSTUDIO_TALKNET_ROOT"])
        normalization_revision = legacy.patch_talknet_video_normalization(talknet)
        demo = talknet / "demo"
        demo.mkdir(exist_ok=True)
        root = demo / analysis_video.stem
        # Migrate archives produced by older runners without deleting evidence.
        for archive in demo.glob(f"{analysis_video.stem}.previous-*"):
            if archive.is_dir():
                archive.rename(archive.with_name(f".previous-{archive.name}-{time.time_ns()}"))
        if root.exists():
            # TalkNet takes the first match for videoName + '.*'. An archive
            # with that prefix would be selected as input instead of the MP4.
            root.rename(root.with_name(f".previous-{root.name}-{time.time_ns()}"))
        shutil.copy2(analysis_video, demo / f"{analysis_video.stem}.mp4")
        # This narrow NumPy repair belongs to the disposable runtime checkout,
        # not the user's immutable source package.
        box_utils = talknet / "model/faceDetector/s3fd/box_utils.py"
        if box_utils.exists():
            source = box_utils.read_text(encoding="utf-8")
            patched = source.replace(".astype(np.int)", ".astype(int)")
            if patched != source:
                box_utils.write_text(patched, encoding="utf-8")
        last_display = [0.0]
        def display_analysis(line):
            if time.monotonic() - last_display[0] >= 2.0 or any(word in line.lower() for word in ("error", "traceback", "finished", "saved")):
                print(f"TalkNet: {line}", flush=True)
                last_display[0] = time.monotonic()
        # Direct execution avoids redundant legacy optical/fallback analyses.
        # The only accepted coordinates below come from this complete TalkNet run.
        legacy.run_streaming([sys.executable, "-u", "demoTalkNet.py", "--videoName", analysis_video.stem, "--videoFolder", str(demo)], cwd=talknet, log=log, on_output=display_analysis)
        analysis = {"status": "complete", "scene_analysis_complete": True,
                    "normalization_revision": normalization_revision}
        tracks_path, scores_path = root / "pywork/tracks.pckl", root / "pywork/scores.pckl"
        normalized = root / "pyavi/video.avi"
        if not tracks_path.exists() or not scores_path.exists() or not normalized.exists():
            raise PipelineError("A análise de rostos não produziu uma trilha verificável.")
        with tracks_path.open("rb") as handle:
            tracks = pickle.load(handle)
        with scores_path.open("rb") as handle:
            scores = pickle.load(handle)
        info = media_info(normalized)
        stream = next(item for item in info["streams"] if item["codec_type"] == "video")
        transcript = read_json(Path(config["segments"]))["segments"] if config.get("segments") else []
        confirmed, ambiguity = identity_segments(tracks, scores, legacy.video_frame_count(normalized), int(stream["width"]), int(stream["height"]), transcript, config.get("face_selection"), config["operation"])
        analysis.update(confirmed_face_segments=confirmed, ambiguity=ambiguity, safe_for_single_face_lipsync=bool(confirmed) and not ambiguity)
        if config["operation"] == "dub_video" and isinstance(tracks, list) and not tracks and not ambiguity:
            # Absence of detected faces is not confirmation of applied lip-sync.
            analysis.update(no_visible_speaking_face=True,
                            preserved_no_speaker_visual_segments=[{"start": 0.0, "end": float(config["duration"]), "reason": "Análise completa: nenhum rosto foi detectado; imagem original preservada"}])
        # A local checkpoint carries normalized media needed by the legacy renderer.
        local_normalized = directory / "talknet-normalized.avi"
        shutil.copy2(normalized, local_normalized)
        analysis["normalized_video"] = str(local_normalized)
        write_json(Path(config["output"]), analysis)
        if ambiguity:
            raise PipelineError("Rosto/locutor ambíguo. Corrija a seleção ou divisão de cenas; nenhum vídeo sem lip-sync confirmado será entregue.")
        if not confirmed and analysis.get("no_visible_speaking_face") is not True:
            raise PipelineError("Nenhum trecho de rosto foi confirmado para lip-sync. Verifique se o vídeo mostra o rosto falante ou selecione o rosto na operação com áudio.")
    elif name == "retime":
        from dubstudio_audio_master import retime_video
        retime_video(config, legacy)
    elif name == "uploaded_master":
        from dubstudio_audio_master import build_timeline
        # Decode the complete uploaded track. No -t, -shortest or tempo filter.
        run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["audio"], "-vn", "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", config["output"]])
        with wave.open(config["output"], "rb") as pcm:
            frames, rate = pcm.getnframes(), pcm.getframerate()
        segments = [{"id": 1, "start": 0.0, "end": float(config["source_duration"]), "speaker": "USER_AUDIO", "text": "Áudio autorizado enviado pelo usuário"}]
        master = build_timeline(segments, [frames], sample_rate=rate)
        master.update(timing_mode="audio_master", target_language=config["language"], input_sha256=config["identity"]["input_sha256"], source_video_sha256=config["source_video_sha256"],
                      submitted_audio_sha256=config["audio_hash"], audio_master={"path": "dialogue.wav", "sha256": sha256(Path(config["output"])), "sample_rate": rate, "sample_count": frames})
        write_json(Path(config["timeline"]), master)
    elif name == "lipsync":
        analysis = read_json(Path(config["analysis"]))
        if not analysis.get("confirmed_face_segments") or analysis.get("ambiguity"):
            raise PipelineError("Lip-sync sem confirmação facial foi recusado.")
        # Legacy renderer uses one normalized path; rebuild from checkpoint locally.
        normalized = Path(os.environ["DUBSTUDIO_TALKNET_ROOT"]) / "demo" / Path(config["video"]).stem / "pyavi/video.avi"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(analysis["normalized_video"]), normalized)
        # Prevent app CUDA libraries from taking precedence over MuseTalk's isolated Torch.
        muse_env = Path(os.environ["DUBSTUDIO_MUSETALK_PYTHON"]).parent.parent
        from dubstudio_bootstrap import musetalk_environment
        # update() only replaces existing values; it never deletes keys omitted
        # by the isolated environment. Torch 2.0 rejects expandable_segments.
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ.update(musetalk_environment(muse_env.parent, os.environ))
        # A previous interrupted render must not be mistaken for a newly produced
        # result, nor excluded by the legacy renderer when overwritten in place.
        for result_directory in directory.glob("musetalk-result-*"):
            if result_directory.is_dir():
                result_directory.rename(result_directory.with_name(f".previous-{result_directory.name}-{time.time_ns()}"))
        original_streaming = legacy.run_streaming
        last_display = [0.0]
        def low_memory_command(command, *args, **kwargs):
            if "--batch_size" in command:
                command = list(command)
                command[command.index("--batch_size") + 1] = os.environ.get("DUBSTUDIO_MUSETALK_BATCH", "4")
            callback = kwargs.get("on_output")
            def display_render(line):
                if callback:
                    callback(line)
                if time.monotonic() - last_display[0] >= 2.0 or any(word in line.lower() for word in ("error", "traceback", "saved", "finished")):
                    print(f"MuseTalk: {line}", flush=True)
                    last_display[0] = time.monotonic()
            kwargs["on_output"] = display_render
            try:
                return original_streaming(command, *args, **kwargs)
            except legacy.WorkerError as exc:
                if "out of memory" not in str(exc).lower() or "--batch_size" not in command or command[command.index("--batch_size") + 1] == "1":
                    raise
                print("Memória GPU insuficiente: retomando somente lip-sync com lote 1.", flush=True)
                command[command.index("--batch_size") + 1] = "1"
                return original_streaming(command, *args, **kwargs)
        legacy.run_streaming = low_memory_command
        rendered, skipped = legacy.musetalk(Path(config["video"]), Path(config["audio"]), directory, analysis, log, True, True)
        if skipped:
            raise PipelineError("O renderizador deixou trechos de rosto sem sincronização.")
        shutil.copy2(rendered, Path(config["output"]))
        write_json(Path(config["proof"]), analysis)
    elif name == "mix":
        duration = float(config["duration"])
        if config.get("timing_mode") == "audio_master":
            with wave.open(config["audio"], "rb") as master:
                frames, rate = master.getnframes(), master.getframerate()
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["audio"]]
            if config.get("background") and config.get("preserve_ambience"):
                command += ["-i", config["background"], "-filter_complex", "[1:a]volume=0.35[bed];[0:a][bed]amix=inputs=2:duration=first:dropout_transition=0:normalize=0,alimiter=limit=0.95:level=false:latency=1[out]", "-map", "[out]"]
            command += ["-ac", "2", "-ar", str(rate), "-c:a", "pcm_s16le", config["output"]]
            run_command(command)
            with wave.open(config["output"], "rb") as mixed:
                if mixed.getframerate() != rate or mixed.getnframes() != frames:
                    raise PipelineError("A mixagem modificou o número de amostras da fala natural; exportação bloqueada.")
        elif config.get("background") and config.get("preserve_ambience"):
            output = legacy.mix_audio(Path(config["background"]), Path(config["audio"]), duration, directory, log)
            if output != Path(config["output"]):
                shutil.copy2(output, Path(config["output"]))
        else:
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["audio"], "-af", "apad,loudnorm=I=-14:TP=-1.5:LRA=11", "-t", str(duration), "-ac", "2", "-ar", "48000", config["output"]])
    elif name == "export":
        export_options = {}
        if config.get("timing_mode") == "audio_master":
            export_options = {"timing_mode": "audio_master", "timeline": read_json(Path(config["timeline"]))}
        final = legacy.finalize(Path(config["video"]), Path(config["audio"]), Path(config["source"]), directory, float(config["duration"]), log, **export_options)
        if final != Path(config["output"]):
            shutil.copy2(final, Path(config["output"]))
        if config.get("subtitle_mode") in {"burn", "burned"}:
            subtitle = Path(config["srt"]).resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
            temporary = directory / "subtitled.mp4"
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(final), "-vf", f"subtitles='{subtitle}'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "copy", str(temporary)])
            shutil.copy2(temporary, Path(config["output"]))
        write_json(Path(config["validation"]), verify_final_media(Path(config["output"]), float(config["duration"]), audio_master=config.get("timing_mode") == "audio_master"))
    else:
        raise PipelineError(f"Etapa desconhecida: {name}")


def bundle(root: Path, output: Path, metadata_name: str, metadata: dict) -> None:
    write_json(root / metadata_name, metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.write(root / metadata_name, metadata_name)
        if metadata_name == "result.json":
            for record in metadata["outputs"]:
                for relative in record.get("files", {}).values():
                    file = safe_path(root, relative)
                    if file.is_file():
                        archive.write(file, relative)


def process(package: Path, work_root: Path, storage: Path, output: Path) -> dict:
    from dubstudio_bootstrap import copy_missing, gpu_precheck
    import dubstudio_colab_worker as legacy
    manifest = validate_manifest(package)
    gpu = gpu_precheck()
    options = manifest.get("options", {})
    operation = manifest["operation"]
    audio_master_mode = options.get("timing_mode", "source_video") == "audio_master" and operation != "text_to_voice"
    directory = work_root / manifest["job_id"]
    saved = storage / "jobs" / manifest["job_id"]
    copy_missing(saved, directory, verify=True)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"job_id": manifest["job_id"], "input_sha256": manifest["input_sha256"], "pipeline_version": PIPELINE_VERSION}
    if audio_master_mode:
        identity["timing_mode"] = "audio_master"
    identity_file = directory / "identity.json"
    if identity_file.exists() and read_json(identity_file) != identity:
        raise PipelineError("Os checkpoints encontrados pertencem a outra entrada/versão. Exporte um novo projeto.")
    write_json(identity_file, identity)
    cache = StageCache(directory, saved)
    inputs = {}
    for kind, item in manifest["inputs"].items():
        source = safe_path(package, item["path"])
        target = directory / "inputs" / f"{kind}{source.suffix.lower()}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha256(target) != item["sha256"]:
            shutil.copy2(source, target)
        inputs[kind] = target
    shared = directory / "shared"
    shared.mkdir(exist_ok=True)
    base_config = {"directory": str(shared), "identity": identity}
    video = inputs.get("video")
    duration = input_stream_duration(media_info(video), "video") if video else 0.0
    if video and (duration <= 0 or duration > 1200):
        raise PipelineError("Vídeos precisam ter duração entre zero e vinte minutos.")
    approved_snapshot = load_approved_snapshot(package, manifest, duration)
    diarization = shared / "diarization.json"
    analysis = shared / "active-speaker.json"
    references = shared / "references.json"
    source_language = str(options.get("source_language", "auto"))
    started = time.time()
    if operation == "dub_video":
        cache.stage("separation", "Separação de áudio na GPU", {**base_config, "stage": "separation", "video": str(video), "video_hash": sha256(video)}, [shared / "vocals.wav", shared / "background.wav"])
        if approved_snapshot:
            frozen = approved_snapshot["source"]
            write_json(diarization, {"segments": frozen["segments"], "language": frozen["source_language"],
                                     "approved_source_snapshot_sha256": approved_snapshot["source_sha256"]})
            cache.used["approved_source_snapshot"] = True
            cache.seconds["approved_source_snapshot"] = 0.0
            print("Transcrição aprovada preservada: ASR e identificação de locutores não serão refeitas.", flush=True)
        else:
            cache.stage("asr", "Transcrição na GPU", {**base_config, "stage": "asr", "vocals": str(shared / "vocals.wav"), "vocals_hash": sha256(shared / "vocals.wav"), "source_language": source_language, "profile": options.get("performance_profile", "fast_lipsync"), "output": str(shared / "transcript.json")}, [shared / "transcript.json"])
            cache.stage("diarization", "Identificação de locutores", {**base_config, "stage": "diarization", "vocals": str(shared / "vocals.wav"), "vocals_hash": sha256(shared / "vocals.wav"), "transcript": str(shared / "transcript.json"), "transcript_hash": sha256(shared / "transcript.json"), "speaker_count": options.get("speaker_count", "auto"), "output": str(diarization)}, [diarization])
        source_language = read_json(diarization)["language"]
        reference_files = [shared / f"reference-{speaker}.wav" for speaker in sorted({item["speaker"] for item in read_json(diarization)["segments"]})]
        cache.stage("references", "Referências do timbre original", {**base_config, "stage": "references", "vocals": str(shared / "vocals.wav"), "vocals_hash": sha256(shared / "vocals.wav"), "segments": str(diarization), "segments_hash": sha256(diarization), "reference": str(inputs["reference"]) if "reference" in inputs else None, "reference_hash": sha256(inputs["reference"]) if "reference" in inputs else None, "output": str(references)}, [references, *reference_files])
    elif operation == "text_to_voice":
        # This mode speaks exactly the entered text; choosing a language does not translate it.
        text = str(options["text"]).strip()
        from dubstudio_translate import split_text
        pieces = split_text(text, 400)
        segments = [{"id": index + 1, "start": 0.0, "end": 0.0, "text": piece, "translated_text": piece, "speaker": "SPEAKER_00"} for index, piece in enumerate(pieces)]
        write_json(diarization, {"segments": segments, "language": source_language})
        write_json(references, {"SPEAKER_00": str(inputs["reference"])})
        if media_info(inputs["reference"])["duration"] < 0.8:
            raise PipelineError("Envie referência vocal com pelo menos um segundo de fala.")
    else:
        audio_duration = input_stream_duration(media_info(inputs["audio"]), "audio")
        if audio_master_mode:
            if not math.isfinite(audio_duration) or audio_duration <= 0:
                raise PipelineError("O áudio enviado precisa ter duração positiva e finita.")
            # Voice/face attribution of multiple uploaded voices cannot be
            # guessed from an unrelated soundtrack. Detect acoustic speakers
            # before allowing the single-face uploaded-audio adapter.
            cache.stage("uploaded-asr", "Conferindo as vozes do áudio enviado", {**base_config, "stage": "asr", "vocals": str(inputs["audio"]), "vocals_hash": sha256(inputs["audio"]), "source_language": source_language, "profile": "quality", "output": str(shared / "uploaded-transcript.json")}, [shared / "uploaded-transcript.json"])
            cache.stage("uploaded-diarization", "Conferindo a associação do áudio enviado", {**base_config, "stage": "diarization", "vocals": str(inputs["audio"]), "vocals_hash": sha256(inputs["audio"]), "transcript": str(shared / "uploaded-transcript.json"), "transcript_hash": sha256(shared / "uploaded-transcript.json"), "speaker_count": "auto", "output": str(shared / "uploaded-speakers.json")}, [shared / "uploaded-speakers.json"])
            if len({item["speaker"] for item in read_json(shared / "uploaded-speakers.json")["segments"]}) != 1:
                raise PipelineError("O áudio enviado contém várias vozes sem associação explícita aos rostos. Use Dublar vídeo para preservar cada locutor, ou envie um áudio de uma só pessoa; nenhuma voz será atribuída a outro rosto.")
        else:
            validate_uploaded_audio_duration(audio_duration, duration)
        write_json(diarization, {"segments": [], "language": source_language})
    source_segments = read_json(diarization)["segments"]
    approved = approved_snapshot["translations"] if approved_snapshot else None
    translations = {}
    translation_errors = {}
    review_required = options.get("review_mode") in {True, "before_voice", "before_synthesis", "manual", "required"}
    if operation == "dub_video":
        for language in manifest["target_languages"]:
            target = directory / "outputs" / language
            target.mkdir(parents=True, exist_ok=True)
            translated = target / "segments.json"
            try:
                if approved:
                    write_json(translated, {"source_language": source_language, "target_language": language, "translation_engine": "revisão aprovada no Studio", "segments": approved[language]})
                else:
                    cache.stage(f"translation-{language}", f"Tradução para {language}", {"directory": str(target), "stage": "translation", "identity": identity, "segments": str(diarization), "segments_hash": sha256(diarization), "source_language": source_language, "language": language, "output": str(translated)}, [translated])
                translations[language] = read_json(translated)["segments"]
            except Exception as exc:
                translation_errors[language] = str(exc)
                print(f"FALHA na tradução de {language}: {exc}. As demais traduções continuarão.", flush=True)
        if review_required and not approved:
            if translation_errors:
                raise PipelineError(f"A revisão completa ainda não está pronta: traduções falharam em {', '.join(translation_errors)}. Traduções bem-sucedidas estão nos checkpoints para retomada.")
            review = {"schema_version": SCHEMA_VERSION, **identity, "source_language": source_language, "translations": translations}
            cache.persist()
            bundle(directory, output, "review.json", review)
            print("Revisão pronta: importe o ZIP no Studio, revise e exporte o pacote aprovado para retomar.", flush=True)
            return review
    require_lipsync = needs_lipsync(manifest)
    if operation == "dub_video" and not require_lipsync and not audio_master_mode:
        print("Dublagem sem lip-sync: análise facial, TalkNet e MuseTalk não serão executados.", flush=True)
    if video and (require_lipsync or audio_master_mode) and (operation != "dub_video" or translations):
        analysis_label = "Detectando rostos para o áudio enviado" if operation == "lipsync_video" else "Confirmação dos rostos e locutores"
        cache.stage("analysis", analysis_label, {**base_config, "stage": "analysis", "video": str(video), "video_hash": sha256(video), "audio": str(inputs["audio"]) if operation == "lipsync_video" else None, "audio_hash": sha256(inputs["audio"]) if operation == "lipsync_video" else None, "duration": duration, "operation": operation, "segments": str(diarization), "segments_hash": sha256(diarization), "face_selection": options.get("face_selection", "auto"), "output": str(analysis)}, [analysis, shared / "talknet-normalized.avi"])
    visual_analysis = read_json(analysis) if video and (require_lipsync or audio_master_mode) and analysis.exists() else {}
    input_checksums = {role: item["sha256"] for role, item in manifest["inputs"].items()}
    records = []
    for language in manifest["target_languages"]:
        target = directory / "outputs" / language
        target.mkdir(parents=True, exist_ok=True)
        timing_file = target / "audio-master.json"
        render_video, render_analysis, render_duration = video, analysis, duration
        background = shared / "background.wav" if operation == "dub_video" and options.get("preserve_ambience", True) else None
        record = {"language": language, "status": "failed", "files": {}, "checksums": {}}
        report = {"schema_version": SCHEMA_VERSION, **identity, "operation": operation, "input_checksums": input_checksums, "target_language": language, "source_language": source_language, "gpu": gpu, "voice_clone": {"required": operation != "lipsync_video", "confirmed": False, "authorized": True}, "lipsync": {"required": require_lipsync, "confirmed": False, "applied": False, "confirmed_segments": [], "ambiguity": []}, "preserved_no_speaker_visual_segments": [], "validation_state": "machine_checks_pending_human_review", "started_at": int(started)}
        if operation == "dub_video" and not require_lipsync:
            report["mode_label"] = "Dublagem sem lip-sync"
        try:
            if language in translation_errors:
                raise PipelineError(f"Tradução de {language} não concluída: {translation_errors[language]}")
            translated = target / "segments.json"
            if operation == "text_to_voice":
                write_json(translated, {"target_language": language, "segments": source_segments})
            if operation != "lipsync_video":
                refs = read_json(references)
                synthesis_outputs = [target / "dialogue.wav", target / "voice-proof.json"]
                if operation == "text_to_voice" or audio_master_mode:
                    synthesis_outputs.append(target / "voice-segments.json")
                if audio_master_mode:
                    synthesis_outputs.append(timing_file)
                cache.stage(f"synthesis-{language}", f"Timbre original em {language}", {"directory": str(target), "stage": "synthesis", "identity": identity, "segments": str(translated), "segments_hash": sha256(translated), "references": refs, "reference_hashes": {key: sha256(Path(value)) for key, value in refs.items()}, "language": language, "duration": duration, "timing_mode": "audio_master" if audio_master_mode else "source_video", "source_video_sha256": sha256(video) if video else None}, synthesis_outputs)
                report.update(read_json(target / "voice-proof.json"))
                audio = target / "dialogue.wav"
                if video:
                    subtitle_segments = read_json(target / "voice-segments.json")["segments"] if audio_master_mode else read_json(translated)["segments"]
                    legacy.write_srt(target / "original.srt", subtitle_segments if audio_master_mode else source_segments, "text")
                    legacy.write_srt(target / "translation.srt", subtitle_segments, "translated_text")
                report["translation"] = {"target_language": language, "segments_validated": len(source_segments), "engine": read_json(translated).get("translation_engine", "texto do usuário"), "reviewed": bool(approved)}
            else:
                audio = inputs["audio"]
                report["voice_engine"] = "áudio enviado pelo usuário"
                report["voice_clone"] = {"required": False, "confirmed": False, "authorized": True}
                if audio_master_mode:
                    audio = target / "dialogue.wav"
                    cache.stage(f"uploaded-master-{language}", "Preservando o áudio completo enviado", {"directory": str(target), "stage": "uploaded_master", "identity": identity, "audio": str(inputs["audio"]), "audio_hash": sha256(inputs["audio"]), "source_duration": duration, "source_video_sha256": sha256(video), "language": language, "output": str(audio), "timeline": str(timing_file)}, [audio, timing_file])
            files = {}
            if video:
                if audio_master_mode:
                    render_video, render_analysis = target / "base-video.mp4", target / "mapped-analysis.json"
                    mapped_background = target / "mapped-background.wav" if background else None
                    retime_outputs = [render_video, render_analysis, target / "timeline.json"]
                    if mapped_background:
                        retime_outputs.append(mapped_background)
                    cache.stage(f"retime-{language}", f"Montagem seguindo a voz natural em {language}", {"directory": str(target), "stage": "retime", "identity": identity, "video": str(video), "video_hash": sha256(video), "timeline": str(timing_file), "timeline_hash": sha256(timing_file), "timeline_output": str(target / "timeline.json"), "analysis": str(analysis), "analysis_hash": sha256(analysis), "require_faces": visual_analysis.get("no_visible_speaking_face") is not True, "prepare_lipsync": require_lipsync, "output": str(render_video), "mapped_analysis": str(render_analysis), "background": str(background) if background else None, "background_hash": sha256(background) if background else None, "background_output": str(mapped_background) if mapped_background else None}, retime_outputs)
                    timing_file = target / "timeline.json"
                    from dubstudio_audio_master import validate_timeline
                    timing = validate_timeline(read_json(timing_file))
                    with wave.open(str(audio), "rb") as pcm:
                        render_duration = pcm.getnframes() / pcm.getframerate()
                    report.update(timing_mode="audio_master", timeline_sha256=sha256(timing_file), master_audio_sha256=sha256(audio), timing={key: timing[key] for key in ("policy", "sample_rate", "sample_count", "source_segments_sha256", "video_base")})
                    background = mapped_background
                    files.update(timeline=timing_file, master_audio=audio)
                cache.stage(f"mix-{language}", f"Mixagem {language}", {"directory": str(target), "stage": "mix", "identity": identity, "audio": str(audio), "audio_hash": sha256(audio), "duration": render_duration, "timing_mode": "audio_master" if audio_master_mode else "source_video", "background": str(background) if background else None, "background_hash": sha256(background) if background else None, "preserve_ambience": bool(options.get("preserve_ambience", True)), "output": str(target / "mix.wav")}, [target / "mix.wav"])
                synced = render_video
                current_analysis = read_json(render_analysis) if audio_master_mode else visual_analysis
                if require_lipsync and current_analysis.get("no_visible_speaking_face") is True:
                    if operation != "dub_video" or current_analysis.get("scene_analysis_complete") is not True:
                        raise PipelineError("A operação de lip-sync com áudio exige ao menos um rosto realmente sincronizado.")
                    report["lipsync"] = {"required": True, "confirmed": False, "applied": False, "confirmed_segments": [], "ambiguity": [], "no_visible_speaking_face": True, "scene_analysis_complete": True}
                    report["preserved_no_speaker_visual_segments"] = current_analysis["preserved_no_speaker_visual_segments"]
                    report["mode_label"] = "Dublagem com narração; nenhum rosto detectado, lip-sync não aplicável"
                elif require_lipsync:
                    sync_audio = target / "mix.wav" if operation == "lipsync_video" else audio
                    cache.stage(f"lipsync-{language}", f"Lip-sync {language}", {"directory": str(target), "stage": "lipsync", "identity": identity, "video": str(render_video), "audio": str(sync_audio), "audio_hash": sha256(sync_audio), "analysis": str(render_analysis), "analysis_hash": sha256(render_analysis), "timing_mode": "audio_master" if audio_master_mode else "source_video", "timeline": str(timing_file) if audio_master_mode else None, "timeline_hash": sha256(timing_file) if audio_master_mode else None, "output": str(target / "synced.mp4"), "proof": str(target / "lipsync-proof.json")}, [target / "synced.mp4", target / "lipsync-proof.json"])
                    synced = target / "synced.mp4"
                    proof = read_json(target / "lipsync-proof.json")
                    confirmed = []
                    for face in proof.get("confirmed_face_segments", []):
                        segment_proof = {"start": face["start_frame"] / face["fps"], "end": face["end_frame"] / face["fps"], "speaker": face.get("speaker", "USER_AUDIO"), "coordinate_source": face["coordinate_source"], "face_confirmation": face.get("face_confirmation"), "speaker_confirmation": face.get("speaker_confirmation"), "neutral_boundary_frames": int(face.get("neutral_boundary_frames", 0))}
                        if face.get("explicit_selected_short_track"):
                            segment_proof["explicit_selected_short_track"] = face["explicit_selected_short_track"]
                        confirmed.append(segment_proof)
                    report["lipsync"] = {"required": True, "confirmed": bool(confirmed) and not proof.get("ambiguity"), "applied": bool(confirmed) and not proof.get("ambiguity"), "confirmed_segments": confirmed, "ambiguity": proof.get("ambiguity", []), "coordinate_source": proof.get("coordinate_source")}
                    report["lipsync"]["neutral_boundary_policy"] = "Somente um quadro neutro isolado entre evidências positivas da mesma trilha, sem outro candidato; não equivale a score positivo."
                    spatial = public_spatial_composition(proof.get("spatial_composition"), len(confirmed))
                    if spatial is not None:
                        report["spatial_composition"] = spatial
                    if operation == "lipsync_video":
                        report["lipsync"]["coordinate_source"] = "S3FD/trilhas faciais para áudio enviado; fala original não confirmada"
                    report["preserved_no_speaker_visual_segments"] = proof.get("preserved_no_speaker_visual_segments", [])
                validate_lipsync_timeline(report, input_stream_duration(media_info(render_video), "video") if audio_master_mode else duration)
                cache.stage(f"export-{language}", f"Exportação {language}", {"directory": str(target), "stage": "export", "identity": identity, "video": str(synced), "video_hash": sha256(synced), "audio": str(target / "mix.wav"), "audio_hash": sha256(target / "mix.wav"), "source": str(video), "source_hash": sha256(video), "duration": render_duration, "timing_mode": "audio_master" if audio_master_mode else "source_video", "timeline": str(timing_file) if audio_master_mode else None, "timeline_hash": sha256(timing_file) if audio_master_mode else None, "subtitle_mode": options.get("subtitle_mode", "none"), "srt": str(target / "translation.srt"), "srt_hash": sha256(target / "translation.srt") if options.get("subtitle_mode", "none") in {"burn", "burned"} else None, "output": str(target / "video.mp4"), "validation": str(target / "media-validation.json")}, [target / "video.mp4", target / "media-validation.json"])
                report["summary"] = read_json(target / "media-validation.json")
                files["video"] = target / "video.mp4"
                files["wav"] = target / "mix.wav"
            else:
                report["summary"] = {"duration": media_info(audio)["duration"], "audio_channels": 2, "timing_tolerance_seconds": 0.0}
                files["wav"] = audio
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(files["wav"]), "-c:a", "libmp3lame", "-b:a", "192k", str(target / "voice.mp3")])
            files["mp3"] = target / "voice.mp3"
            if operation == "dub_video":
                files.update(segments=translated, original_srt=target / "original.srt", translation_srt=target / "translation.srt")
            elif operation == "text_to_voice":
                files["segments"] = target / "voice-segments.json"
            report.update(finished_at=int(time.time()), cache=cache.used, stages_seconds=cache.seconds, gpu_activity=cache.gpu_activity, metrics={"processing_seconds": round(time.time() - started, 3)})
            write_json(target / "report.json", report)
            files["report"] = target / "report.json"
            record.update(status="complete", files={kind: path.relative_to(directory).as_posix() for kind, path in files.items()}, checksums={path.relative_to(directory).as_posix(): sha256(path) for path in files.values()})
            print(f"OK {language}: arquivos e duração conferidos. Revise tradução, timbre e boca na reprodução do Studio.", flush=True)
        except Exception as exc:
            public_error = safe_error_text(exc, 1800)
            diagnostic = failure_diagnostic(exc)
            report.update(error=public_error, diagnostic=diagnostic, finished_at=int(time.time()), cache=cache.used, stages_seconds=cache.seconds, gpu_activity=cache.gpu_activity)
            write_json(target / "report.json", report)
            record.update(error=public_error, diagnostic=diagnostic, files={"report": (target / "report.json").relative_to(directory).as_posix()}, checksums={(target / "report.json").relative_to(directory).as_posix(): sha256(target / "report.json")})
            print(f"FALHA {language}: {exc}. Checkpoints preservados; os demais idiomas continuarão.", flush=True)
        records.append(record)
        cache.persist()
    result = {"schema_version": SCHEMA_VERSION, **identity, "operation": operation, "input_checksums": input_checksums, "outputs": records}
    bundle(directory, output, "result.json", result)
    print(f"RESULTADO={output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--work-root", type=Path, default=Path("/content/dubstudio/jobs"))
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--stage-config", type=Path)
    args = parser.parse_args()
    if args.stage:
        try:
            stage_main(args.stage, read_json(args.stage_config))
        except Exception as exc:
            print(f"ERRO_ETAPA={safe_error_text(exc)}", flush=True)
            raise
    else:
        if not args.package_root or not args.storage_root or not args.output:
            parser.error("Informe --package-root, --storage-root e --output.")
        try:
            process(args.package_root, args.work_root, args.storage_root, args.output)
        except Exception as exc:
            manifest = validate_manifest(args.package_root)
            root = args.work_root / manifest["job_id"]
            failure_identity = {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION,
                                "job_id": manifest["job_id"], "input_sha256": manifest["input_sha256"],
                                "operation": manifest["operation"],
                                "input_checksums": {role: item["sha256"] for role, item in manifest["inputs"].items()}}
            records = []
            public_error = safe_error_text(exc, 1800)
            diagnostic = failure_diagnostic(exc)
            for language in manifest["target_languages"]:
                report = root / "outputs" / language / "report.json"
                write_json(report, {**failure_identity, "target_language": language, "error": public_error, "diagnostic": diagnostic, "status": "failed"})
                relative = report.relative_to(root).as_posix()
                records.append({"language": language, "status": "failed", "error": public_error, "diagnostic": diagnostic, "files": {"report": relative}, "checksums": {relative: sha256(report)}})
            bundle(root, args.output, "result.json", {**failure_identity, "outputs": records})
            print(f"ERRO_PIPELINE={public_error}\nRESULTADO={args.output}", flush=True)
            raise


if __name__ == "__main__":
    main()
