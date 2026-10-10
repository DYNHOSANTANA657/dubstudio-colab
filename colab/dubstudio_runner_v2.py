#!/usr/bin/env python3
"""Orquestrador Colab v2: etapas isoladas, retomada íntegra e saídas por idioma.

Modelos vivem em processos diferentes, portanto VRAM é liberada entre etapas.
Computação ocorre em /content; Drive recebe somente cache e checkpoints.
"""
from __future__ import annotations

import argparse
from array import array
import copy
import gc
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
import zipfile
from pathlib import Path, PurePosixPath

SCHEMA_VERSION = "2.0"
PIPELINE_VERSION = "3.0.0"
OPERATIONS = {"dub_video", "lipsync_video", "text_to_voice"}
CLONE_LANGUAGES = {"ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja", "ko", "ms", "nb", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"}


class PipelineError(RuntimeError):
    pass


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
    files = [Path(__file__), Path(__file__).with_name("dubstudio_colab_worker.py"), Path(__file__).with_name("dubstudio_translate.py")]
    return fingerprint({file.name: sha256(file) for file in files if file.is_file()})


def needs_lipsync(manifest: dict) -> bool:
    operation = manifest.get("operation")
    return operation == "lipsync_video" or (operation == "dub_video" and manifest.get("options", {}).get("require_lipsync", True))


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
        with log.open("a", encoding="utf-8") as stream:
            stream.write("\n$ " + " ".join(command) + "\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise PipelineError(f"A etapa falhou. Confira o log preservado: {log.name}")
    else:
        subprocess.run(command, check=True)


def media_info(file: Path) -> dict:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=index,codec_type,duration,width,height,channels,sample_rate", "-of", "json", str(file)], check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    payload["duration"] = float(payload.get("format", {}).get("duration", 0))
    return payload


def verify_final_media(video: Path, expected_duration: float) -> dict:
    info = media_info(video)
    types = {stream.get("codec_type"): stream for stream in info["streams"]}
    if not {"video", "audio"}.issubset(types):
        raise PipelineError("O resultado está sem faixa de vídeo ou áudio.")
    durations = {kind: float(types[kind].get("duration", 0)) for kind in ("video", "audio")}
    if any(duration <= 0 or abs(duration - expected_duration) > 1.0 for duration in durations.values()):
        raise PipelineError(f"Resultado incompleto: vídeo/áudio {durations}, esperado {expected_duration:.3f}s.")
    return {"duration": info["duration"], "video_duration": durations["video"], "audio_duration": durations["audio"], "timing_tolerance_seconds": max(abs(value - expected_duration) for value in durations.values()), "audio_channels": int(types["audio"].get("channels", 0))}


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
            cached = record.get("signature") == signature and all(file.is_file() and sha256(file) == record.get("checksums", {}).get(file.relative_to(self.root).as_posix()) for file in required)
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


def synthesize_stage(config: dict, legacy) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from pydub import AudioSegment

    directory = Path(config["directory"])
    language = config["language"].split("-")[0]
    language = "no" if language == "nb" else language
    if language not in CLONE_LANGUAGES:
        raise PipelineError("Idioma sem clonagem de timbre. Não há substituição por voz genérica.")
    payload = read_json(Path(config["segments"]))
    segments = payload["segments"]
    references = {key: Path(value) for key, value in config["references"].items()}
    if not references or any(not file.is_file() for file in references.values()):
        raise PipelineError("As referências autorizadas de voz não estão disponíveis.")
    reference_validation = {speaker: validate_voice_reference(file) for speaker, file in references.items()}
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    duration = float(config.get("duration", 0))
    timeline = AudioSegment.silent(duration=math.ceil(duration * 1000), frame_rate=48000).set_channels(2) if duration > 0 else None
    generated = []
    adjustments = []
    try:
        for index, item in enumerate(segments):
            speaker = item.get("speaker", "SPEAKER_00")
            if speaker not in references:
                raise PipelineError(f"Locutor {speaker} sem referência vocal autorizada.")
            print(f"Voz {config['language']}: segmento {index + 1}/{len(segments)}", flush=True)
            raw = directory / f"voice-{index:04d}.wav"
            wave = model.generate(str(item["translated_text"]), language_id=language, audio_prompt_path=str(references[speaker]))
            array = wave.detach().float().cpu().numpy().squeeze()
            if array.size == 0 or float(np.sqrt(np.mean(np.square(array)))) < 1e-6:
                raise PipelineError(f"Síntese silenciosa no segmento {item['id']}.")
            sf.write(raw, array, model.sr)
            actual = media_info(raw)["duration"]
            if timeline is None:
                generated.append(AudioSegment.from_wav(raw).set_frame_rate(48000).set_channels(2))
                continue
            slot = duration_slot(segments, index, duration)
            speaking_time = max(0.15, float(item["end"]) - float(item["start"]))
            # Fit the whole generated speech. Never truncate the last syllables.
            ratio = actual / speaking_time if actual > speaking_time else 1.0
            if ratio > 2.5 and actual > slot:
                raise PipelineError(f"A tradução do segmento {item['id']} é longa demais para a cena. Revise/encurte esse texto e retome o projeto.")
            ratio = max(1.0, actual / min(slot, speaking_time))
            adjusted = directory / f"voice-fitted-{index:04d}.wav"
            position_ms = round(float(item["start"]) * 1000)
            position_frames = position_ms * 48
            available_frames = min(int(slot * 48000), int(timeline.frame_count()) - position_frames)
            ratio = fit_voice_to_frames(raw, adjusted, ratio, available_frames)
            clip = AudioSegment.from_wav(adjusted)
            if int(clip.frame_count()) > available_frames:
                raise PipelineError(f"A fala {item['id']} excede a janela e seria cortada; exportação recusada.")
            timeline = timeline.overlay(clip, position=position_ms)
            adjustments.append({"segment_id": item["id"], "factor": round(ratio, 4), "speech_seconds": actual, "rendered_seconds": len(clip) / 1000, "slot_seconds": slot})
            item["speed"] = ratio
        if timeline is None:
            if not generated:
                raise PipelineError("Nenhuma fala gerada.")
            timeline = generated[0]
            for clip in generated[1:]:
                timeline += AudioSegment.silent(duration=180, frame_rate=48000).set_channels(2) + clip
            rendered_segments = tts_segments_for_frames(segments, [int(clip.frame_count()) for clip in generated])
            write_json(directory / "voice-segments.json", {"target_language": config["language"], "segments": rendered_segments})
            segments = rendered_segments
        timeline.export(directory / "dialogue.wav", format="wav")
        write_json(directory / "voice-proof.json", {"voice_engine": "Chatterbox Multilingual 0.1.7", "voice_clone": {"required": True, "confirmed": True, "authorized": True}, "references": {key: sha256(file) for key, file in references.items()}, "reference_validation": reference_validation, "timing_adjustments": adjustments, "segments": segments})
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def asr_stage(config: dict) -> None:
    from faster_whisper import WhisperModel
    source = config.get("source_language", "auto")
    model_name = "large-v3" if config.get("profile") in {"quality", "maximum_quality"} else "large-v3-turbo"
    model = WhisperModel(model_name, device="cuda", compute_type="int8_float16", download_root=os.environ.get("DUBSTUDIO_WHISPER_CACHE"))
    segments, info = model.transcribe(config["vocals"], language=None if source == "auto" else source.split("-")[0], word_timestamps=True, vad_filter=True, beam_size=5)
    output = []
    for item in segments:
        if not item.text.strip():
            continue
        output.append({"id": len(output) + 1, "start": round(item.start, 3), "end": round(item.end, 3), "text": item.text.strip(), "words": [{"start": word.start, "end": word.end, "word": word.word, "probability": word.probability} for word in (item.words or [])]})
        print(f"Transcrição: {item.end:.1f}s de {info.duration:.1f}s", flush=True)
    if not output:
        raise PipelineError("Nenhuma fala foi reconhecida na mídia.")
    probabilities = [word["probability"] for item in output for word in item["words"]]
    average = sum(probabilities) / len(probabilities) if probabilities else 0.0
    del model
    gc.collect()
    # A second process avoids simultaneous ASR models in VRAM.
    if model_name != "large-v3" and average < 0.6:
        request = dict(config, profile="quality")
        request_file = Path(config["output"]).with_name("asr-quality-request.json")
        write_json(request_file, request)
        run_command([sys.executable, str(Path(__file__).resolve()), "--stage", "asr", "--stage-config", str(request_file)])
        return
    write_json(Path(config["output"]), {"segments": output, "language": info.language, "probability": float(info.language_probability), "word_probability": average, "model": model_name})


def diarize_stage(config: dict, segments: list[dict]) -> list[dict]:
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
        if not vectors:
            raise PipelineError("Não há referência de fala longa o suficiente para identificar locutores. Selecione um locutor explicitamente ou use uma gravação mais clara.")
        matrix = np.stack(vectors)
        requested = str(config.get("speaker_count", "auto"))
        if len(matrix) == 1:
            labels = [0]
        elif requested != "auto":
            labels = AgglomerativeClustering(n_clusters=min(12, len(matrix), max(1, int(requested))), metric="cosine", linkage="average").fit_predict(matrix)
        else:
            labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.55, metric="cosine", linkage="average").fit_predict(matrix)
            if len(set(labels)) > 12:
                raise PipelineError("A identificação encontrou mais de doze vozes. Revise o áudio ou indique a quantidade de locutores.")
        mapping = {index: int(label) for index, label in zip(valid_indices, labels)}
        for index, item in enumerate(segments):
            nearest = index if index in mapping else min(valid_indices, key=lambda candidate: abs(float(segments[candidate]["start"]) - float(item["start"])))
            item.update(speaker=f"SPEAKER_{mapping[nearest]:02d}", speaker_assignment="ecapa_clustering" if index in mapping else "nearest_speech_segment")
        return segments
    finally:
        del encoder
        huggingface_hub.hf_hub_download = original_download
        gc.collect()
        torch.cuda.empty_cache()


def identity_segments(tracks: list, scores: list, frame_count: int, width: int, height: int, transcript: list[dict], selection: object, operation: str) -> tuple[list[dict], list[dict]]:
    """Strict face/voice association; detected ambiguity is never preserved silently."""
    by_frame: dict[int, list[tuple[int, list[int]]]] = {}
    selected_track = selection.get("track_id") if isinstance(selection, dict) else None
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
    for track_index, (track, score_list) in enumerate(zip(tracks, scores)):
        if selected_track is not None and track_index != int(selected_track) and operation == "lipsync_video":
            continue
        raw = track.get("track", {})
        frames, boxes = list(raw.get("frame", [])), list(raw.get("bbox", []))
        values = list(score_list)
        for position, (frame, box) in enumerate(zip(frames, boxes)):
            index = int(frame)
            active = bool(values) and float(values[min(len(values) - 1, position * len(values) // max(1, len(frames)))]) >= 0
            if selected_track is not None and track_index == int(selected_track):
                active = True  # Explicit user choice still requires a real detected track.
            if operation == "dub_video" and not active:
                continue
            bbox = [max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))]
            if 0 <= index < frame_count and bbox[2] - bbox[0] >= 24 and bbox[3] - bbox[1] >= 24:
                by_frame.setdefault(index, []).append((track_index, bbox))
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
            groups.append({"start_frame": frame, "end_frame": frame + 1, "fps": 25.0, "track_id": track, "boxes": [box], "coordinate_source": "TalkNet com seleção do usuário" if selected_track is not None and track == int(selected_track) else "TalkNet confirmado"})
        else:
            groups[-1]["end_frame"] += 1
            groups[-1]["boxes"].append(box)
    for group in groups:
        if len(group["boxes"]) < 20:
            ambiguity.append({"start": group["start_frame"] / 25, "end": group["end_frame"] / 25, "reason": "trecho facial curto demais para confirmar"})
            continue
        overlap: dict[str, float] = {}
        start, end = group["start_frame"] / 25, group["end_frame"] / 25
        for item in transcript:
            seconds = max(0.0, min(end, float(item["end"])) - max(start, float(item["start"])))
            if seconds > 0:
                overlap[item.get("speaker", "SPEAKER_00")] = overlap.get(item.get("speaker", "SPEAKER_00"), 0.0) + seconds
        if len(overlap) > 1:
            ambiguity.append({"start": start, "end": end, "reason": "uma trilha facial contém mais de um locutor; revise a associação"})
        group["speaker"] = next(iter(overlap), "USER_AUDIO" if operation == "lipsync_video" else "SPEAKER_00")
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
            for item in segments:
                item.update(speaker="SPEAKER_00", speaker_confidence=1.0)
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
        if not any(item.get("codec_type") == "audio" for item in media_info(analysis_video)["streams"]) and config.get("audio"):
            analysis_video = directory / "speaker-analysis-input.mp4"
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["video"], "-i", config["audio"], "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-af", "apad", "-t", str(config["duration"]), str(analysis_video)])
        talknet = Path(os.environ["DUBSTUDIO_TALKNET_ROOT"])
        demo = talknet / "demo"
        demo.mkdir(exist_ok=True)
        root = demo / analysis_video.stem
        if root.exists():
            root.rename(root.with_name(root.name + f".previous-{time.time_ns()}"))
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
        analysis = {"status": "complete", "scene_analysis_complete": True}
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
        # A local checkpoint carries normalized media needed by the legacy renderer.
        local_normalized = directory / "talknet-normalized.avi"
        shutil.copy2(normalized, local_normalized)
        analysis["normalized_video"] = str(local_normalized)
        write_json(Path(config["output"]), analysis)
        if ambiguity:
            raise PipelineError("Rosto/locutor ambíguo. Corrija a seleção ou divisão de cenas; nenhum vídeo sem lip-sync confirmado será entregue.")
        if not confirmed:
            raise PipelineError("Nenhum trecho de rosto foi confirmado para lip-sync. Verifique se o vídeo mostra o rosto falante ou selecione o rosto na operação com áudio.")
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
        os.environ["LD_LIBRARY_PATH"] = str(muse_env / "lib/python3.10/site-packages/torch/lib")
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
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
        if config.get("background") and config.get("preserve_ambience"):
            output = legacy.mix_audio(Path(config["background"]), Path(config["audio"]), duration, directory, log)
            if output != Path(config["output"]):
                shutil.copy2(output, Path(config["output"]))
        else:
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", config["audio"], "-af", "apad,loudnorm=I=-14:TP=-1.5:LRA=11", "-t", str(duration), "-ac", "2", "-ar", "48000", config["output"]])
    elif name == "export":
        final = legacy.finalize(Path(config["video"]), Path(config["audio"]), Path(config["source"]), directory, float(config["duration"]), log)
        if final != Path(config["output"]):
            shutil.copy2(final, Path(config["output"]))
        if config.get("subtitle_mode") in {"burn", "burned"}:
            subtitle = Path(config["srt"]).resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
            temporary = directory / "subtitled.mp4"
            run_command(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(final), "-vf", f"subtitles='{subtitle}'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-c:a", "copy", str(temporary)])
            shutil.copy2(temporary, Path(config["output"]))
        write_json(Path(config["validation"]), verify_final_media(Path(config["output"]), float(config["duration"])))
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
    directory = work_root / manifest["job_id"]
    saved = storage / "jobs" / manifest["job_id"]
    copy_missing(saved, directory, verify=True)
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"job_id": manifest["job_id"], "input_sha256": manifest["input_sha256"], "pipeline_version": PIPELINE_VERSION}
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
    duration = media_info(video)["duration"] if video else 0.0
    if video and (duration <= 0 or duration > 1200):
        raise PipelineError("Vídeos precisam ter duração entre zero e vinte minutos.")
    approved_snapshot = load_approved_snapshot(package, manifest, duration)
    diarization = shared / "diarization.json"
    analysis = shared / "active-speaker.json"
    references = shared / "references.json"
    source_language = str(options.get("source_language", "auto"))
    started = time.time()
    if operation == "dub_video":
        cache.stage("separation", "Separação de áudio na GPU", {**base_config, "stage": "separation", "video": str(video)}, [shared / "vocals.wav", shared / "background.wav"])
        if approved_snapshot:
            frozen = approved_snapshot["source"]
            write_json(diarization, {"segments": frozen["segments"], "language": frozen["source_language"],
                                     "approved_source_snapshot_sha256": approved_snapshot["source_sha256"]})
            cache.used["approved_source_snapshot"] = True
            cache.seconds["approved_source_snapshot"] = 0.0
            print("Transcrição aprovada preservada: ASR e identificação de locutores não serão refeitas.", flush=True)
        else:
            cache.stage("asr", "Transcrição na GPU", {**base_config, "stage": "asr", "vocals": str(shared / "vocals.wav"), "source_language": source_language, "profile": options.get("performance_profile", "fast_lipsync"), "output": str(shared / "transcript.json")}, [shared / "transcript.json"])
            cache.stage("diarization", "Identificação de locutores", {**base_config, "stage": "diarization", "vocals": str(shared / "vocals.wav"), "transcript": str(shared / "transcript.json"), "transcript_hash": sha256(shared / "transcript.json"), "speaker_count": options.get("speaker_count", "auto"), "output": str(diarization)}, [diarization])
        source_language = read_json(diarization)["language"]
        reference_files = [shared / f"reference-{speaker}.wav" for speaker in sorted({item["speaker"] for item in read_json(diarization)["segments"]})]
        cache.stage("references", "Referências do timbre original", {**base_config, "stage": "references", "vocals": str(shared / "vocals.wav"), "segments": str(diarization), "segments_hash": sha256(diarization), "reference": str(inputs["reference"]) if "reference" in inputs else None, "reference_hash": sha256(inputs["reference"]) if "reference" in inputs else None, "output": str(references)}, [references, *reference_files])
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
        audio_duration = media_info(inputs["audio"])["duration"]
        if audio_duration <= 0 or audio_duration > duration + 1.0:
            raise PipelineError("O áudio enviado excede a duração do vídeo e seria cortado. Ajuste os arquivos antes de continuar.")
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
    if operation == "dub_video" and not require_lipsync:
        print("Dublagem sem lip-sync: análise facial, TalkNet e MuseTalk não serão executados.", flush=True)
    if video and require_lipsync and (operation != "dub_video" or translations):
        cache.stage("analysis", "Confirmação dos rostos e locutores", {**base_config, "stage": "analysis", "video": str(video), "audio": str(inputs["audio"]) if operation == "lipsync_video" else None, "duration": duration, "operation": operation, "segments": str(diarization), "segments_hash": sha256(diarization), "face_selection": options.get("face_selection", "auto"), "output": str(analysis)}, [analysis, shared / "talknet-normalized.avi"])
    records = []
    for language in manifest["target_languages"]:
        target = directory / "outputs" / language
        target.mkdir(parents=True, exist_ok=True)
        record = {"language": language, "status": "failed", "files": {}, "checksums": {}}
        report = {"schema_version": SCHEMA_VERSION, **identity, "operation": operation, "target_language": language, "source_language": source_language, "gpu": gpu, "voice_clone": {"required": operation != "lipsync_video", "confirmed": False, "authorized": True}, "lipsync": {"required": require_lipsync, "confirmed": False, "applied": False, "confirmed_segments": [], "ambiguity": []}, "preserved_no_speaker_visual_segments": [], "validation_state": "machine_checks_pending_human_review", "started_at": int(started)}
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
                if operation == "text_to_voice":
                    synthesis_outputs.append(target / "voice-segments.json")
                cache.stage(f"synthesis-{language}", f"Timbre original em {language}", {"directory": str(target), "stage": "synthesis", "identity": identity, "segments": str(translated), "segments_hash": sha256(translated), "references": refs, "reference_hashes": {key: sha256(Path(value)) for key, value in refs.items()}, "language": language, "duration": duration}, synthesis_outputs)
                report.update(read_json(target / "voice-proof.json"))
                audio = target / "dialogue.wav"
                if video:
                    legacy.write_srt(target / "original.srt", source_segments, "text")
                    legacy.write_srt(target / "translation.srt", read_json(translated)["segments"], "translated_text")
                report["translation"] = {"target_language": language, "segments_validated": len(source_segments), "engine": read_json(translated).get("translation_engine", "texto do usuário"), "reviewed": bool(approved)}
            else:
                audio = inputs["audio"]
                report["voice_engine"] = "áudio enviado pelo usuário"
                report["voice_clone"] = {"required": False, "confirmed": False, "authorized": True}
            files = {}
            if video:
                cache.stage(f"mix-{language}", f"Mixagem {language}", {"directory": str(target), "stage": "mix", "identity": identity, "audio": str(audio), "audio_hash": sha256(audio), "duration": duration, "background": str(shared / "background.wav") if operation == "dub_video" else None, "preserve_ambience": bool(options.get("preserve_ambience", True)), "output": str(target / "mix.wav")}, [target / "mix.wav"])
                synced = video
                if require_lipsync:
                    sync_audio = target / "mix.wav" if operation == "lipsync_video" else audio
                    cache.stage(f"lipsync-{language}", f"Lip-sync {language}", {"directory": str(target), "stage": "lipsync", "identity": identity, "video": str(video), "audio": str(sync_audio), "audio_hash": sha256(sync_audio), "analysis": str(analysis), "analysis_hash": sha256(analysis), "output": str(target / "synced.mp4"), "proof": str(target / "lipsync-proof.json")}, [target / "synced.mp4", target / "lipsync-proof.json"])
                    synced = target / "synced.mp4"
                    proof = read_json(target / "lipsync-proof.json")
                    confirmed = []
                    for face in proof.get("confirmed_face_segments", []):
                        confirmed.append({"start": face["start_frame"] / face["fps"], "end": face["end_frame"] / face["fps"], "speaker": face.get("speaker", "USER_AUDIO"), "coordinate_source": face.get("coordinate_source", "TalkNet confirmado")})
                    report["lipsync"] = {"required": True, "confirmed": bool(confirmed) and not proof.get("ambiguity"), "applied": bool(confirmed) and not proof.get("ambiguity"), "confirmed_segments": confirmed, "ambiguity": proof.get("ambiguity", []), "coordinate_source": proof.get("coordinate_source")}
                    report["preserved_no_speaker_visual_segments"] = proof.get("preserved_no_speaker_visual_segments", [])
                cache.stage(f"export-{language}", f"Exportação {language}", {"directory": str(target), "stage": "export", "identity": identity, "video": str(synced), "video_hash": sha256(synced), "audio": str(target / "mix.wav"), "audio_hash": sha256(target / "mix.wav"), "source": str(video), "duration": duration, "subtitle_mode": options.get("subtitle_mode", "none"), "srt": str(target / "translation.srt"), "output": str(target / "video.mp4"), "validation": str(target / "media-validation.json")}, [target / "video.mp4", target / "media-validation.json"])
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
            report.update(error=str(exc), finished_at=int(time.time()), cache=cache.used, stages_seconds=cache.seconds, gpu_activity=cache.gpu_activity)
            write_json(target / "report.json", report)
            record.update(error=str(exc), files={"report": (target / "report.json").relative_to(directory).as_posix()}, checksums={(target / "report.json").relative_to(directory).as_posix(): sha256(target / "report.json")})
            print(f"FALHA {language}: {exc}. Checkpoints preservados; os demais idiomas continuarão.", flush=True)
        records.append(record)
        cache.persist()
    result = {"schema_version": SCHEMA_VERSION, **identity, "operation": operation, "outputs": records}
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
            print(f"ERRO_ETAPA={exc}", flush=True)
            raise
    else:
        if not args.package_root or not args.storage_root or not args.output:
            parser.error("Informe --package-root, --storage-root e --output.")
        try:
            process(args.package_root, args.work_root, args.storage_root, args.output)
        except Exception as exc:
            manifest = validate_manifest(args.package_root)
            root = args.work_root / manifest["job_id"]
            records = []
            for language in manifest["target_languages"]:
                report = root / "outputs" / language / "report.json"
                write_json(report, {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "job_id": manifest["job_id"], "input_sha256": manifest["input_sha256"], "target_language": language, "error": str(exc), "status": "failed"})
                relative = report.relative_to(root).as_posix()
                records.append({"language": language, "status": "failed", "error": str(exc), "files": {"report": relative}, "checksums": {relative: sha256(report)}})
            bundle(root, args.output, "result.json", {"schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "job_id": manifest["job_id"], "input_sha256": manifest["input_sha256"], "outputs": records})
            print(f"ERRO_PIPELINE={exc}\nRESULTADO={args.output}", flush=True)
            raise


if __name__ == "__main__":
    main()
