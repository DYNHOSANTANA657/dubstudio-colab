"""Pacotes offline entre o DubStudio local e um notebook executado manualmente."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from pipeline import Job, PipelineError, probe_video


ROOT = Path(__file__).resolve().parent
NOTEBOOK_PATH = ROOT / "DubStudio_Colab.ipynb"
WORKER_PATH = ROOT / "colab" / "dubstudio_colab_worker.py"
TRANSLATOR_PATH = ROOT / "colab" / "dubstudio_translate.py"
SCHEMA_VERSION = "1.0"
PACKAGE_NAME = "pacote-colab.zip"
RESULT_NAME = "resultado-colab.zip"
FINAL_VIDEO = "dublagem-final.mp4"
FINAL_REPORT = "report.json"
MAX_DURATION_SECONDS = 20 * 60
RESULT_OPTIONAL_FILES = {
    "segments.json",
    "transcript-original.srt",
    "translation.srt",
    "pipeline.log",
}

# Os três primeiros artefatos são evidências do resultado, não simples
# conveniências para download.  Sem eles não há como o computador local
# conferir que o vídeo recebido corresponde à tradução e ao lip-sync que o
# worker informou ter executado.
REQUIRED_EVIDENCE_FILES = {
    "segments.json",
    "transcript-original.srt",
    "translation.srt",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def target_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise PipelineError("Não foi possível determinar a resolução do vídeo.")
    max_width, max_height = ((1920, 1080) if width >= height else (1080, 1920))
    ratio = min(1.0, max_width / width, max_height / height)
    out_width = max(2, int(width * ratio) // 2 * 2)
    out_height = max(2, int(height * ratio) // 2 * 2)
    return out_width, out_height


def requires_authorized_clone(options: dict[str, Any]) -> bool:
    """Mantém o contrato de clonagem também para trabalhos salvos antes do campo explícito."""
    if bool(options.get("require_authorized_clone")):
        return True
    return (
        _text(options.get("mode")) in {"colab", "runpod"}
        and bool(options.get("voice_clone_consent"))
    )


def _package_source(job: Job, metadata: dict[str, Any]) -> tuple[Path, bool]:
    video = metadata.get("video") or {}
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    target_width, target_height = target_dimensions(width, height)
    if (width, height) == (target_width, target_height):
        return job.input_path, False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PipelineError("O vídeo excede 1080p e o FFmpeg não está disponível para criar o proxy.")
    proxy = job.directory / "proxy-1080p.mp4"
    command = [
        ffmpeg, "-y", "-i", str(job.input_path), "-vf", f"scale={target_width}:{target_height}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(proxy),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode or not proxy.exists() or proxy.stat().st_size == 0:
        raise PipelineError("Não foi possível gerar o proxy 1080p para o Colab.")
    return proxy, True


def build_manifest(job: Job, source: Path, metadata: dict[str, Any], proxied: bool) -> dict[str, Any]:
    options = job.options
    return {
        "schema_version": SCHEMA_VERSION,
        "job_id": job.id,
        "created_at": int(time.time()),
        "input_file": f"input/{source.name}",
        "original_name": options.get("original_name", job.input_path.name),
        "input_sha256": sha256_file(source),
        "source_language": options.get("source_language", "auto"),
        "target_language": options.get("language", "pl-PL"),
        "preset": options.get("preset", "natural"),
        "performance_profile": options.get("performance_profile", "fast_lipsync"),
        "voice_mode": options.get("voice_mode", "clone_fallback"),
        "require_authorized_clone": requires_authorized_clone(options),
        "review_mode": options.get("review_mode", "automatic_optional"),
        "require_lipsync": bool(options.get("require_lipsync", True)),
        "speaker_mode": options.get("speaker_mode", "auto"),
        "speaker_count": options.get("speaker_count", "auto"),
        "preserve_ambience": bool(options.get("preserve_ambience", True)),
        "subtitle_policy": "preserve_existing",
        "max_resolution": "1080p",
        "proxy_created": proxied,
        "authorization": {
            "voice_clone_confirmed": bool(options.get("voice_clone_consent")),
            "scope": "references extracted only from the submitted video",
        },
        "input_metadata": metadata,
        "output_contract": {
            "required": [FINAL_VIDEO, FINAL_REPORT],
            "optional": sorted(RESULT_OPTIONAL_FILES),
            "duration_tolerance_seconds": 1.0,
        },
    }


def prepare_colab_package(job: Job) -> Path:
    if not job.options.get("voice_clone_consent"):
        raise PipelineError("Confirme a autorização das pessoas gravadas antes de preparar a clonagem vocal.")
    metadata = job.metadata or probe_video(job.input_path)
    duration = float(metadata.get("duration") or 0)
    if duration <= 0 or duration > MAX_DURATION_SECONDS:
        raise PipelineError("O modo Colab aceita vídeos com duração máxima de 20 minutos.")
    source, proxied = _package_source(job, metadata)
    if not WORKER_PATH.exists() or not TRANSLATOR_PATH.exists() or not NOTEBOOK_PATH.exists():
        raise PipelineError("Os arquivos do notebook Colab não foram encontrados nesta instalação.")
    manifest = build_manifest(job, source, metadata, proxied)
    manifest_path = job.directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    package_path = job.directory / PACKAGE_NAME
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(manifest_path, "manifest.json")
        archive.write(source, manifest["input_file"])
        archive.write(WORKER_PATH, "worker/dubstudio_colab_worker.py")
        archive.write(TRANSLATOR_PATH, "worker/dubstudio_translate.py")
    if package_path.stat().st_size == 0:
        raise PipelineError("O pacote Colab ficou vazio.")
    job.metadata["colab"] = {
        "schema_version": SCHEMA_VERSION,
        "package_bytes": package_path.stat().st_size,
        "input_sha256": manifest["input_sha256"],
        "proxy_created": proxied,
    }
    return package_path


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    allowed = {FINAL_VIDEO, FINAL_REPORT, *RESULT_OPTIONAL_FILES}
    members: list[zipfile.ZipInfo] = []
    seen: set[str] = set()
    total_uncompressed = 0
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir():
            continue
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise PipelineError("O ZIP de resultado contém um caminho inseguro.")
        if path.name not in allowed:
            continue
        if path.name in seen:
            raise PipelineError("O ZIP de resultado contém nomes de arquivo duplicados.")
        seen.add(path.name)
        total_uncompressed += member.file_size
        if total_uncompressed > 12 * 1024 * 1024 * 1024:
            raise PipelineError("O conteúdo descompactado do resultado excede o limite seguro.")
        members.append(member)
    names = {PurePosixPath(item.filename).name for item in members}
    missing = {FINAL_VIDEO, FINAL_REPORT} - names
    if missing:
        raise PipelineError(f"Resultado incompleto: faltando {', '.join(sorted(missing))}.")
    return members


def _text(value: object) -> str:
    return str(value or "").strip()


def _validate_report_contract(report: dict[str, Any], job: Job, source_duration: float) -> None:
    """Confere as provas declaradas pelo worker antes de liberar o arquivo.

    O relatório não é tratado como mera mensagem de sucesso.  Ele precisa
    confirmar o mesmo contrato que o usuário escolheu no Studio: idioma alvo,
    clonagem autorizada, rosto falante seguro e lip-sync sem trechos pulados.
    A mídia e os arquivos de evidência são checados separadamente depois.
    """
    target_language = _text(job.options.get("language", "pl-PL"))
    if _text(report.get("target_language")) != target_language:
        raise PipelineError("O relatório não confirma o idioma alvo solicitado para a dublagem.")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise PipelineError("O relatório não contém o resumo técnico obrigatório do resultado.")
    try:
        reported_duration = float(summary.get("duration"))
        reported_tolerance = float(summary.get("timing_tolerance_seconds"))
        channels = int(summary.get("audio_channels"))
    except (TypeError, ValueError):
        raise PipelineError("O relatório possui métricas de áudio ou duração inválidas.") from None
    if abs(reported_duration - source_duration) > 1.0 or reported_tolerance > 1.0:
        raise PipelineError("O relatório não confirma a duração dentro da tolerância de 1 segundo.")
    if channels < 1:
        raise PipelineError("O relatório não confirma uma faixa de áudio válida no vídeo final.")

    if requires_authorized_clone(job.options):
        voice_engine = _text(report.get("voice_engine"))
        voice_clone = report.get("voice_clone")
        if (
            not voice_engine.startswith("Chatterbox Multilingual V3 authorized clone")
            or not isinstance(voice_clone, dict)
            or voice_clone.get("required") is not True
            or voice_clone.get("confirmed") is not True
        ):
            raise PipelineError("O resultado não confirma a clonagem de voz autorizada exigida para este trabalho.")

    if not bool(job.options.get("require_lipsync", True)):
        return
    lipsync = report.get("lipsync")
    active_speaker = report.get("active_speaker")
    skipped = report.get("no_lipsync_segments")
    if not isinstance(lipsync, dict) or lipsync.get("required") is not True or lipsync.get("confirmed") is not True:
        raise PipelineError("O relatório não confirma que o lip-sync obrigatório foi concluído.")
    if not isinstance(active_speaker, dict) or active_speaker.get("safe_for_single_face_lipsync") is not True:
        raise PipelineError("O relatório não confirma um rosto falante seguro para o lip-sync.")
    if not isinstance(skipped, list) or skipped:
        raise PipelineError("O resultado não recebeu lip-sync em todos os trechos; ele não será liberado como dublagem final.")


def _validate_translation_evidence(root: Path, target_language: str) -> None:
    try:
        payload = json.loads((root / "segments.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PipelineError("A evidência de segmentos traduzidos não é válida.") from exc
    if _text(payload.get("target_language")) != target_language:
        raise PipelineError("A evidência de segmentos não corresponde ao idioma alvo solicitado.")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise PipelineError("A evidência de tradução não contém trechos de fala.")
    for index, segment in enumerate(segments, start=1):
        if not isinstance(segment, dict) or not _text(segment.get("text")) or not _text(segment.get("translated_text")):
            raise PipelineError(f"O trecho {index} não possui texto original e tradução válidos.")
        try:
            start, end = float(segment.get("start")), float(segment.get("end"))
        except (TypeError, ValueError):
            raise PipelineError(f"O trecho {index} possui tempos inválidos.") from None
        if start < 0 or end <= start:
            raise PipelineError(f"O trecho {index} possui intervalo de fala inválido.")
    for name in ("transcript-original.srt", "translation.srt"):
        try:
            content = (root / name).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            raise PipelineError(f"A evidência {name} não pôde ser lida.") from exc
        if not content:
            raise PipelineError(f"A evidência {name} está vazia.")


def _audio_stream_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise PipelineError("FFprobe não foi encontrado para validar a faixa de áudio final.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise PipelineError("Não foi possível medir a duração da faixa de áudio final.")
    try:
        return float(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise PipelineError("A faixa de áudio final não possui duração mensurável.") from None


def _video_stream_duration(path: Path) -> float:
    """Mede ``v:0`` para impedir áudio longo com imagem congelada no final."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise PipelineError("FFprobe não foi encontrado para validar a faixa de vídeo final.")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise PipelineError("Não foi possível medir a duração da faixa de vídeo final.")
    try:
        return float(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise PipelineError("A faixa de vídeo final não possui duração mensurável.") from None


def _tail_max_volume(path: Path, seconds: float) -> float:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PipelineError("FFmpeg não foi encontrado para validar o áudio no final do vídeo.")
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-sseof", f"-{max(0.25, seconds):.3f}", "-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode:
        raise PipelineError("Não foi possível medir o áudio no final do vídeo.")
    match = re.search(r"max_volume:\s*(-?(?:\d+(?:\.\d+)?)|-inf)\s*dB", result.stderr)
    if not match:
        raise PipelineError("O medidor de áudio não retornou o volume final do vídeo.")
    return float("-1000") if match.group(1) == "-inf" else float(match.group(1))


def _validate_audio_coverage(path: Path, video_duration: float, source_path: Path) -> None:
    audio_duration = _audio_stream_duration(path)
    # AAC pode encerrar alguns milissegundos antes do vídeo por padding do
    # codificador. Meio segundo cobre esse comportamento, mas não deixa passar
    # um arquivo cuja voz/áudio tenha sido cortado no final.
    if audio_duration < video_duration - 0.5:
        raise PipelineError("A faixa de áudio final termina antes do vídeo; o resultado não será liberado.")
    tail_seconds = min(5.0, max(0.5, video_duration * 0.1))
    source_peak = _tail_max_volume(source_path, tail_seconds)
    output_peak = _tail_max_volume(path, tail_seconds)
    # Só bloqueamos silêncio no resultado quando havia áudio audível no mesmo
    # fim do original. Assim, pausas naturais do próprio vídeo continuam
    # válidas, mas uma faixa que some antes do final não é entregue.
    if source_peak > -45.0 and output_peak < -55.0:
        raise PipelineError("O áudio do resultado ficou silencioso no final, apesar de haver áudio no original; ele não será liberado.")


def _validate_video_coverage(path: Path, source_duration: float, container_duration: float) -> None:
    """Exige que a imagem alcance o mesmo fim já validado para o áudio."""
    video_duration = _video_stream_duration(path)
    expected = min(source_duration, container_duration)
    # A duração de vídeo pode ter um pequeno arredondamento pelo FPS/MP4, mas
    # nunca deve faltar dezenas de segundos enquanto o áudio continua.
    if video_duration < expected - 0.5:
        raise PipelineError("A faixa de vídeo final termina antes do áudio/original; o resultado não será liberado.")


def import_colab_result(job: Job, result_zip: Path) -> dict[str, Any]:
    if not zipfile.is_zipfile(result_zip):
        raise PipelineError("O arquivo importado não é um ZIP válido.")
    source_duration = float((job.metadata or {}).get("duration") or probe_video(job.input_path)["duration"])
    with tempfile.TemporaryDirectory(prefix="dubstudio-import-", dir=job.directory) as temp_name:
        temp = Path(temp_name)
        with zipfile.ZipFile(result_zip) as archive:
            members = _safe_members(archive)
            report_member = next(item for item in members if PurePosixPath(item.filename).name == FINAL_REPORT)
            if report_member.file_size > 4 * 1024 * 1024:
                raise PipelineError("O report.json excede o limite de 4 MB.")
            try:
                report = json.loads(archive.read(report_member).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise PipelineError("O report.json do Colab não é válido.") from exc
            if str(report.get("job_id")) != job.id:
                raise PipelineError("O resultado pertence a outro trabalho.")
            if str(report.get("schema_version")) != SCHEMA_VERSION:
                raise PipelineError("A versão do resultado não é compatível com este DubStudio.")
            _validate_report_contract(report, job, source_duration)
            names = {PurePosixPath(item.filename).name for item in members}
            missing_evidence = REQUIRED_EVIDENCE_FILES - names
            if missing_evidence:
                raise PipelineError(f"Resultado sem evidências obrigatórias: {', '.join(sorted(missing_evidence))}.")
            for member in members:
                name = PurePosixPath(member.filename).name
                with archive.open(member) as source, (temp / name).open("wb") as target:
                    shutil.copyfileobj(source, target, 4 * 1024 * 1024)
        _validate_translation_evidence(temp, _text(job.options.get("language", "pl-PL")))
        report_path = temp / FINAL_REPORT
        output = temp / FINAL_VIDEO
        output_metadata = probe_video(output)
        if not output_metadata.get("video") or not output_metadata.get("audio"):
            raise PipelineError("O vídeo final precisa conter imagem e áudio.")
        if abs(float(output_metadata["duration"]) - source_duration) > 1.0:
            raise PipelineError("A duração do resultado ficou fora da tolerância de 1 segundo.")
        _validate_video_coverage(output, source_duration, float(output_metadata["duration"]))
        _validate_audio_coverage(output, float(output_metadata["duration"]), job.input_path)
        stream = output_metadata["video"]
        width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        max_width, max_height = ((1920, 1080) if width >= height else (1080, 1920))
        if width > max_width or height > max_height:
            raise PipelineError("O resultado excede a resolução máxima de 1080p.")
        final_output = job.directory / FINAL_VIDEO
        final_report = job.directory / FINAL_REPORT
        shutil.copy2(output, final_output)
        shutil.copy2(report_path, final_report)
        imported = []
        for name in RESULT_OPTIONAL_FILES:
            source = temp / name
            if source.exists():
                shutil.copy2(source, job.directory / name)
                imported.append(name)
    job.output_path = final_output
    job.report_path = final_report
    job.metadata["output"] = output_metadata
    job.metadata["colab_result"] = {
        "imported_files": imported,
        "report_summary": report.get("summary", {}),
        # Só é incluído após todas as verificações locais acima passarem. A
        # interface o usa como uma prova de liberação, não como mensagem do Pod.
        "validation": {
            "validated_at": int(time.time()),
            "target_language": _text(job.options.get("language", "pl-PL")),
            "translation_evidence": True,
            "authorized_clone": requires_authorized_clone(job.options),
            "lipsync_confirmed": bool(job.options.get("require_lipsync", True)),
            "audio_coverage": True,
            "duration_tolerance_seconds": 1.0,
        },
    }
    return report
