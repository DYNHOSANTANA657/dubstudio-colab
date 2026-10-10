"""Orquestração local e adaptadores configuráveis do DubStudio.

O modo demo apenas analisa o arquivo; ele nunca declara lip-sync concluído.
O modo real exige comandos externos configurados explicitamente.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[int, str, str], None]


class PipelineError(RuntimeError):
    pass


def executable_status(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    return {"ready": bool(path), "path": path or ""}


def probe_video(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise PipelineError("FFprobe não foi encontrado. Instale o FFmpeg e reinicie o estúdio.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode:
        raise PipelineError("O arquivo não pôde ser analisado como vídeo válido.")
    payload = json.loads(result.stdout)
    video = next((s for s in payload.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in payload.get("streams", []) if s.get("codec_type") == "audio"), {})
    fmt = payload.get("format", {})
    return {
        "duration": round(float(fmt.get("duration") or 0), 3),
        "size": int(fmt.get("size") or path.stat().st_size),
        "format": fmt.get("format_name", ""),
        "video": video,
        "audio": audio,
    }


def safe_command_template(template: str, values: dict[str, str]) -> list[str]:
    """Expande um comando sem shell; cada placeholder vira um argumento literal."""
    if not template.strip():
        raise PipelineError("Um dos motores reais ainda não possui comando configurado.")
    try:
        parts = shlex.split(template, posix=os.name != "nt")
        normalized = [
            part[1:-1] if len(part) >= 2 and part[0] == part[-1] and part[0] in {'"', "'"} else part
            for part in parts
        ]
        return [part.format_map(values) for part in normalized]
    except (KeyError, ValueError) as exc:
        raise PipelineError(f"Comando de motor inválido: {exc}") from exc


def run_engine(
    label: str,
    template: str,
    values: dict[str, str],
    expected_output: Path,
    log_path: Path,
) -> None:
    command = safe_command_template(template, values)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{label}] {' '.join(command[:1])} [argumentos ocultados]\n")
        result = subprocess.run(command, stdout=log, stderr=log, cwd=values["workdir"])
    if result.returncode:
        raise PipelineError(f"O motor de {label} terminou com erro. Consulte o log técnico do trabalho.")
    if not expected_output.exists() or expected_output.stat().st_size == 0:
        raise PipelineError(f"O motor de {label} não criou o arquivo esperado: {expected_output.name}.")


@dataclass
class Job:
    id: str
    directory: Path
    input_path: Path
    options: dict[str, Any]
    status: str = "queued"
    progress: int = 0
    stage: str = "Na fila"
    message: str = "Preparando o trabalho"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: Path | None = None
    report_path: Path | None = None
    error: str = ""

    def public(self) -> dict[str, Any]:
        package = self.directory / "pacote-colab.zip"
        translated_srt = self.directory / "translation.srt"
        original_srt = self.directory / "transcript-original.srt"
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "options": self.options,
            "error": self.error,
            "has_output": bool(self.output_path and self.output_path.exists()),
            "has_report": bool(self.report_path and self.report_path.exists()),
            "has_colab_package": package.exists() and package.stat().st_size > 0,
            "has_original_srt": original_srt.exists(),
            "has_translation_srt": translated_srt.exists(),
        }


class JobStore:
    def __init__(self) -> None:
        self._items: dict[str, Job] = {}
        self._lock = threading.Lock()

    def add(self, job: Job) -> None:
        with self._lock:
            self._items[job.id] = job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._items.get(job_id)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.public() for job in sorted(self._items.values(), key=lambda item: item.created_at, reverse=True)]

    def remove(self, job_id: str) -> Job | None:
        with self._lock:
            return self._items.pop(job_id, None)


def engine_readiness(config: dict[str, Any]) -> dict[str, Any]:
    commands = config.get("commands", {})
    ffmpeg = executable_status("ffmpeg")
    ffprobe = executable_status("ffprobe")
    stages = {
        key: {"ready": bool(commands.get(key, "").strip()), "type": "command"}
        for key in ("asr", "translate", "tts", "lipsync")
    }
    return {
        "ffmpeg": ffmpeg,
        "ffprobe": ffprobe,
        "colab_notebook_ready": (Path(__file__).resolve().parent / "DubStudio_Colab.ipynb").exists(),
        "stages": stages,
        "real_pipeline_ready": ffmpeg["ready"] and ffprobe["ready"] and all(item["ready"] for item in stages.values()),
    }


def process_job(job: Job, config: dict[str, Any], persist: Callable[[Job], None]) -> None:
    def update(progress: int, stage: str, message: str) -> None:
        job.progress = progress
        job.stage = stage
        job.message = message
        job.updated_at = time.time()
        persist(job)

    try:
        job.status = "running"
        update(5, "Validação", "Conferindo vídeo, áudio e duração")
        job.metadata = probe_video(job.input_path)
        if not job.metadata.get("audio"):
            raise PipelineError("O vídeo não possui uma faixa de áudio para dublagem.")

        if job.options.get("mode") == "demo":
            update(45, "Análise", "Mapeando o fluxo e validando as escolhas")
            time.sleep(0.35)
            report = {
                "notice": "MODO DEMONSTRAÇÃO: nenhum áudio foi gerado e nenhum lip-sync foi aplicado.",
                "input": job.input_path.name,
                "metadata": job.metadata,
                "options": job.options,
                "required_for_real_export": ["asr", "translate", "tts", "lipsync"],
            }
            job.report_path = job.directory / "relatorio-demonstracao.json"
            job.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            job.status = "demo_ready"
            update(100, "Demonstração pronta", "Análise concluída; o vídeo não foi alterado")
            return

        if job.options.get("mode") == "colab":
            from colab_bridge import prepare_colab_package

            job.status = "preparing_colab"
            update(18, "Preparando Colab", "Analisando mídia e criando o pacote offline")
            prepare_colab_package(job)
            job.status = "awaiting_colab"
            update(30, "Aguardando Colab", "Pacote pronto: execute o notebook manualmente no Colab")
            return

        readiness = engine_readiness(config)
        if not readiness["real_pipeline_ready"]:
            raise PipelineError("Pipeline real incompleto. Configure os quatro motores em Configurações.")

        workdir = job.directory
        audio = workdir / "audio.wav"
        transcript = workdir / "transcript.json"
        translated = workdir / "translated.json"
        dub_audio = workdir / "dub.wav"
        output = workdir / "dublagem-final.mp4"
        log_path = workdir / "pipeline.log"
        ffmpeg = shutil.which("ffmpeg")
        assert ffmpeg
        update(12, "Áudio", "Extraindo áudio com qualidade para reconhecimento")
        extraction = subprocess.run(
            [ffmpeg, "-y", "-i", str(job.input_path), "-vn", "-ac", "1", "-ar", "16000", str(audio)],
            capture_output=True,
        )
        if extraction.returncode or not audio.exists():
            raise PipelineError("Não foi possível extrair o áudio do vídeo.")

        values = {
            "input_video": str(job.input_path),
            "audio": str(audio),
            "transcript_json": str(transcript),
            "translated_json": str(translated),
            "dub_audio": str(dub_audio),
            "output_video": str(output),
            "workdir": str(workdir),
            "language": str(job.options.get("language", "pl-PL")),
            "source_language": str(job.options.get("source_language", "auto")),
            "speaker_count": str(job.options.get("speaker_count", "auto")),
            "preset": str(job.options.get("preset", "natural")),
        }
        commands = config["commands"]
        update(24, "Transcrição", "Reconhecendo fala e tempos")
        run_engine("transcrição", commands["asr"], values, transcript, log_path)
        update(43, "Tradução", "Adaptando o texto ao tempo das falas")
        run_engine("tradução", commands["translate"], values, translated, log_path)
        update(62, "Vozes", "Gerando vozes genéricas por falante")
        run_engine("síntese de voz", commands["tts"], values, dub_audio, log_path)
        update(79, "Lip-sync", "Sincronizando movimentos labiais")
        run_engine("lip-sync", commands["lipsync"], values, output, log_path)
        update(94, "Qualidade", "Verificando o arquivo exportado")
        output_metadata = probe_video(output)
        if abs(output_metadata["duration"] - job.metadata["duration"]) > 1.5:
            raise PipelineError("A duração exportada ficou fora da tolerância de 1,5 segundo.")
        job.output_path = output
        job.metadata["output"] = output_metadata
        job.status = "complete"
        update(100, "Concluído", "Dublagem pronta para prévia e download")
    except Exception as exc:  # registra erro limpo para a interface
        job.status = "failed"
        job.error = str(exc)
        update(job.progress, "Atenção necessária", str(exc))

