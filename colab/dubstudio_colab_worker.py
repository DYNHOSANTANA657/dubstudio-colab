#!/usr/bin/env python3
"""Worker reproduzível do DubStudio para execução manual no Google Colab.

O script nunca abre túnel, não automatiza login e não tenta impedir o encerramento
do runtime. Cada estágio confirmado é salvo no Drive e pode ser retomado.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
import codecs
from fractions import Fraction
from pathlib import Path
from typing import Callable


SCHEMA_VERSION = "1.0"
# v13 também separa, de forma explícita, cada janela realmente confirmada de
# rosto falante. Em vídeo que muda de cena, só essa janela recebe MuseTalk;
# os demais quadros originais são preservados, nunca recebem uma caixa facial
# inventada e nunca são aceitos como vídeo congelado.
LIPSYNC_CACHE_REVISION = "musetalk-talknet-multisegment-v21-source-landmarks-perioral"
CANONICAL_FACE_FPS = 25
TALKNET_NORMALIZATION_REVISION = "canonical-pts-fps25-pcm-v1"
TALKNET_TIMELINE_TOLERANCE_SECONDS = 0.15
CHATTERBOX_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
    "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh",
}
EDGE_PREFERRED = {
    "pt-BR": "pt-BR-FranciscaNeural", "pt-PT": "pt-PT-RaquelNeural",
    "pl-PL": "pl-PL-ZofiaNeural", "en-US": "en-US-JennyNeural",
    "en-GB": "en-GB-SoniaNeural", "es-ES": "es-ES-ElviraNeural",
    "es-MX": "es-MX-DaliaNeural", "fr-FR": "fr-FR-DeniseNeural",
    "de-DE": "de-DE-KatjaNeural", "it-IT": "it-IT-ElsaNeural",
}


class WorkerError(RuntimeError):
    pass


def run(command: list[str], cwd: Path | None = None, log: Path | None = None) -> subprocess.CompletedProcess:
    print("$", " ".join(str(item) for item in command[:3]), "…", flush=True)
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ {' '.join(command)}\n{result.stdout}\n{result.stderr}\n")
    if result.returncode:
        tail = (result.stderr or result.stdout)[-1800:]
        raise WorkerError(f"Comando falhou ({result.returncode}): {tail}")
    return result


def run_streaming(
    command: list[str],
    cwd: Path | None = None,
    log: Path | None = None,
    on_output: Callable[[str], None] | None = None,
) -> subprocess.CompletedProcess:
    """Executa um processo longo e propaga suas etapas sem inventar progresso.

    O MuseTalk escreve a evolução das etapas no stdout/stderr. Ao ler o fluxo
    combinado durante a execução, a interface recebe o estado realmente
    informado pelo renderizador, em vez de permanecer parada em 82%.
    """
    print("$", " ".join(str(item) for item in command[:3]), "…", flush=True)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # O tqdm atualiza a mesma linha com ``\r``. Em modo texto/readline
        # esses eventos ficam presos até a etapa terminar, fazendo a interface
        # parecer congelada mesmo com GPU ativa.
        text=False,
        bufsize=0,
    )
    chunks: list[str] = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    with (log.open("a", encoding="utf-8") if log else open(os.devnull, "w", encoding="utf-8")) as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        assert process.stdout is not None
        while True:
            block = process.stdout.read(4096)
            if not block:
                break
            text = decoder.decode(block)
            chunks.append(text)
            handle.write(text)
            handle.flush()
            if on_output:
                pending += text
                parts = re.split(r"[\r\n]", pending)
                pending = parts.pop()
                for line in parts:
                    if line:
                        on_output(line)
        tail = decoder.decode(b"", final=True)
        if tail:
            chunks.append(tail)
            handle.write(tail)
            pending += tail
        if on_output and pending:
            on_output(pending)
        process.stdout.close()
    returncode = process.wait()
    result = subprocess.CompletedProcess(command, returncode, "".join(chunks), "")
    if result.returncode:
        raise WorkerError(f"Comando falhou ({result.returncode}): {result.stdout[-3000:]}")
    if "scripts.inference" in command:
        failure = musetalk_failure_detail(result.stdout)
        if failure:
            # MuseTalk catches task exceptions and can still exit zero. Raise
            # here, inside the runner's low-memory wrapper, to allow batch 1.
            raise WorkerError(f"MuseTalk declarou falha apesar de retornar zero: {failure}")
    return result


def musetalk_failure_detail(output: str) -> str:
    failure_lines = [
        line.strip() for line in output.splitlines()
        if any(marker in line.lower() for marker in (
            "error occurred during processing:", "traceback (most recent call last):", "out of memory",
        ))
    ]
    if not failure_lines:
        return ""
    tail = output[-3000:].strip()
    cause = "\n".join(line for line in failure_lines[-3:] if line not in tail)
    return "\n".join(part for part in (cause, tail) if part)


def select_musetalk_output(
    result_dir: Path,
    before: set[Path],
    result: subprocess.CompletedProcess,
    failure_message: str,
) -> Path:
    """Do not treat MuseTalk's swallowed task exception or old MP4 as success."""
    output = getattr(result, "stdout", "") or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    elif not isinstance(output, str):
        output = str(output)
    # The pinned official CLI catches task exceptions, prints them, and exits
    # zero. Preserve those diagnostics so the caller can retry GPU OOM safely.
    failure = musetalk_failure_detail(output)
    candidates = [
        path for path in result_dir.rglob("*.mp4")
        if path.resolve() not in before and path.is_file() and path.stat().st_size > 0
    ]
    if not candidates or failure or getattr(result, "returncode", 0):
        detail = failure or output[-3000:].strip()
        raise WorkerError(f"{failure_message}{' Saída do MuseTalk: ' + detail if detail else ''}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict:
    result = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,format_name:stream=codec_type,duration,width,height,channels,sample_rate,r_frame_rate,avg_frame_rate", "-of", "json", str(path),
    ])
    payload = json.loads(result.stdout)
    fmt = payload.get("format", {})
    payload["_media_path"] = str(path)
    kind = "video" if any(item.get("codec_type") == "video" for item in payload.get("streams", [])) else "audio"
    try:
        duration = _measured_input_duration(payload, kind)
    except RuntimeError as exc:
        raise WorkerError(str(exc)) from exc
    selected = next(item for item in payload["streams"] if item.get("codec_type") == kind)
    selected["duration"] = duration
    return {"duration": duration, "container_duration": float(fmt.get("duration") or 0), "format": fmt, "streams": payload.get("streams", [])}


def _measured_input_duration(info: dict, kind: str) -> float:
    # Both the standalone worker and local Studio import the same public helper;
    # it imports no models and adds no new package/module to notebooks.
    try:
        from .dubstudio_runner_v2 import input_stream_duration
    except ImportError:
        from dubstudio_runner_v2 import input_stream_duration
    return input_stream_duration(info, kind)


def video_frame_count(path: Path) -> int:
    """Obtém a contagem real de frames sem carregar o vídeo em memória."""
    result = run([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ])
    value = result.stdout.strip().splitlines()
    try:
        return int(value[-1]) if value and value[-1].isdigit() else 0
    except (TypeError, ValueError):
        return 0


def video_stream_duration(path: Path) -> float:
    """Lê a duração da faixa de vídeo, não a duração do contêiner.

    Um MP4 pode ter áudio longo e vídeo curto.  Nesse caso o ``format`` do
    FFprobe parece correto e a última imagem fica congelada.  A aprovação do
    lip-sync precisa medir especificamente ``v:0``.
    """
    info = ffprobe(path)
    try:
        return _measured_input_duration({**info, "_media_path": str(path)}, "video")
    except RuntimeError as exc:
        raise WorkerError(str(exc)) from exc


def assert_video_stream_coverage(rendered: Path, source: Path, tolerance: float = 0.5) -> None:
    """Recusa vídeo que termine antes do original, mesmo que o áudio continue."""
    expected = video_stream_duration(source)
    actual = video_stream_duration(rendered)
    if expected <= 0 or actual <= 0:
        raise WorkerError("Não foi possível validar a duração da imagem no lip-sync.")
    if actual < expected - tolerance:
        raise WorkerError(
            "O renderizador de lip-sync gerou imagem por apenas "
            f"{actual:.3f}s de {expected:.3f}s. O áudio não será combinado com vídeo incompleto."
        )


def confirmed_face_minimum_frames(segment: dict, default: int = 20) -> int:
    """Validate a narrow user-selected exception; summaries alone are no proof.

    Short windows contain the original detected boxes and per-frame scores.
    The worker recalculates every count, ratio and digest before relaxing its
    minimum. This is a consistency contract, not a cryptographic attestation
    that a human reviewed MuseTalk's visual output.
    """
    if not isinstance(segment, dict):
        raise WorkerError("O trecho facial não possui uma estrutura válida.")
    proof = segment.get("explicit_selected_short_track")
    if proof is None:
        return default
    message = "A prova do trecho facial curto selecionado é inválida."
    try:
        start, end = segment["start_frame"], segment["end_frame"]
        boxes, track_id = segment["boxes"], segment["track_id"]
        count = end - start
        if (
            not isinstance(proof, dict)
            or type(start) is not int or type(end) is not int
            or type(track_id) is not int or track_id < 0
            or not isinstance(boxes, list) or len(boxes) != count
            or not 12 <= count < 20
            or type(segment["fps"]) not in {int, float} or float(segment["fps"]) != 25.0
            or segment.get("face_confirmation") != "user_selected"
            or segment.get("selection_source") not in {"track_id", "bbox"}
            or proof.get("selection_source") != segment["selection_source"]
            or type(proof.get("schema_version")) is not int or proof.get("schema_version") != 1
            or proof.get("policy") != "explicit-user-selected-short-track-v1"
            or type(proof.get("minimum_frames")) is not int or proof.get("minimum_frames") != 12
            or any(type(proof.get(key)) is not int for key in ("track_id", "start_frame", "end_frame", "frame_count"))
            or proof.get("track_id") != track_id
            or proof.get("start_frame") != start or proof.get("end_frame") != end
            or proof.get("frame_count") != count or proof.get("fps") != 25.0
            or any(not isinstance(box, list) or len(box) != 4
                   or any(type(value) is not int for value in box)
                   or min(box) < 0 or box[2] - box[0] < 24 or box[3] - box[1] < 24 for box in boxes)
        ):
            raise WorkerError(message)
        digest = hashlib.sha256(json.dumps(boxes, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        if proof.get("boxes_sha256") != digest:
            raise WorkerError(message)
        operation = proof.get("operation")
        if operation == "lipsync_video":
            if (
                segment.get("speaker_confirmation") != "external_audio"
                or proof.get("score_evidence_required") is not False
                or proof.get("scores") != []
                or any(proof.get(key) is not None for key in ("finite_score_count", "positive_score_count", "positive_score_ratio", "peak_score"))
            ):
                raise WorkerError(message)
        elif operation == "dub_video":
            values = proof.get("scores")
            if (
                segment.get("speaker_confirmation") != "user_selected"
                or proof.get("score_evidence_required") is not True
                or not isinstance(values, list) or len(values) != count
                or any(type(value) not in {int, float} or not math.isfinite(value) for value in values)
            ):
                raise WorkerError(message)
            positive, peak = sum(value > 0 for value in values), max(values)
            if (
                positive * 5 < count * 4 or peak < 0.8
                or type(proof.get("finite_score_count")) is not int or proof.get("finite_score_count") != count
                or type(proof.get("positive_score_count")) is not int or proof.get("positive_score_count") != positive
                or type(proof.get("positive_score_ratio")) not in {int, float}
                or not math.isclose(proof["positive_score_ratio"], positive / count, rel_tol=0, abs_tol=1e-12)
                or type(proof.get("peak_score")) not in {int, float} or proof.get("peak_score") != peak
            ):
                raise WorkerError(message)
        else:
            raise WorkerError(message)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise WorkerError(message) from None
    return 12


def bound_confirmed_talknet_segments(
    raw_segments: list[dict],
    source_duration: float,
    normalized_duration: float,
    minimum_frames: int = 20,
) -> tuple[list[dict], dict]:
    """Bind detected boxes to the real source timeline before rendering/proof.

    TalkNet's 25-fps AVI can include a few repeated frames after the original
    video ends (503 frames versus a 20-second source, for example). Only that
    small normalization tail may be removed. We never stretch the timestamps,
    fabricate coordinates, omit an ambiguous window, or repair a large drift.
    Complete detected frames are kept; a fractional final frame is preserved
    from the original instead of claiming facial evidence past its duration.
    """
    if (
        not math.isfinite(source_duration) or not math.isfinite(normalized_duration)
        or min(source_duration, normalized_duration) <= 0
        or abs(normalized_duration - source_duration) > TALKNET_TIMELINE_TOLERANCE_SECONDS + 1e-6
    ):
        raise WorkerError("A duração normalizada do TalkNet não corresponde à faixa de vídeo original.")
    if not isinstance(raw_segments, list) or not raw_segments or minimum_frames < 1:
        raise WorkerError("TalkNet não forneceu trechos faciais confirmados para sincronizar.")
    bounded = []
    clipped = []
    previous_end = 0.0
    for index, raw in enumerate(raw_segments, start=1):
        segment_minimum = confirmed_face_minimum_frames(raw, minimum_frames)
        try:
            start_frame, end_frame = int(raw["start_frame"]), int(raw["end_frame"])
            fps = float(raw["fps"])
            boxes = list(raw["boxes"])
            integral_frames = float(raw["start_frame"]) == start_frame and float(raw["end_frame"]) == end_frame
        except (KeyError, TypeError, ValueError, OverflowError):
            raise WorkerError(f"O trecho facial {index} não possui coordenadas válidas.") from None
        if (
            not integral_frames or start_frame < 0 or not math.isfinite(fps)
            or not 1 <= fps <= 120 or len(boxes) != end_frame - start_frame
            or len(boxes) < segment_minimum
        ):
            raise WorkerError(f"O trecho facial {index} não cobre uma janela segura.")
        start_seconds, raw_end_seconds = start_frame / fps, end_frame / fps
        if start_seconds < previous_end - 1e-6:
            raise WorkerError("Os trechos faciais confirmados estão sobrepostos ou fora de ordem.")
        if raw_end_seconds > normalized_duration + 1e-6:
            raise WorkerError(f"O trecho facial {index} excede a duração normalizada do TalkNet.")
        if raw_end_seconds > source_duration + TALKNET_TIMELINE_TOLERANCE_SECONDS + 1e-6:
            raise WorkerError(f"O trecho facial {index} excede a faixa de vídeo original além da normalização permitida.")
        # floor, not round/ceil: every retained coordinate must fit completely
        # inside v:0. Slice the tail rather than resampling/relabeling the boxes.
        bounded_end = min(end_frame, math.floor(source_duration * fps + 1e-6))
        retained_frames = bounded_end - start_frame
        if retained_frames < segment_minimum:
            raise WorkerError(f"O trecho facial {index} fica vazio ou curto demais na faixa de vídeo original.")
        if raw.get("explicit_selected_short_track") is not None and bounded_end != end_frame:
            raise WorkerError("A faixa de vídeo original recorta a prova do trecho facial curto selecionado; revise a seleção.")
        segment = dict(raw)
        segment.update(start_frame=start_frame, end_frame=bounded_end, fps=fps, boxes=boxes[:retained_frames])
        bounded.append(segment)
        previous_end = bounded_end / fps
        if bounded_end != end_frame:
            clipped.append({
                "segment": index, "original_end_frame": end_frame, "end_frame": bounded_end,
                "removed_normalized_frames": end_frame - bounded_end,
                "reason": "Quadros finais da normalização fora da faixa de vídeo original; não renderizados",
            })
    return bounded, {
        "source_video_duration": source_duration,
        "normalized_video_duration": normalized_duration,
        "normalization_delta_seconds": normalized_duration - source_duration,
        "maximum_normalization_delta_seconds": TALKNET_TIMELINE_TOLERANCE_SECONDS,
        "clipped_segments": clipped,
    }


def _sequence(value: object) -> list:
    return value.tolist() if hasattr(value, "tolist") else list(value)  # type: ignore[arg-type]


def build_talknet_coordinate_cache(
    tracks: list,
    scores: list,
    frame_count: int,
    width: int,
    height: int,
) -> tuple[list[tuple[int, int, int, int]] | None, str]:
    """Converte tracks confirmadas do TalkNet em coords aceitas pelo MuseTalk.

    O atalho só é válido quando cada frame normalizado a 25 fps possui uma
    caixa facial íntegra.  A falta de uma única coordenada devolve ``None``;
    quem chama então mantém o pré-processamento completo do MuseTalk, em vez
    de arriscar alterar um rosto não confirmado.

    O TalkNet recorta o áudio e o vídeo para a duração comum antes de calcular
    as pontuações.  Em vídeos com um último quadro parcial, é normal a lista
    de ``scores`` ter poucos itens a menos que a trilha interpolada.  A caixa
    já foi confirmada pela trilha; neste caso reamostramos apenas a pontuação
    para o mesmo número de quadros.  Nunca inventamos caixas, interpolamos
    lacunas ou aceitamos uma lista vazia.
    """
    if frame_count <= 0 or width <= 0 or height <= 0 or len(tracks) != len(scores):
        return None, "trilhas ou dimensões inválidas"
    candidates: dict[int, list[tuple[float, tuple[int, int, int, int]]]] = {}
    for track_index, (track, score_list) in enumerate(zip(tracks, scores)):
        raw_track = track.get("track", {}) if isinstance(track, dict) else {}
        try:
            frames = _sequence(raw_track.get("frame", []))
            boxes = _sequence(raw_track.get("bbox", []))
            track_scores = _sequence(score_list)
        except TypeError:
            return None, f"trilha {track_index} não pode ser lida"
        if not frames or len(frames) != len(boxes) or not track_scores:
            return None, f"trilha {track_index} sem caixas ou pontuações"
        for position, (frame, bbox) in enumerate(zip(frames, boxes)):
            try:
                index = int(frame)
                x1, y1, x2, y2 = (int(round(float(part))) for part in bbox[:4])
                # Preserva a ordem da trilha.  Só a lista de pontuações pode
                # ter sido encurtada pelo recorte comum de áudio/vídeo.
                score_index = min(len(track_scores) - 1, (position * len(track_scores)) // len(frames))
                confidence = float(track_scores[score_index])
            except (TypeError, ValueError, IndexError):
                return None, f"trilha {track_index} contém coordenada inválida"
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if not (0 <= index < frame_count and x2 - x1 >= 24 and y2 - y1 >= 24):
                continue
            candidates.setdefault(index, []).append((confidence, (x1, y1, x2, y2)))
    coordinates = []
    for index in range(frame_count):
        choices = candidates.get(index, [])
        if not choices:
            return None, f"rosto não confirmado no quadro {index + 1} de {frame_count}"
        # A análise anterior já recusou dois falantes simultâneos. Durante
        # silêncio ou respiração, a caixa mais estável/mais bem pontuada evita
        # que um quadro seja aplicado em outro rosto.
        coordinates.append(max(choices, key=lambda item: item[0])[1])
    return coordinates, "coordenadas TalkNet completas"


def confirmed_talknet_face_segments(
    tracks: list,
    scores: list,
    frame_count: int,
    width: int,
    height: int,
    minimum_frames: int = 20,
) -> tuple[list[dict] | None, str]:
    """Extrai cada janela contínua de rosto falante confirmada pelo TalkNet.

    Ao contrário do cache de vídeo inteiro, lacunas entre cenas são esperadas.
    Cada janela precisa ter exatamente um rosto com pontuação de fala não
    negativa em todos os seus quadros; uma troca ambígua ou um trecho curto
    encerra a rota, em vez de aplicar lip-sync a uma pessoa errada.
    """
    candidates: dict[int, list[tuple[float, tuple[int, int, int, int]]]] = {}
    for track_index, (track, score_list) in enumerate(zip(tracks, scores)):
        raw_track = track.get("track", {}) if isinstance(track, dict) else {}
        try:
            frames = _sequence(raw_track.get("frame", []))
            boxes = _sequence(raw_track.get("bbox", []))
            track_scores = _sequence(score_list)
        except TypeError:
            return None, f"trilha TalkNet {track_index} não pode ser lida"
        if not frames or len(frames) != len(boxes) or not track_scores:
            continue
        for position, (frame, bbox) in enumerate(zip(frames, boxes)):
            try:
                index = int(frame)
                x1, y1, x2, y2 = (int(round(float(part))) for part in bbox[:4])
                score_index = min(len(track_scores) - 1, (position * len(track_scores)) // len(frames))
                confidence = float(track_scores[score_index])
            except (TypeError, ValueError, IndexError):
                return None, f"trilha TalkNet {track_index} contém coordenada inválida"
            if confidence < 0:
                continue
            bounded = (max(0, x1), max(0, y1), min(width, x2), min(height, y2))
            if 0 <= index < frame_count and bounded[2] - bounded[0] >= 24 and bounded[3] - bounded[1] >= 24:
                candidates.setdefault(index, []).append((confidence, bounded))
    if not candidates:
        return None, "TalkNet não confirmou nenhum quadro com rosto falante"
    indexed_boxes: list[tuple[int, tuple[int, int, int, int]]] = []
    for index in sorted(candidates):
        choices = candidates[index]
        if len(choices) != 1:
            return None, f"TalkNet encontrou mais de um rosto falante no quadro {index + 1}"
        indexed_boxes.append((index, choices[0][1]))
    groups: list[list[tuple[int, tuple[int, int, int, int]]]] = []
    current: list[tuple[int, tuple[int, int, int, int]]] = []
    for item in indexed_boxes:
        if current and item[0] != current[-1][0] + 1:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    short = next((group for group in groups if len(group) < minimum_frames), None)
    if short:
        return None, f"TalkNet confirmou um trecho facial curto demais no quadro {short[0][0] + 1}"
    segments = [{
        "start_frame": group[0][0],
        "end_frame": group[-1][0] + 1,
        "fps": 25.0,
        "boxes": [list(box) for _index, box in group],
    } for group in groups]
    covered = sum(len(group) for group in groups)
    return segments, f"TalkNet confirmou {len(segments)} trecho(s) facial(is) em {covered} quadro(s) ativos"


def build_s3fd_coordinate_cache(
    faces: list,
    target_frame_count: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[list[tuple[int, int, int, int]] | None, str]:
    """Reamostra caixas S3FD completas para todos os frames do MuseTalk.

    O TalkNet já executa S3FD em todos os frames normalizados.  Quando há
    apenas um rosto e o próprio TalkNet confirma o locutor, essas caixas são
    uma origem mais estável que os placeholders do detector interno do
    MuseTalk. Caixas ambíguas sempre falham. Ausências só podem ser
    interpoladas em lacunas de até 6 frames, cercadas por caixas estáveis;
    qualquer corte ou perda de rosto maior mantém a falha segura.
    """
    if (
        not faces
        or target_frame_count <= 0
        or min(source_width, source_height, target_width, target_height) <= 0
    ):
        return None, "detecções S3FD ou dimensões inválidas"

    source_coordinates: list[tuple[int, int, int, int] | None] = []
    for index, detected_faces in enumerate(faces):
        if not isinstance(detected_faces, list):
            return None, f"S3FD retornou dados inválidos no quadro {index + 1}"
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for detected in detected_faces:
            if not isinstance(detected, dict):
                continue
            try:
                frame = int(detected.get("frame", index))
                raw_bbox = detected.get("bbox", [])
                x1, y1, x2, y2 = (int(round(float(part))) for part in raw_bbox[:4])
                confidence = float(detected.get("conf", 0.0))
            except (TypeError, ValueError, IndexError):
                continue
            if frame != index:
                continue
            x1, x2 = max(0, x1), min(source_width, x2)
            y1, y2 = max(0, y1), min(source_height, y2)
            if x2 - x1 >= 24 and y2 - y1 >= 24:
                candidates.append((confidence, (x1, y1, x2, y2)))
        if len(candidates) > 1:
            return None, f"S3FD confirmou mais de um rosto no quadro {index + 1}"
        source_coordinates.append(candidates[0][1] if candidates else None)

    gaps: list[tuple[int, int]] = []
    start = None
    for index, coordinate in enumerate(source_coordinates):
        if coordinate is None and start is None:
            start = index
        elif coordinate is not None and start is not None:
            gaps.append((start, index - 1))
            start = None
    if start is not None:
        gaps.append((start, len(source_coordinates) - 1))
    missing = sum(end - start + 1 for start, end in gaps)
    if missing > max(2, int(len(source_coordinates) * 0.02)):
        return None, f"S3FD deixou {missing} quadros sem rosto confirmado"
    for start, end in gaps:
        gap_size = end - start + 1
        if gap_size > 6 or start == 0 or end >= len(source_coordinates) - 1:
            return None, f"S3FD deixou uma lacuna insegura no quadro {start + 1}"
        before, after = source_coordinates[start - 1], source_coordinates[end + 1]
        assert before is not None and after is not None
        before_width, before_height = before[2] - before[0], before[3] - before[1]
        after_width, after_height = after[2] - after[0], after[3] - after[1]
        if min(before_width, before_height, after_width, after_height) < 24:
            return None, f"caixa S3FD instável perto do quadro {start + 1}"
        # Uma troca brusca de pessoa/cena não pode receber uma caixa inventada.
        center_delta = abs((before[0] + before[2]) - (after[0] + after[2])) / 2 + abs((before[1] + before[3]) - (after[1] + after[3])) / 2
        max_motion = max(before_width, before_height, after_width, after_height) * 0.75
        if center_delta > max_motion:
            return None, f"movimento facial incerto perto do quadro {start + 1}"
        for offset in range(gap_size):
            fraction = (offset + 1) / (gap_size + 1)
            source_coordinates[start + offset] = tuple(
                int(round(left + (right - left) * fraction))
                for left, right in zip(before, after)
            )

    complete_coordinates = [coordinate for coordinate in source_coordinates if coordinate is not None]
    if len(complete_coordinates) != len(source_coordinates):
        return None, "S3FD não confirmou todos os quadros"

    scale_x, scale_y = target_width / source_width, target_height / source_height
    coordinates = []
    source_last = len(complete_coordinates) - 1
    target_last = max(1, target_frame_count - 1)
    for target_index in range(target_frame_count):
        # Mapeia por tempo relativo, portanto cobre corretamente 25 -> 30 fps
        # e vídeos cujo último quadro tenha duração parcial.
        source_index = int(math.floor((target_index * source_last) / target_last + 0.5))
        x1, y1, x2, y2 = complete_coordinates[source_index]
        scaled = (
            max(0, int(round(x1 * scale_x))),
            max(0, int(round(y1 * scale_y))),
            min(target_width, int(round(x2 * scale_x))),
            min(target_height, int(round(y2 * scale_y))),
        )
        if scaled[2] - scaled[0] < 24 or scaled[3] - scaled[1] < 24:
            return None, f"caixa S3FD inválida ao reamostrar o quadro {target_index + 1}"
        coordinates.append(scaled)
    gap_label = "sem lacunas" if not gaps else f"com {missing} lacunas curtas interpoladas"
    return coordinates, f"coordenadas S3FD completas e reamostradas ({gap_label})"


def stable_s3fd_face_box(
    faces: list,
    source_width: int,
    source_height: int,
) -> tuple[tuple[int, int, int, int] | None, str]:
    """Encontra uma caixa facial estável para um retrato de locutor único.

    Este é um fallback restrito para vídeos em que o detector S3FD encontra a
    mesma face por um trecho consistente, mas perde a ilustração/foto em
    quadros seguintes. Não escolhe entre duas pessoas e exige cobertura
    suficiente e pouca variação geométrica antes de sequer permitir a
    validação visual quadro a quadro.
    """
    if not faces or min(source_width, source_height) <= 0:
        return None, "S3FD sem dados para validar um retrato estático"
    boxes: list[tuple[int, int, int, int]] = []
    for index, detected_faces in enumerate(faces):
        if not isinstance(detected_faces, list):
            return None, f"S3FD retornou dados inválidos no quadro {index + 1}"
        candidates = []
        for detected in detected_faces:
            if not isinstance(detected, dict):
                continue
            try:
                frame = int(detected.get("frame", index))
                raw_bbox = detected.get("bbox", [])
                bbox = tuple(int(round(float(part))) for part in raw_bbox[:4])
            except (TypeError, ValueError, IndexError):
                continue
            if len(bbox) != 4 or frame != index:
                continue
            x1, y1, x2, y2 = bbox
            bounded = (max(0, x1), max(0, y1), min(source_width, x2), min(source_height, y2))
            if bounded[2] - bounded[0] >= 24 and bounded[3] - bounded[1] >= 24:
                candidates.append(bounded)
        if len(candidates) > 1:
            return None, f"S3FD encontrou mais de um rosto no quadro {index + 1}"
        if candidates:
            boxes.append(candidates[0])
    minimum = max(60, int(len(faces) * 0.05))
    if len(boxes) < minimum:
        return None, f"S3FD confirmou rosto em apenas {len(boxes)} de {len(faces)} quadros"
    median_box = tuple(int(round(statistics.median([box[part] for box in boxes]))) for part in range(4))
    median_width, median_height = median_box[2] - median_box[0], median_box[3] - median_box[1]
    if median_width < 24 or median_height < 24:
        return None, "caixa facial S3FD estável ficou pequena demais"
    for box in boxes:
        center_delta = abs((box[0] + box[2]) - (median_box[0] + median_box[2])) / 2 + abs((box[1] + box[3]) - (median_box[1] + median_box[3])) / 2
        size_delta = abs((box[2] - box[0]) - median_width) + abs((box[3] - box[1]) - median_height)
        if center_delta > max(median_width, median_height) * 0.35 or size_delta > (median_width + median_height) * 0.45:
            return None, "S3FD indicou movimento facial grande; retrato estático recusado"
    return median_box, f"retrato S3FD estável em {len(boxes)} de {len(faces)} quadros"


def s3fd_frame_boxes(
    faces: list,
    source_width: int,
    source_height: int,
) -> tuple[list[tuple[int, int, int, int] | None] | None, str]:
    """Normaliza a saída crua do S3FD para uma caixa facial por quadro."""
    if not faces or min(source_width, source_height) <= 0:
        return None, "S3FD sem dados de rosto para rastrear"
    boxes: list[tuple[int, int, int, int] | None] = []
    for index, detected_faces in enumerate(faces):
        if not isinstance(detected_faces, list):
            return None, f"S3FD retornou dados inválidos no quadro {index + 1}"
        candidates: list[tuple[int, int, int, int]] = []
        for detected in detected_faces:
            if not isinstance(detected, dict):
                continue
            try:
                frame = int(detected.get("frame", index))
                bbox = tuple(int(round(float(value))) for value in detected.get("bbox", [])[:4])
            except (TypeError, ValueError, IndexError):
                continue
            if frame != index or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bounded = (max(0, x1), max(0, y1), min(source_width, x2), min(source_height, y2))
            if bounded[2] - bounded[0] >= 24 and bounded[3] - bounded[1] >= 24:
                candidates.append(bounded)
        if len(candidates) > 1:
            return None, f"S3FD encontrou mais de um rosto no quadro {index + 1}"
        boxes.append(candidates[0] if candidates else None)
    return boxes, f"S3FD confirmou rosto em {sum(box is not None for box in boxes)} de {len(boxes)} quadros"


def initial_confirmed_s3fd_segment(
    faces: list,
    source_width: int,
    source_height: int,
    active_frames: set[int],
    minimum_frames: int = 50,
) -> tuple[dict | None, str]:
    """Obtém somente o trecho inicial com um rosto e locutor confirmados.

    Um vídeo pode começar com um locutor e, em seguida, cortar para imagens,
    paisagens ou pessoas que não falam. Não é seguro arrastar a caixa do
    primeiro rosto por essas cenas. Esta função aceita apenas uma sequência
    inicial contínua de caixas S3FD únicas e exige que o TalkNet não tenha
    marcado fala ativa fora dela.
    """
    if not faces or min(source_width, source_height) <= 0:
        return None, "S3FD sem dados para localizar o trecho facial"
    boxes: list[tuple[int, int, int, int]] = []
    for index, detected_faces in enumerate(faces):
        if not isinstance(detected_faces, list):
            return None, f"S3FD retornou dados inválidos no quadro {index + 1}"
        candidates: list[tuple[int, int, int, int]] = []
        for detected in detected_faces:
            if not isinstance(detected, dict):
                continue
            try:
                frame = int(detected.get("frame", index))
                bbox = tuple(int(round(float(value))) for value in detected.get("bbox", [])[:4])
            except (TypeError, ValueError, IndexError):
                continue
            if frame != index or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            bounded = (max(0, x1), max(0, y1), min(source_width, x2), min(source_height, y2))
            if bounded[2] - bounded[0] >= 24 and bounded[3] - bounded[1] >= 24:
                candidates.append(bounded)
        if len(candidates) != 1:
            break
        boxes.append(candidates[0])
    if len(boxes) < minimum_frames:
        return None, f"S3FD confirmou rosto somente nos primeiros {len(boxes)} quadros"
    later_active = sorted(frame for frame in active_frames if frame >= len(boxes))
    if later_active:
        return None, f"TalkNet confirmou fala fora do trecho facial no quadro {later_active[0] + 1}"
    return {
        "start_frame": 0,
        "end_frame": len(boxes),
        "fps": 25.0,
        "boxes": [list(box) for box in boxes],
    }, f"rosto falante confirmado nos primeiros {len(boxes)} quadros; cenas posteriores não têm fala facial ativa"


def resample_face_boxes(
    source_coordinates: list[tuple[int, int, int, int]],
    target_frame_count: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[list[tuple[int, int, int, int]] | None, str]:
    """Reamostra caixas por tempo relativo, inclusive de 25 para 30 FPS."""
    if not source_coordinates or target_frame_count <= 0:
        return None, "não há coordenadas faciais suficientes para reamostrar"
    if min(source_width, source_height, target_width, target_height) <= 0:
        return None, "dimensões inválidas para reamostrar o rosto"
    scale_x, scale_y = target_width / source_width, target_height / source_height
    source_last = len(source_coordinates) - 1
    target_last = max(1, target_frame_count - 1)
    coordinates: list[tuple[int, int, int, int]] = []
    for target_index in range(target_frame_count):
        source_index = int(math.floor((target_index * source_last) / target_last + 0.5))
        x1, y1, x2, y2 = source_coordinates[source_index]
        scaled = (
            max(0, int(round(x1 * scale_x))),
            max(0, int(round(y1 * scale_y))),
            min(target_width, int(round(x2 * scale_x))),
            min(target_height, int(round(y2 * scale_y))),
        )
        if scaled[2] - scaled[0] < 24 or scaled[3] - scaled[1] < 24:
            return None, f"caixa facial inválida ao reamostrar o quadro {target_index + 1}"
        coordinates.append(scaled)
    return coordinates, "coordenadas faciais completas e reamostradas"


def track_s3fd_face_optical_flow(
    normalized_video: Path,
    faces: list,
    source_width: int,
    source_height: int,
) -> tuple[list[tuple[int, int, int, int]] | None, str]:
    """Estende uma trilha S3FD única com fluxo óptico, sem inventar caixas.

    É usado somente quando TalkNet já confirmou um único locutor, mas o S3FD
    deixa de reconhecer uma ilustração durante zoom/pan. Cada quadro posterior
    precisa preservar pontos visuais rastreáveis; qualquer perda, salto ou
    ambiguidade rejeita o cache para que nenhum rosto seja aplicado às cegas.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None, "OpenCV ou NumPy indisponível para rastrear o rosto"
    detected, detail = s3fd_frame_boxes(faces, source_width, source_height)
    if not detected:
        return None, detail
    first = next((index for index, box in enumerate(detected) if box is not None), None)
    if first is None:
        return None, "S3FD não confirmou uma caixa inicial"
    contiguous = first
    while contiguous < len(detected) and detected[contiguous] is not None:
        contiguous += 1
    if first != 0 or contiguous - first < 60:
        return None, "S3FD não confirmou uma sequência inicial longa o bastante para rastrear"
    capture = cv2.VideoCapture(str(normalized_video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0 or abs(frame_count - len(detected)) > 2:
            return None, "a duração do vídeo normalizado não coincide com a análise S3FD"
        # A análise TalkNet é a fonte de verdade para os quadros que ela
        # detectou; o rastreador só preenche o trecho posterior que ficou sem
        # detecção e nunca substitui uma caixa confirmada.
        output: list[tuple[int, int, int, int]] = []
        previous_gray = None
        previous_points = None
        current_box: tuple[int, int, int, int] | None = None

        def points_in_box(gray, box):
            x1, y1, x2, y2 = box
            region = gray[y1:y2, x1:x2]
            if region.size == 0:
                return None
            found = cv2.goodFeaturesToTrack(region, maxCorners=180, qualityLevel=0.012, minDistance=3, blockSize=5)
            if found is None:
                return None
            found[:, 0, 0] += x1
            found[:, 0, 1] += y1
            return found

        for index in range(frame_count):
            ok, frame = capture.read()
            if not ok or frame is None:
                return None, f"não foi possível ler o quadro {index + 1} para rastrear o rosto"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            confirmed = detected[index] if index < len(detected) else None
            if confirmed is not None:
                current_box = confirmed
                previous_gray = gray
                previous_points = points_in_box(gray, confirmed)
                if previous_points is None or len(previous_points) < 12:
                    return None, f"poucos detalhes visuais no rosto confirmado no quadro {index + 1}"
                output.append(confirmed)
                continue
            if previous_gray is None or previous_points is None or current_box is None:
                return None, f"não há trilha facial segura para o quadro {index + 1}"
            if len(previous_points) < 18:
                previous_points = points_in_box(previous_gray, current_box)
            if previous_points is None or len(previous_points) < 12:
                return None, f"rastreador perdeu detalhes faciais antes do quadro {index + 1}"
            next_points, valid, _error = cv2.calcOpticalFlowPyrLK(
                previous_gray, gray, previous_points, None, winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
            )
            if next_points is None or valid is None:
                return None, f"fluxo óptico não respondeu no quadro {index + 1}"
            back_points, back_valid, _back_error = cv2.calcOpticalFlowPyrLK(
                gray, previous_gray, next_points, None, winSize=(31, 31), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, 0.01),
            )
            if back_points is None or back_valid is None:
                return None, f"validação inversa do rosto falhou no quadro {index + 1}"
            forward_mask = valid.reshape(-1).astype(bool) & back_valid.reshape(-1).astype(bool)
            distances = np.linalg.norm(previous_points.reshape(-1, 2) - back_points.reshape(-1, 2), axis=1)
            forward_mask &= distances <= 1.7
            old_points = previous_points.reshape(-1, 2)[forward_mask]
            new_points = next_points.reshape(-1, 2)[forward_mask]
            if len(old_points) < 12:
                return None, f"pontos faciais insuficientes no quadro {index + 1}"
            transform, inliers = cv2.estimateAffinePartial2D(old_points, new_points, method=cv2.RANSAC, ransacReprojThreshold=2.5)
            if transform is None or inliers is None or int(inliers.sum()) < 10:
                return None, f"movimento facial incerto no quadro {index + 1}"
            x1, y1, x2, y2 = current_box
            corners = np.array([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32)
            moved = cv2.transform(corners, transform)[0]
            next_box = (
                max(0, int(round(float(moved[:, 0].min())))),
                max(0, int(round(float(moved[:, 1].min())))),
                min(source_width, int(round(float(moved[:, 0].max())))),
                min(source_height, int(round(float(moved[:, 1].max())))),
            )
            width, height = next_box[2] - next_box[0], next_box[3] - next_box[1]
            old_width, old_height = x2 - x1, y2 - y1
            center_delta = abs((next_box[0] + next_box[2]) - (x1 + x2)) / 2 + abs((next_box[1] + next_box[3]) - (y1 + y2)) / 2
            if width < 24 or height < 24 or width > source_width * 0.9 or height > source_height * 0.9:
                return None, f"caixa facial saiu dos limites no quadro {index + 1}"
            if center_delta > max(old_width, old_height) * 0.45 + 6 or not (0.6 <= width / old_width <= 1.65 and 0.6 <= height / old_height <= 1.65):
                return None, f"movimento facial brusco no quadro {index + 1}"
            inlier_points = new_points[inliers.reshape(-1).astype(bool)].reshape(-1, 1, 2).astype(np.float32)
            replenished = points_in_box(gray, next_box)
            if replenished is not None and len(replenished):
                previous_points = np.concatenate((inlier_points, replenished), axis=0)[:220]
            else:
                previous_points = inlier_points
            if len(previous_points) < 12:
                return None, f"rastreador perdeu o rosto no quadro {index + 1}"
            previous_gray, current_box = gray, next_box
            output.append(next_box)
        return output, f"trilha óptica facial validada em {len(output)} quadros após {contiguous} confirmações S3FD"
    finally:
        capture.release()


def verify_locked_face_visual(video: Path, bbox: tuple[int, int, int, int], samples: int = 9) -> tuple[bool, str]:
    """Confirma visualmente que uma caixa facial estática continua no vídeo.

    Compara a região do rosto em amostras distribuídas pela duração. Esse
    fallback só vale para uma imagem/retrato visualmente estável; vídeos com
    cortes, movimento relevante ou outro rosto são recusados antes do
    MuseTalk, em vez de receberem uma coordenada fixa errada.
    """
    try:
        import cv2
    except ImportError:
        return False, "OpenCV indisponível para validar o retrato estático"
    capture = cv2.VideoCapture(str(video))
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if frame_count <= 0:
            return False, "não foi possível contar os quadros para validar o retrato"
        x1, y1, x2, y2 = bbox
        reference = None
        correlations = []
        for position in range(max(3, samples)):
            index = round((frame_count - 1) * position / max(1, samples - 1))
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.shape[0] < y2 or frame.shape[1] < x2:
                return False, f"quadro {index + 1} não contém a região facial esperada"
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False, f"região facial vazia no quadro {index + 1}"
            gray = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
            if reference is None:
                reference = gray
                continue
            correlation = float(cv2.matchTemplate(gray, reference, cv2.TM_CCOEFF_NORMED)[0][0])
            correlations.append(correlation)
        minimum = min(correlations, default=1.0)
        if minimum < 0.42:
            return False, f"retrato mudou demais entre os quadros (correlação mínima {minimum:.2f})"
        return True, f"retrato estático validado em {samples} amostras (correlação mínima {minimum:.2f})"
    finally:
        capture.release()


def scale_locked_face_box(
    bbox: tuple[int, int, int, int],
    target_frame_count: int,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[list[tuple[int, int, int, int]] | None, str]:
    if target_frame_count <= 0 or min(source_width, source_height, target_width, target_height) <= 0:
        return None, "dimensões inválidas para o retrato estático"
    x1, y1, x2, y2 = bbox
    scaled = (
        max(0, int(round(x1 * target_width / source_width))),
        max(0, int(round(y1 * target_height / source_height))),
        min(target_width, int(round(x2 * target_width / source_width))),
        min(target_height, int(round(y2 * target_height / source_height))),
    )
    if scaled[2] - scaled[0] < 24 or scaled[3] - scaled[1] < 24:
        return None, "caixa estática inválida na resolução do MuseTalk"
    return [scaled] * target_frame_count, "caixa estática validada para todos os quadros"


def lipsync_coordinate_mode(analysis: dict, fast_profile: bool) -> str:
    """Escolhe o atalho seguro de coordenadas sem tornar o lip-sync opcional.

    O TalkNet confirma *quem* está falando. Quando suas caixas não cobrem todos
    os frames do vídeo normalizado, elas não podem ser reutilizadas como cache.
    Isso não torna o rosto inseguro por si só: o MuseTalk ainda consegue fazer
    a detecção quadro a quadro e executar a inferência na GPU. Só recusamos o
    trabalho quando o TalkNet não confirmou um rosto falante seguro.
    """
    if not analysis.get("safe_for_single_face_lipsync"):
        return "rejected"
    if fast_profile and analysis.get("fast_coordinates"):
        return "talknet_cache"
    return "musetalk_gpu_detection"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(job_dir: Path, progress: int, stage: str, message: str) -> None:
    """Arquivo pequeno acompanhado pelo serviço do Pod sem tocar no player/UI."""
    target = job_dir / "progress.json"
    temporary = target.with_suffix(".writing")
    temporary.write_text(json.dumps({
        "progress": max(0, min(100, int(progress))), "stage": stage,
        "message": message, "updated_at": int(time.time()),
    }, ensure_ascii=False), encoding="utf-8")
    temporary.replace(target)


def srt_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, segments: list[dict], key: str) -> None:
    rows = []
    for index, segment in enumerate(segments, 1):
        rows.extend([str(index), f"{srt_time(segment['start'])} --> {srt_time(segment['end'])}", segment.get(key, "").strip(), ""])
    path.write_text("\n".join(rows), encoding="utf-8")


class Checkpoints:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"completed": [], "updated_at": 0}

    def done(self, stage: str, required: list[Path] | None = None) -> bool:
        return stage in self.data["completed"] and all(path.exists() and path.stat().st_size > 0 for path in (required or []))

    def mark(self, stage: str) -> None:
        if stage not in self.data["completed"]:
            self.data["completed"].append(stage)
        self.data["updated_at"] = int(time.time())
        write_json(self.path, self.data)

    def invalidate(self, *stages: str) -> None:
        removed = set(stages)
        self.data["completed"] = [stage for stage in self.data["completed"] if stage not in removed]
        self.data["updated_at"] = int(time.time())
        write_json(self.path, self.data)


def reset_incompatible_lipsync_cache(checkpoints: Checkpoints, job_dir: Path) -> bool:
    """Mantém áudio/transcrição, mas nunca mistura um renderer antigo ao novo."""
    marker = job_dir / "lipsync-cache-revision.json"
    current = {}
    try:
        current = json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    if current.get("revision") == LIPSYNC_CACHE_REVISION:
        return False
    checkpoints.invalidate("active_speaker", "lipsync", "finalize", "package")
    for path in (
        job_dir / "active-speaker.json", job_dir / "skipped-lipsync.json", job_dir / "synced.mp4",
        job_dir / "dublagem-final.mp4", job_dir / "report.json", job_dir / "video.pkl",
        job_dir / "talknet-faces.pkl", job_dir / "talknet-optical-coordinates.pkl", job_dir / "musetalk-input.mp4", job_dir / "musetalk-audio-16k.wav", job_dir / "musetalk-input.pkl",
        job_dir / "musetalk-confirmed-face.mp4", job_dir / "musetalk-confirmed-face.wav", job_dir / "musetalk-confirmed-face.pkl", job_dir / "musetalk-confirmed-face.yaml", job_dir / "lipsync-segmented.mp4",
    ):
        path.unlink(missing_ok=True)
    write_json(marker, {"revision": LIPSYNC_CACHE_REVISION, "updated_at": int(time.time())})
    return True


def separate_audio(video: Path, job_dir: Path, log: Path) -> tuple[Path, Path]:
    output_root = job_dir / "demucs"
    run([sys.executable, "-m", "demucs", "--device", "cuda", "--two-stems", "vocals", "-n", "htdemucs", "-o", str(output_root), str(video)], log=log)
    stem_dir = output_root / "htdemucs" / video.stem
    vocals, background = stem_dir / "vocals.wav", stem_dir / "no_vocals.wav"
    if not vocals.exists() or not background.exists():
        raise WorkerError("O Demucs não produziu vocals.wav e no_vocals.wav.")
    return vocals, background


def transcribe(vocals: Path, source_language: str, job_dir: Path, fast: bool = False) -> tuple[list[dict], str, float, str]:
    import torch
    from faster_whisper import WhisperModel

    model_name = "distil-large-v3" if fast else "large-v3"
    # O cache é definido pelo ambiente do Pod e fica no volume de rede. Sem
    # isso o Faster-Whisper pode escolher ~/.cache e baixar o modelo de novo
    # quando a sessão de GPU for reiniciada.
    cache_dir = os.environ.get("DUBSTUDIO_WHISPER_CACHE") or None
    model = WhisperModel(model_name, device="cuda", compute_type="float16", download_root=cache_dir)
    language = None if source_language == "auto" else source_language.split("-")[0]
    raw_segments, info = model.transcribe(str(vocals), language=language, word_timestamps=True, vad_filter=True)
    segments = []
    for index, item in enumerate(raw_segments, 1):
        text = item.text.strip()
        if not text:
            continue
        segments.append({
            "id": index, "start": round(item.start, 3), "end": round(item.end, 3), "text": text,
            "words": [{"start": word.start, "end": word.end, "word": word.word, "probability": word.probability} for word in (item.words or [])],
        })
    if not segments:
        raise WorkerError("Nenhuma fala foi reconhecida no vídeo.")
    word_probabilities = [float(word["probability"]) for segment in segments for word in segment["words"] if word.get("probability") is not None]
    average_word_probability = sum(word_probabilities) / len(word_probabilities) if word_probabilities else 0.0
    write_json(job_dir / "transcript.json", {
        "language": info.language, "probability": info.language_probability, "model": model_name,
        "word_probability": average_word_probability, "segments": segments,
    })
    del model
    torch.cuda.empty_cache()
    return segments, info.language, float(info.language_probability), model_name


def diarize(vocals: Path, segments: list[dict], speaker_count: str, job_dir: Path) -> list[dict]:
    import numpy as np
    import torch
    import torchaudio
    from sklearn.cluster import AgglomerativeClustering
    from speechbrain.inference.speaker import EncoderClassifier

    waveform, sample_rate = torchaudio.load(str(vocals))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"})
    embeddings = []
    valid_indices = []
    for index, segment in enumerate(segments):
        start, end = int(segment["start"] * sample_rate), int(segment["end"] * sample_rate)
        clip = waveform[:, start:end]
        if clip.shape[1] < int(0.45 * sample_rate):
            continue
        with torch.no_grad():
            embedding = encoder.encode_batch(clip.to("cuda")).squeeze().detach().cpu().numpy()
        embeddings.append(embedding / (np.linalg.norm(embedding) + 1e-9))
        valid_indices.append(index)
    if not embeddings:
        for item in segments:
            item["speaker"] = "SPEAKER_00"
        return segments
    matrix = np.stack(embeddings)
    requested = None if str(speaker_count) == "auto" else max(1, min(12, int(speaker_count)))
    if requested:
        labels = AgglomerativeClustering(n_clusters=min(requested, len(matrix)), metric="cosine", linkage="average").fit_predict(matrix)
    elif len(matrix) == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=0.55, metric="cosine", linkage="average").fit_predict(matrix)
        if len(set(labels)) > 12:
            labels = AgglomerativeClustering(n_clusters=12, metric="cosine", linkage="average").fit_predict(matrix)
    assigned = {index: int(label) for index, label in zip(valid_indices, labels)}
    last = 0
    for index, item in enumerate(segments):
        last = assigned.get(index, last)
        item["speaker"] = f"SPEAKER_{last:02d}"
        item["speaker_confidence"] = 0.75 if index in assigned else 0.45
    write_json(job_dir / "diarization.json", {"speaker_count": len({item['speaker'] for item in segments}), "segments": segments})
    del encoder
    torch.cuda.empty_cache()
    return segments


def translate_segments(segments: list[dict], source_language: str, target_language: str, job_dir: Path) -> list[dict]:
    translator = Path(__file__).with_name("dubstudio_translate.py")
    if not translator.exists():
        raise WorkerError("O adaptador de tradução não foi incluído no pacote.")
    translator_python = os.environ.get("DUBSTUDIO_TRANSLATE_PYTHON", sys.executable)
    source = job_dir / "translation-input.json"
    output = job_dir / "segments.json"
    write_json(source, {"segments": segments})
    run([translator_python, str(translator), "--input", str(source), "--output", str(output), "--source", source_language, "--target", target_language])
    return json.loads(output.read_text(encoding="utf-8"))["segments"]


def voice_references(vocals: Path, segments: list[dict], job_dir: Path) -> dict[str, Path]:
    refs: dict[str, Path] = {}
    by_speaker: dict[str, list[dict]] = {}
    for segment in segments:
        by_speaker.setdefault(segment["speaker"], []).append(segment)
    for speaker, items in by_speaker.items():
        best = max(items, key=lambda item: item["end"] - item["start"])
        ref = job_dir / f"reference-{speaker}.wav"
        run(["ffmpeg", "-y", "-i", str(vocals), "-ss", str(best["start"]), "-t", str(min(12, best["end"] - best["start"])), "-ac", "1", "-ar", "24000", str(ref)])
        refs[speaker] = ref
    return refs


async def edge_voice(target_language: str) -> str:
    import edge_tts

    if target_language in EDGE_PREFERRED:
        return EDGE_PREFERRED[target_language]
    voices = await edge_tts.list_voices()
    locale = target_language.lower()
    matches = [item["ShortName"] for item in voices if item.get("Locale", "").lower() == locale]
    if not matches:
        base = locale.split("-")[0]
        matches = [item["ShortName"] for item in voices if item.get("Locale", "").lower().startswith(base + "-")]
    if not matches:
        raise WorkerError(f"Nenhuma voz Edge TTS encontrada para {target_language}.")
    return matches[0]


def synthesize(
    segments: list[dict],
    target_language: str,
    vocals: Path,
    job_dir: Path,
    report: dict,
    require_authorized_clone: bool = False,
) -> Path:
    import torch
    import torchaudio
    from pydub import AudioSegment

    target_base = target_language.split("-")[0].lower()
    use_clone = target_base in CHATTERBOX_LANGUAGES
    if require_authorized_clone and not use_clone:
        raise WorkerError(
            f"A clonagem autorizada ainda não está disponível para {target_language}; "
            "nenhuma voz genérica será usada neste trabalho."
        )
    refs = voice_references(vocals, segments, job_dir)
    model = None
    if use_clone:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
        report["voice_engine"] = "Chatterbox Multilingual V3 authorized clone"
        report["voice_clone"] = {"required": require_authorized_clone, "confirmed": True}
    else:
        report["voice_engine"] = "Edge TTS generic fallback"
        report["voice_clone"] = {"required": require_authorized_clone, "confirmed": False}
        selected_edge = asyncio.run(edge_voice(target_language))
    total_ms = math.ceil(max(item["end"] for item in segments) * 1000) + 1000
    timeline = AudioSegment.silent(duration=total_ms, frame_rate=48000).set_channels(2)
    report["timing_adjustments"] = []
    for item in segments:
        raw = job_dir / f"tts-{item['id']:05d}.wav"
        if use_clone:
            wav = model.generate(item["translated_text"], language_id=target_base, audio_prompt_path=str(refs[item["speaker"]]))
            torchaudio.save(str(raw), wav.cpu(), model.sr)
        else:
            import edge_tts
            mp3 = raw.with_suffix(".mp3")
            asyncio.run(edge_tts.Communicate(item["translated_text"], selected_edge).save(str(mp3)))
            run(["ffmpeg", "-y", "-i", str(mp3), "-ac", "1", "-ar", "24000", str(raw)])
        target_duration = max(0.25, item["end"] - item["start"])
        actual_duration = ffprobe(raw)["duration"]
        speed = min(1.25, max(0.8, actual_duration / target_duration))
        adjusted = job_dir / f"tts-adjusted-{item['id']:05d}.wav"
        run(["ffmpeg", "-y", "-i", str(raw), "-filter:a", f"atempo={speed:.6f}", "-ac", "2", "-ar", "48000", str(adjusted)])
        clip = AudioSegment.from_file(adjusted)[: round(target_duration * 1000)]
        timeline = timeline.overlay(clip, position=round(item["start"] * 1000))
        item["speed"] = round(speed, 4)
        if speed in {0.8, 1.25}:
            report["timing_adjustments"].append({"segment_id": item["id"], "requested": actual_duration / target_duration, "applied": speed})
    dub = job_dir / "dub-dialogue.wav"
    timeline.export(dub, format="wav")
    if model is not None:
        del model
    torch.cuda.empty_cache()
    return dub


def mix_audio(background: Path, dub: Path, duration: float, job_dir: Path, log: Path) -> Path:
    output = job_dir / "dub-mix.wav"
    run([
        "ffmpeg", "-y", "-i", str(background), "-i", str(dub), "-filter_complex",
        "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=0.95[bg];"
        "[1:a]aformat=sample_rates=48000:channel_layouts=stereo,volume=1.0[voice];"
        "[bg][voice]amix=inputs=2:duration=longest:dropout_transition=0,"
        "loudnorm=I=-14:TP=-1.5:LRA=11[out]", "-map", "[out]", "-t", str(duration),
        # Dynamic loudnorm internally upsamples to 192 kHz for true-peak
        # detection. Constrain only its output; do not trim speech or change
        # the filter, input clock, loudness targets or duration contract.
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(output),
    ], log=log)
    return output


def canonical_talknet_source(source: str) -> str:
    """Repair only the pinned whole/partial AVI conversion commands.

    AVI's default encoded audio can pad the video with leading frames. An
    explicit timestamp-based fps filter plus PCM audio avoids that encoder
    delay. Detection and rendering must consume the same normalized timeline.
    """
    old = "-async 1 -r 25"
    new = "-async 1 -vf 'setpts=PTS-STARTPTS,fps=25:start_time=0' -c:a pcm_s16le"
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkerError("O código do TalkNet não possui estrutura válida.") from exc
    commands = [node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
                and node.value.startswith("ffmpeg -y -i %s -qscale:v 2 -threads %d ")
                and "-async " in node.value]
    # The same prefix is also used for image extraction; it is not a normalizer.
    if len(commands) != 2:
        raise WorkerError("Os comandos de normalização do TalkNet mudaram; atualização segura necessária.")
    if all(new in command and old not in command for command in commands) and source.count(new) == 2:
        return source
    if not all(old in command for command in commands) or source.count(old) != 2 or new in source:
        raise WorkerError("A normalização do TalkNet está alterada ou incompleta; não será usada.")
    repaired = source.replace(old, new)
    ast.parse(repaired)
    return repaired


def patch_musetalk_audio_frame_clock(root: "Path") -> dict:
    """Repair only the audited pinned PCM counters; no media is extended.

    This function is extractable on its own for an existing runtime: it uses
    only stdlib, verifies the Git commit and both AST expressions, and never
    changes packages, models, checkpoints, frame guards or tolerance values.
    """
    import ast
    import hashlib
    import subprocess
    from pathlib import Path

    root = Path(root)
    commit = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
    revision = "musetalk-pcm-integer-frame-clock-v1"
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True)
        if result.stdout.strip() != commit:
            raise RuntimeError("O contador de áudio do MuseTalk exige a revisão de código fixada; fonte desconhecida recusada.")
        path = root / "musetalk/utils/audio_processor.py"
        before = path.read_bytes()
        source = before.decode("utf-8")
        tree = ast.parse(source)
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AudioProcessor"]
        methods = [node for cls in classes for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "get_whisper_chunk"]
        if len(classes) != 1 or len(methods) != 1:
            raise ValueError("método incompatível")
        method = methods[0]
        expected = {
            "sr": "16000", "audio_fps": "50", "fps": "int(fps)",
            "num_frames": "math.floor((librosa_length / sr) * fps)",
            "actual_length": "math.floor((librosa_length / sr) * audio_fps)",
        }
        corrected = {"num_frames": "(librosa_length * fps) // sr",
                     "actual_length": "(librosa_length * audio_fps) // sr"}
        assignments = {}
        for node in ast.walk(method):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in expected:
                    if name in assignments or node not in method.body:
                        raise ValueError("contador duplicado ou condicional")
                    assignments[name] = node
        if set(assignments) != set(expected):
            raise ValueError("contadores ausentes")
        dump = lambda value: ast.dump(ast.parse(value, mode="eval").body, include_attributes=False)
        for name in ("sr", "audio_fps", "fps"):
            if ast.dump(assignments[name].value, include_attributes=False) != dump(expected[name]):
                raise ValueError("relógio incompatível")
        before_clock = all(ast.dump(assignments[name].value, include_attributes=False) == dump(expected[name]) for name in corrected)
        after_clock = all(ast.dump(assignments[name].value, include_attributes=False) == dump(corrected[name]) for name in corrected)
        if not (before_clock or after_clock):
            raise ValueError("contador desconhecido ou correção parcial")
        after = before
        if before_clock:
            lines = source.splitlines(keepends=True)
            for name, expression in corrected.items():
                node = assignments[name]
                if node.end_lineno != node.lineno:
                    raise ValueError("contador multilinha incompatível")
                line = lines[node.lineno - 1]
                indent = line[:len(line) - len(line.lstrip())]
                newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                lines[node.lineno - 1] = f"{indent}{name} = {expression}{newline}"
            patched = "".join(lines)
            compile(patched, str(path), "exec")
            after = patched.encode("utf-8")
            temporary = path.with_name(".dubstudio-audio-frame-clock.tmp")
            temporary.write_bytes(after)
            temporary.replace(path)
            if path.read_bytes() != after:
                raise OSError("não foi possível confirmar o código corrigido")
    except (OSError, UnicodeError, SyntaxError, ValueError, subprocess.CalledProcessError) as error:
        raise RuntimeError("A correção segura do relógio PCM do MuseTalk recusou uma fonte incompatível; nenhum resultado foi aprovado.") from error
    return {"revision": revision, "commit": commit, "changed": before != after,
            "source_sha256_before": hashlib.sha256(before).hexdigest(),
            "source_sha256_after": hashlib.sha256(after).hexdigest()}


def patch_talknet_video_normalization(root: Path) -> str:
    target = root / "demoTalkNet.py"
    source = target.read_text(encoding="utf-8")
    repaired = canonical_talknet_source(source)
    if repaired != source:
        backup = target.with_suffix(".py.dubstudio-normalization-original")
        if not backup.exists():
            shutil.copy2(target, backup)
        temporary = target.with_suffix(".py.normalization-writing")
        temporary.write_text(repaired, encoding="utf-8")
        temporary.replace(target)
    return TALKNET_NORMALIZATION_REVISION


def talknet_analysis(video: Path, job_dir: Path, log: Path) -> dict:
    root = Path(os.environ.get("DUBSTUDIO_TALKNET_ROOT", "/content/TalkNet-ASD"))
    if not root.exists():
        return {"status": "unavailable", "safe_for_single_face_lipsync": False, "ambiguous_intervals": []}
    normalization_revision = patch_talknet_video_normalization(root)
    # TalkNet still uses the removed NumPy alias ``np.int``.  Patch the
    # narrow call site instead of downgrading NumPy for the whole pipeline.
    box_utils = root / "model/faceDetector/s3fd/box_utils.py"
    if box_utils.exists():
        source = box_utils.read_text(encoding="utf-8")
        patched = source.replace(".astype(np.int)", ".astype(int)")
        if patched != source:
            box_utils.write_text(patched, encoding="utf-8")
    demo = root / "demo"
    demo.mkdir(exist_ok=True)
    target = demo / f"{video.stem}.mp4"
    coordinate_cache = job_dir / "video.pkl"
    s3fd_cache = job_dir / "talknet-faces.pkl"
    optical_cache = job_dir / "talknet-optical-coordinates.pkl"
    coordinate_cache.unlink(missing_ok=True)
    s3fd_cache.unlink(missing_ok=True)
    optical_cache.unlink(missing_ok=True)
    shutil.copy2(video, target)
    try:
        run([sys.executable, "demoTalkNet.py", "--videoName", video.stem, "--videoFolder", str(demo)], cwd=root, log=log)
    except Exception as exc:
        return {
            "status": "failed_safe",
            "safe_for_single_face_lipsync": False,
            "ambiguous_intervals": [],
            "reason": f"TalkNet não concluiu; quadros preservados por segurança: {str(exc)[-500:]}",
        }
    work = demo / video.stem / "pywork"
    tracks_path, scores_path, faces_path = work / "tracks.pckl", work / "scores.pckl", work / "faces.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        return {"status": "no_scores", "safe_for_single_face_lipsync": False, "ambiguous_intervals": []}
    with tracks_path.open("rb") as handle:
        tracks = pickle.load(handle)
    with scores_path.open("rb") as handle:
        scores = pickle.load(handle)
    active_by_frame: dict[int, int] = {}
    confident = 0
    for track, score_list in zip(tracks, scores):
        frames = track["track"]["frame"].tolist()
        for frame, score in zip(frames, score_list):
            if float(score) >= 0:
                active_by_frame[int(frame)] = active_by_frame.get(int(frame), 0) + 1
                if float(score) >= 0.5:
                    confident += 1
    ambiguous_frames = sorted(frame for frame, count in active_by_frame.items() if count > 1)
    total_active = max(1, len(active_by_frame))
    safe = bool(active_by_frame) and confident / total_active >= 0.45 and len(ambiguous_frames) / total_active <= 0.05
    active_frames = set(active_by_frame)
    intervals = []
    for frame in ambiguous_frames:
        moment = round(frame / 25, 3)
        if not intervals or moment - intervals[-1][1] > 0.08:
            intervals.append([moment, moment + 0.04])
        else:
            intervals[-1][1] = moment + 0.04
    normalized_video = demo / video.stem / "pyavi" / "video.avi"
    fast_coordinates = False
    coordinate_status = "análise TalkNet não foi segura para reutilizar coordenadas"
    s3fd_coordinates = False
    s3fd_status = "S3FD não foi confirmado para todos os quadros"
    locked_face_coordinates = False
    locked_face_status = "retrato estático não foi necessário"
    locked_face_box: list[int] = []
    optical_coordinates = False
    optical_status = "rastreamento óptico não foi necessário"
    confirmed_face_segment: dict = {}
    confirmed_face_segment_status = "não foi necessário separar um trecho facial"
    confirmed_face_segments: list[dict] = []
    confirmed_face_segments_status = "TalkNet ainda não mapeou janelas faciais"
    if safe and normalized_video.exists():
        normalized_meta = ffprobe(normalized_video)
        stream = next((item for item in normalized_meta["streams"] if item.get("codec_type") == "video"), {})
        frame_count = video_frame_count(normalized_video)
        confirmed, confirmed_face_segments_status = confirmed_talknet_face_segments(
            tracks,
            scores,
            frame_count,
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
        )
        if confirmed:
            confirmed_face_segments = confirmed
        coordinates, coordinate_status = build_talknet_coordinate_cache(
            tracks,
            scores,
            frame_count,
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
        )
        if coordinates:
            with coordinate_cache.open("wb") as handle:
                pickle.dump(coordinates, handle)
            fast_coordinates = True
        if not confirmed_face_segments and faces_path.exists() and faces_path.stat().st_size:
            try:
                with faces_path.open("rb") as handle:
                    faces = pickle.load(handle)
                _s3fd_coordinates, s3fd_status = build_s3fd_coordinate_cache(
                    faces,
                    frame_count,
                    int(stream.get("width") or 0),
                    int(stream.get("height") or 0),
                    int(stream.get("width") or 0),
                    int(stream.get("height") or 0),
                )
                if _s3fd_coordinates:
                    with s3fd_cache.open("wb") as handle:
                        pickle.dump(faces, handle)
                    s3fd_coordinates = True
                else:
                    confirmed, confirmed_face_segment_status = initial_confirmed_s3fd_segment(
                        faces,
                        int(stream.get("width") or 0),
                        int(stream.get("height") or 0),
                        active_frames,
                    )
                    if confirmed:
                        confirmed_face_segment = confirmed
                    else:
                        stable_box, locked_face_status = stable_s3fd_face_box(
                            faces,
                            int(stream.get("width") or 0),
                            int(stream.get("height") or 0),
                        )
                        if stable_box:
                            visually_stable, visual_status = verify_locked_face_visual(normalized_video, stable_box)
                            locked_face_status = f"{locked_face_status}; {visual_status}"
                            if visually_stable:
                                with s3fd_cache.open("wb") as handle:
                                    pickle.dump(faces, handle)
                                locked_face_coordinates = True
                                locked_face_box = list(stable_box)
                        if not locked_face_coordinates:
                            tracked, optical_status = track_s3fd_face_optical_flow(
                                normalized_video,
                                faces,
                                int(stream.get("width") or 0),
                                int(stream.get("height") or 0),
                            )
                            if tracked:
                                with optical_cache.open("wb") as handle:
                                    pickle.dump(tracked, handle)
                                optical_coordinates = True
            except Exception as exc:
                s3fd_status = f"cache S3FD não pôde ser lido: {exc}"
    else:
        coordinate_status = "vídeo normalizado do TalkNet indisponível"
    return {
        "status": "complete", "safe_for_single_face_lipsync": safe,
        "normalization_revision": normalization_revision,
        "tracks": len(tracks), "ambiguous_intervals": intervals,
        "fast_coordinates": fast_coordinates,
        "coordinate_cache": coordinate_status,
        "s3fd_coordinates": s3fd_coordinates,
        "s3fd_cache": s3fd_cache.name if s3fd_coordinates else "",
        "s3fd_status": s3fd_status,
        "locked_face_coordinates": locked_face_coordinates,
        "locked_face_box": locked_face_box,
        "locked_face_status": locked_face_status,
        "optical_coordinates": optical_coordinates,
        "optical_cache": optical_cache.name if optical_coordinates else "",
        "optical_status": optical_status,
        "confirmed_face_segment": confirmed_face_segment,
        "confirmed_face_segment_status": confirmed_face_segment_status,
        "confirmed_face_segments": confirmed_face_segments,
        "confirmed_face_segments_status": confirmed_face_segments_status,
        "face_cache": s3fd_cache.name if (s3fd_coordinates or locked_face_coordinates) else "",
        "normalization_fps": 25 if (fast_coordinates or s3fd_coordinates or locked_face_coordinates or optical_coordinates or confirmed_face_segment or confirmed_face_segments) else None,
    }


def prepare_musetalk_media(
    video: Path,
    audio: Path,
    job_dir: Path,
    log: Path,
    upscale_for_detector: bool = True,
) -> tuple[Path, Path]:
    """Normaliza somente a entrada do MuseTalk, preservando o mix final.

    O detector facial do MuseTalk é sensível a vídeos verticais de 360 px.
    A ampliação para 720 px melhora a detecção sem alterar o arquivo de
    origem.  O áudio de inferência é PCM mono em 16 kHz, exatamente o formato
    esperado pelo extrator Whisper interno; a exportação final volta a usar o
    mix estéreo completo.
    """
    meta = ffprobe(video)
    stream = next((item for item in meta["streams"] if item.get("codec_type") == "video"), {})
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise WorkerError("O vídeo não possui dimensões válidas para o lip-sync.")
    prepared_video = job_dir / "musetalk-input.mp4"
    # Para a fonte vertical curta enviada ao Studio, 720 px de largura entrega
    # uma face mais legível ao DWPose.  Vídeos já maiores não são reescalados.
    if upscale_for_detector and min(width, height) < 540:
        target_width = 720 if height >= width else 1280
        run([
            "ffmpeg", "-y", "-i", str(video), "-vf", f"scale={target_width}:-2:flags=lanczos,setsar=1",
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(prepared_video),
        ], log=log)
    else:
        prepared_video = video
    prepared_audio = job_dir / "musetalk-audio-16k.wav"
    run([
        "ffmpeg", "-y", "-i", str(audio), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(prepared_audio),
    ], log=log)
    source_duration = ffprobe(video)["duration"]
    if abs(ffprobe(prepared_audio)["duration"] - source_duration) > 0.5:
        raise WorkerError("O áudio preparado para o lip-sync perdeu duração; a renderização foi bloqueada.")
    if abs(video_stream_duration(prepared_video) - video_stream_duration(video)) > 0.5:
        raise WorkerError("O vídeo preparado para o lip-sync perdeu quadros; a renderização foi bloqueada.")
    return prepared_video, prepared_audio


def source_frame_rate(path: Path) -> float:
    """Lê a taxa de quadros para recompor somente o trecho sincronizado."""
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=r_frame_rate", "-of", "default=nokey=1:noprint_wrappers=1", str(path),
    ])
    value = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            rate = float(numerator) / float(denominator)
        else:
            rate = float(value)
    except (ValueError, ZeroDivisionError):
        rate = 0.0
    if not 1.0 <= rate <= 120.0:
        raise WorkerError("Não foi possível identificar a taxa de quadros do vídeo original.")
    return rate


def compose_confirmed_face_segment(
    rendered_segment: Path,
    source_video: Path,
    segment_duration: float,
    job_dir: Path,
    log: Path,
    face_boxes: object = None,
    box_fps: float = 0.0,
    box_size: tuple[int, int] | None = None,
) -> Path:
    """Legacy entry point delegates to the same bounded spatial compositor."""
    source_duration = video_stream_duration(source_video)
    if not (0.25 <= segment_duration < source_duration):
        raise WorkerError("O trecho confirmado de lip-sync não possui duração ou dimensões seguras.")
    return compose_face_patch(rendered_segment, source_video, source_video, 0.0, segment_duration, 1,
                              job_dir, log, face_boxes=face_boxes, box_fps=box_fps, box_size=box_size)


def render_confirmed_face_segment(
    video: Path,
    audio: Path,
    job_dir: Path,
    analysis: dict,
    log: Path,
    fast_profile: bool,
) -> Path:
    """Executa MuseTalk apenas onde há rosto falante confirmado pelo TalkNet."""
    partial = analysis.get("confirmed_face_segment") or {}
    raw_boxes = partial.get("boxes") if isinstance(partial, dict) else None
    try:
        start_frame = int(partial.get("start_frame", -1))
        end_frame = int(partial.get("end_frame", -1))
        fps = float(partial.get("fps", 0))
        boxes = [tuple(int(value) for value in box) for box in raw_boxes]
    except (TypeError, ValueError):
        raise WorkerError("O trecho facial confirmado não possui coordenadas válidas.") from None
    if start_frame != 0 or fps <= 0 or len(boxes) != end_frame - start_frame or len(boxes) < 50:
        raise WorkerError("O trecho facial confirmado não cobre uma janela inicial segura.")
    segment_duration = len(boxes) / fps
    talknet_root = Path(os.environ.get("DUBSTUDIO_TALKNET_ROOT", "/content/TalkNet-ASD"))
    normalized = talknet_root / "demo" / video.stem / "pyavi" / "video.avi"
    root = Path(os.environ.get("DUBSTUDIO_MUSETALK_ROOT", "/content/MuseTalk"))
    if not normalized.exists() or not root.exists():
        raise WorkerError("Os arquivos do TalkNet ou MuseTalk necessários para o trecho facial não estão disponíveis.")
    analysis["musetalk_audio_frame_clock"] = patch_musetalk_audio_frame_clock(root)
    normalized_duration = video_stream_duration(normalized)
    if segment_duration > normalized_duration + 0.15:
        raise WorkerError("O trecho facial excede o vídeo normalizado confirmado pelo TalkNet.")
    segment_video = job_dir / "musetalk-confirmed-face.mp4"
    segment_audio = job_dir / "musetalk-confirmed-face.wav"
    run([
        "ffmpeg", "-y", "-i", str(normalized), "-t", f"{segment_duration:.6f}",
        "-map", "0:v:0", "-an", "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "16", "-pix_fmt", "yuv420p", str(segment_video),
    ], log=log)
    run([
        "ffmpeg", "-y", "-i", str(audio), "-t", f"{segment_duration:.6f}", "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(segment_audio),
    ], log=log)
    if abs(video_stream_duration(segment_video) - segment_duration) > 0.35:
        raise WorkerError("A preparação do trecho facial perdeu duração; o lip-sync foi bloqueado.")
    if abs(ffprobe(segment_audio)["duration"] - segment_duration) > 0.35:
        raise WorkerError("O áudio do trecho facial perdeu duração; o lip-sync foi bloqueado.")
    normalized_meta = ffprobe(normalized)
    normalized_stream = next((item for item in normalized_meta["streams"] if item.get("codec_type") == "video"), {})
    segment_meta = ffprobe(segment_video)
    segment_stream = next((item for item in segment_meta["streams"] if item.get("codec_type") == "video"), {})
    coordinates, reason = resample_face_boxes(
        boxes,
        video_frame_count(segment_video),
        int(normalized_stream.get("width") or 0),
        int(normalized_stream.get("height") or 0),
        int(segment_stream.get("width") or 0),
        int(segment_stream.get("height") or 0),
    )
    if not coordinates:
        raise WorkerError(f"As coordenadas do trecho facial foram recusadas: {reason}")
    result_dir = job_dir / f"musetalk-result-{LIPSYNC_CACHE_REVISION}-confirmed-face"
    coordinate_path = result_dir.parent / f"{segment_video.stem}.pkl"
    with coordinate_path.open("wb") as handle:
        pickle.dump(coordinates, handle)
    config = job_dir / "musetalk-confirmed-face.yaml"
    config.write_text(f"task_0:\n  video_path: {segment_video}\n  audio_path: {segment_audio}\n", encoding="utf-8")
    before = {path.resolve() for path in result_dir.rglob("*.mp4")} if result_dir.exists() else set()
    command = [
        os.environ.get("DUBSTUDIO_MUSETALK_PYTHON", sys.executable), "-m", "scripts.inference",
        "--inference_config", str(config), "--result_dir", str(result_dir),
        "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15", "--ffmpeg_path", shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
        "--use_saved_coord", "--saved_coord",
    ]
    if fast_profile:
        command.extend(["--use_float16", "--batch_size", "24"])
    write_progress(job_dir, 82, "Aplicando lip-sync", "Sincronizando somente o rosto falante confirmado")
    last_progress = 82

    def report_musetalk_progress(line: str) -> None:
        nonlocal last_progress
        normalized_line = line.lower()
        target, message = 83, "Preparando o rosto confirmado para sincronização"
        if any(token in normalized_line for token in ("inference", "batch", "generating")):
            target, message = 86, "Inferência labial em execução na GPU"
        elif any(token in normalized_line for token in ("blend", "paste", "combine", "merge", "padding generated", "padding")):
            target, message = 90, "Compondo o rosto sincronizado no vídeo"
        elif any(token in normalized_line for token in ("ffmpeg", "mux", "encode", "video generation")):
            target, message = 92, "Exportando o trecho com lip-sync"
        if target > last_progress:
            last_progress = target
            write_progress(job_dir, target, "Aplicando lip-sync", message)

    os.environ["MPLBACKEND"] = "Agg"
    result = run_streaming(command, cwd=root, log=log, on_output=report_musetalk_progress)
    rendered = select_musetalk_output(result_dir, before, result, "MuseTalk não criou o trecho facial esperado.")
    assert_video_stream_coverage(rendered, segment_video)
    composed = compose_confirmed_face_segment(
        rendered, video, segment_duration, job_dir, log, face_boxes=coordinates, box_fps=fps,
        box_size=(int(segment_stream.get("width") or 0), int(segment_stream.get("height") or 0)),
    )
    source_duration = video_stream_duration(video)
    analysis["coordinate_source"] = "TalkNet S3FD no trecho facial confirmado"
    analysis["musetalk_input"] = {
        "video": segment_video.name,
        "audio": segment_audio.name,
        "video_duration": round(video_stream_duration(segment_video), 3),
        "coordinate_source": analysis["coordinate_source"],
    }
    analysis["lipsync_confirmed_segments"] = [{"start": 0.0, "end": round(segment_duration, 3)}]
    analysis["spatial_composition"] = perioral_composition_summary(job_dir, video, 1)
    analysis["preserved_no_speaker_visual_segments"] = ([{
        "start": round(segment_duration, 3), "end": round(source_duration, 3),
        "reason": "TalkNet não confirmou rosto falante; imagem original preservada",
    }] if source_duration - segment_duration > 0.05 else [])
    return composed


def spatial_face_boxes(
    boxes: object,
    box_fps: float,
    duration: float,
    box_size: tuple[int, int],
    output_size: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Validate every box before creating a strictly bounded alpha mask."""
    try:
        fps = float(box_fps)
        source_width, source_height = box_size
        width, height = output_size
    except (TypeError, ValueError):
        raise WorkerError("A máscara facial não possui dimensões válidas.") from None
    if isinstance(box_fps, bool) or not math.isfinite(fps) or fps != CANONICAL_FACE_FPS:
        raise WorkerError("A máscara facial exige o contrato canônico de 25 fps.")
    if (min(source_width, source_height, width, height) <= 0
            or not isinstance(boxes, (list, tuple)) or not boxes
            or len(boxes) != int(round(duration * fps))
            or abs(len(boxes) / fps - duration) > 1e-6):
        raise WorkerError("A máscara facial não cobre exatamente a janela confirmada.")
    scaled = []
    for raw in boxes:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise WorkerError("A máscara facial contém caixa inválida.")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or int(value) != value for value in raw):
            raise WorkerError("A máscara facial contém coordenadas inválidas.")
        x1, y1, x2, y2 = (int(value) for value in raw)
        if not (0 <= x1 < x2 <= source_width and 0 <= y1 < y2 <= source_height):
            raise WorkerError("A máscara facial ultrapassa a imagem rastreada.")
        mapped = (round(x1 * width / source_width), round(y1 * height / source_height),
                  round(x2 * width / source_width), round(y2 * height / source_height))
        if mapped[2] - mapped[0] < 24 or mapped[3] - mapped[1] < 24:
            raise WorkerError("A máscara facial ficou pequena demais após ajustar a resolução.")
        scaled.append(mapped)
    return scaled


PERIORAL_MASK_CONTRACT = "perioral-source-dwpose68-inner-feather-v1"
PERIORAL_MUSE_COMMIT = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"


def perioral_landmark_polygon(width: int, height: int, bbox: tuple[int, int, int, int],
                              points: object) -> tuple[list[tuple[float, float]], float]:
    """Real 68-point source evidence, never mouth coordinates invented from a box."""
    if not isinstance(points, (list, tuple)) or len(points) != 68:
        raise WorkerError("A máscara perioral exige 68 landmarks reais com confiança; composição bloqueada.")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise WorkerError("O track da máscara perioral ultrapassa o quadro original.")
    values = []
    required = {8, *range(17, 68)}
    for index, point in enumerate(points):
        if (not isinstance(point, (list, tuple)) or len(point) != 3
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in point)):
            raise WorkerError("A máscara perioral contém landmark inválido.")
        x, y, confidence = (float(value) for value in point)
        if not 0 <= confidence <= 1 or (index in required and
                (confidence < 0.55 or not x1 <= x < x2 or not y1 <= y < y2)):
            raise WorkerError(f"Landmark facial {index} ausente, fora do track ou incerto; composição perioral bloqueada.")
        values.append((x, y))
    left, right = values[48], values[54]
    mouth_width = math.hypot(right[0] - left[0], right[1] - left[1])
    if not 10 <= mouth_width <= 0.85 * math.hypot(x2 - x1, y2 - y1):
        raise WorkerError("Os landmarks não delimitam uma boca perioral segura.")
    ux, uy = (right[0] - left[0]) / mouth_width, (right[1] - left[1]) / mouth_width
    vx, vy = -uy, ux
    center = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
    if (values[8][0] - center[0]) * vx + (values[8][1] - center[1]) * vy < 0:
        vx, vy = -vx, -vy
    def vertical(point):
        return (point[0] - center[0]) * vx + (point[1] - center[1]) * vy
    protected = max(vertical(values[index]) for index in [*range(17, 27), *range(36, 48)])
    # This anatomical half-plane rotates with the mouth/face, not the image axes.
    upper_limit = max(protected + mouth_width * 0.18, vertical(values[30]) - mouth_width * 0.08)
    if (vertical(values[30]) >= -mouth_width * 0.12 or vertical(values[8]) <= mouth_width * 0.18
            or min(vertical(values[index]) for index in range(48, 60)) <= upper_limit + mouth_width * 0.04):
        raise WorkerError("Boca, nariz e olhos não têm separação anatômica confiável; composição bloqueada.")
    margin = min(14.0, mouth_width * 0.12)
    cloud = [(values[index][0] + math.cos(angle * math.pi / 4) * margin,
              values[index][1] + math.sin(angle * math.pi / 4) * margin)
             for index in [*range(31, 36), *range(48, 60)] for angle in range(8)]
    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    ordered = sorted(set(cloud))
    lower, upper = [], []
    for point in ordered:
        while len(lower) > 1 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(ordered):
        while len(upper) > 1 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    polygon = lower[:-1] + upper[:-1]
    # Intersect with the tracked face and a strict eye/forehead exclusion plane.
    planes = [(1, 0, -x1), (-1, 0, x2 - 1), (0, 1, -y1), (0, -1, y2 - 1),
              (vx, vy, -center[0] * vx - center[1] * vy - upper_limit)]
    for a, b, c in planes:
        clipped = []
        for previous, current in zip(polygon[-1:] + polygon[:-1], polygon):
            before, after = a * previous[0] + b * previous[1] + c, a * current[0] + b * current[1] + c
            if (before >= 0) != (after >= 0):
                ratio = before / (before - after)
                clipped.append((previous[0] + ratio * (current[0] - previous[0]),
                                previous[1] + ratio * (current[1] - previous[1])))
            if after >= 0:
                clipped.append(current)
        polygon = clipped
        if len(polygon) < 3:
            raise WorkerError("A região perioral não cabe no track confirmado.")
    return polygon, max(2.0, min(8.0, mouth_width * 0.08))


def perioral_mask_frame(width: int, height: int, bbox: tuple[int, int, int, int], points: object) -> bytes:
    """Convex mouth/lower-nose alpha; zero outside, with inward-only feather."""
    polygon, feather = perioral_landmark_polygon(width, height, bbox, points)
    edges = []
    for previous, current in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = current[0] - previous[0], current[1] - previous[1]
        length = math.hypot(dx, dy)
        if length > 1e-8:
            edges.append((-dy / length, dx / length, (dy * previous[0] - dx * previous[1]) / length))
    frame = bytearray(width * height)
    first_y, last_y = max(0, math.floor(min(p[1] for p in polygon))), min(height - 1, math.ceil(max(p[1] for p in polygon)))
    for y in range(first_y, last_y + 1):
        py = y + 0.5
        left, right, full_left, full_right = 0.0, float(width - 1), 0.0, float(width - 1)
        ceiling = 1.0
        for a, b, c in edges:
            constant = b * py + c
            if a > 1e-8:
                left = max(left, -constant / a - 0.5)
                full_left = max(full_left, (feather - constant) / a - 0.5)
            elif a < -1e-8:
                right = min(right, -constant / a - 0.5)
                full_right = min(full_right, (feather - constant) / a - 0.5)
            else:
                ceiling = min(ceiling, constant / feather)
        if ceiling <= 0 or left > right:
            continue
        start, end = max(0, math.ceil(left)), min(width - 1, math.floor(right))
        interior_start, interior_end = max(start, math.ceil(full_left)), min(end, math.floor(full_right))
        if ceiling >= 1 and interior_start <= interior_end:
            frame[y * width + interior_start:y * width + interior_end + 1] = b'\xff' * (interior_end - interior_start + 1)
            boundary = [*range(start, interior_start), *range(interior_end + 1, end + 1)]
        else:
            boundary = range(start, end + 1)
        for x in boundary:
            distance = min(a * (x + 0.5) + b * py + c for a, b, c in edges)
            frame[y * width + x] = max(0, min(255, int(255 * min(1.0, distance / feather))))
    if not any(frame):
        raise WorkerError("A máscara perioral ficou vazia; nenhum rosto inteiro será usado como fallback.")
    return bytes(frame)


def perioral_landmark_binding(reference_video: Path, boxes: list, width: int, height: int,
                               start_seconds: float) -> dict:
    if (isinstance(start_seconds, bool) or not isinstance(start_seconds, (int, float))
            or not math.isfinite(start_seconds)):
        raise WorkerError("O início dos landmarks periorais deve ser um instante canônico válido.")
    start_frame = round(start_seconds * CANONICAL_FACE_FPS)
    if (start_seconds < 0
            or abs(start_frame / CANONICAL_FACE_FPS - start_seconds) > 1e-6):
        raise WorkerError("Os landmarks periorais exigem a mesma janela canônica de 25 fps.")
    return {"contract": PERIORAL_MASK_CONTRACT, "musetalk_commit": PERIORAL_MUSE_COMMIT,
            "model": "dw-ll_ucoco_384-face-23:91", "source_sha256": sha256(reference_video),
            "boxes_sha256": hashlib.sha256(json.dumps(boxes, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
            "source_size": [width, height], "fps": CANONICAL_FACE_FPS,
            "start_frame": start_frame, "frame_count": len(boxes)}


def validate_perioral_landmarks(payload: object, binding: dict, boxes: list) -> list:
    if (not isinstance(payload, dict) or any(payload.get(key) != value for key, value in binding.items())
            or payload.get("device") not in {"cpu", "cuda:0"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(payload.get("model_sha256", "")))):
        raise WorkerError("O contrato/cache de landmarks periorais não corresponde à fonte, ao track ou à revisão; prova antiga recusada.")
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) != len(boxes):
        raise WorkerError("Faltam landmarks periorais em quadros confirmados; composição bloqueada.")
    width, height = binding["source_size"]
    for index, (frame, box) in enumerate(zip(frames, boxes)):
        if not isinstance(frame, dict) or frame.get("frame_index") != binding["start_frame"] + index:
            raise WorkerError("O relógio dos landmarks periorais não corresponde ao track confirmado.")
        try:
            perioral_landmark_polygon(width, height, box, frame.get("points"))
        except WorkerError as error:
            raise WorkerError(f"Landmarks periorais no quadro {frame['frame_index']} recusados: {error}") from error
    return frames


def generate_perioral_landmarks(reference_video: Path, boxes: list, width: int, height: int,
                                start_seconds: float, job_dir: Path, log: Path,
                                device: str | None = None) -> dict:
    """Load the pinned local DWPose once, inside the isolated MuseTalk runtime."""
    binding = perioral_landmark_binding(reference_video, boxes, width, height, start_seconds)
    device = device or os.environ.get("DUBSTUDIO_PERIORAL_DEVICE", "cuda:0")
    if device not in {"cpu", "cuda:0"}:
        raise WorkerError("Escolha explicitamente cpu ou cuda:0 para landmarks periorais.")
    root = Path(os.environ.get("DUBSTUDIO_MUSETALK_ROOT", "/content/MuseTalk"))
    executable = os.environ.get("DUBSTUDIO_MUSETALK_PYTHON", "")
    if not executable or not Path(executable).is_file() or not root.is_dir():
        raise WorkerError("Runtime MuseTalk de landmarks ausente; configure o Python isolado, sem fallback para caixa facial.")
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    if head.returncode or head.stdout.strip() != PERIORAL_MUSE_COMMIT:
        raise WorkerError("Os landmarks periorais exigem a revisão MuseTalk fixada.")
    checkpoint = root / "models/dwpose/dw-ll_ucoco_384.pth"
    config = root / "musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py"
    if not checkpoint.is_file() or not config.is_file():
        raise WorkerError("Modelo/configuração DWPose local ausente; landmarks não serão inventados.")
    model_sha = sha256(checkpoint)
    identity = hashlib.sha256(json.dumps({**binding, "model_sha256": model_sha}, sort_keys=True).encode()).hexdigest()
    target = job_dir / f"perioral-landmarks-{identity}.json"
    if target.is_file():
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise WorkerError("O cache de landmarks periorais está ilegível; composição bloqueada.") from error
        validate_perioral_landmarks(payload, binding, boxes)
        if payload["model_sha256"] != model_sha:
            raise WorkerError("Os pesos dos landmarks diferem do cache perioral.")
        return payload
    request = job_dir / f".perioral-request-{time.time_ns()}.json"
    staged = job_dir / f".perioral-landmarks-{time.time_ns()}.json"
    write_json(request, {**binding, "boxes": boxes, "source": str(reference_video.resolve()),
                         "model_sha256": model_sha, "device": device})
    script = r'''
import os, sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import cv2, torch, mmpose
from mmpose.apis import init_model, inference_topdown
request = json.loads(Path(sys.argv[1]).read_text())
if mmpose.__version__ != "1.1.0":
    raise RuntimeError("Versão mmpose inesperada; landmarks recusados")
device = request["device"]
if device == "cuda:0" and not torch.cuda.is_available():
    raise RuntimeError("GPU de landmarks solicitada indisponível; não haverá fallback CPU")
torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
model = init_model("./musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py",
                   "./models/dwpose/dw-ll_ucoco_384.pth", device=device)
width, height = request["source_size"]
start, count = request["start_frame"], request["frame_count"]
clock = f"setpts=PTS-STARTPTS,fps=25:start_time=0,trim=start_frame={start}:end_frame={start+count},setpts=PTS-STARTPTS"
command = ["ffmpeg", "-nostdin", "-v", "error", "-i", request["source"], "-vf", clock,
           "-map", "0:v:0", "-an", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
frames = []
try:
    for offset, box in enumerate(request["boxes"]):
        needed, chunks = width*height*3, []
        while needed:
            chunk = process.stdout.read(needed)
            if not chunk: raise RuntimeError(f"Fonte sem quadro perioral {start+offset}")
            chunks.append(chunk); needed -= len(chunk)
        frame = np.frombuffer(b"".join(chunks), np.uint8).reshape(height, width, 3)
        x1,y1,x2,y2 = box
        crop = frame[y1:y2,x1:x2].copy()
        predictions = inference_topdown(model, crop,
            bboxes=np.array([[0,0,x2-x1,y2-y1]], dtype=np.float32), bbox_format="xyxy")
        if len(predictions) != 1: raise RuntimeError("Não há um único rosto no track perioral")
        instances = predictions[0].pred_instances
        points, scores = instances.keypoints, instances.keypoint_scores
        if points.shape != (1,133,2) or scores.shape != (1,133):
            raise RuntimeError("DWPose não produziu o contrato facial 133/68 com confiança")
        face = points[0,23:91]
        confidence = scores[0,23:91]
        frames.append({"frame_index":start+offset,"points":[[float(p[0]+x1),float(p[1]+y1),float(s)] for p,s in zip(face,confidence)]})
        if offset%25 == 0: print(f"Landmarks periorais {offset+1}/{count} ({device})", flush=True)
    if process.stdout.read(1): raise RuntimeError("Fonte possui quadros extras no contrato perioral")
    if process.wait(timeout=30): raise RuntimeError(process.stderr.read().decode(errors="replace")[-1500:])
finally:
    if process.poll() is None: process.kill(); process.wait()
payload = {key:value for key,value in request.items() if key not in {"source","boxes"}}
payload["frames"] = frames
Path(sys.argv[2]).write_text(json.dumps(payload,allow_nan=False), encoding="utf-8")
'''
    environment = dict(os.environ)
    environment.update(MPLBACKEND="Agg", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    try:
        result = subprocess.run([executable, "-c", script, str(request), str(staged)], cwd=root,
                                env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=max(180, min(7200, len(boxes) * 3)))
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\n$ perioral landmarks ({device})\n{result.stdout}\n{result.stderr}\n")
        if result.returncode or not staged.is_file():
            raise WorkerError(f"Landmarks periorais falharam; composição bloqueada: {(result.stderr or result.stdout)[-1600:]}")
        try:
            payload = json.loads(staged.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise WorkerError("A inferência não produziu landmarks periorais legíveis; composição bloqueada.") from error
        validate_perioral_landmarks(payload, binding, boxes)
        if (payload["model_sha256"] != model_sha or sha256(reference_video) != binding["source_sha256"]
                or sha256(checkpoint) != model_sha):
            raise WorkerError("Fonte ou pesos mudaram durante os landmarks periorais; cache recusado.")
        staged.replace(target)
        return payload
    except subprocess.TimeoutExpired as error:
        raise WorkerError("Tempo de landmarks periorais excedido; nenhum rosto inteiro será usado.") from error
    finally:
        request.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def spatial_face_mask_frame(width: int, height: int, bbox: tuple[int, int, int, int]) -> bytes:
    """Zero outside the tracked box; feathering never expands beyond it."""
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise WorkerError("A máscara facial ultrapassa o quadro original.")
    box_width, box_height = x2 - x1, y2 - y1
    feather = max(2, min(12, min(box_width, box_height) // 10))
    horizontal = [min(255, 255 * min(x, box_width - 1 - x) // feather) for x in range(box_width)]
    rows: dict[int, bytes] = {}
    frame = bytearray(width * height)
    for y in range(box_height):
        opacity = min(255, 255 * min(y, box_height - 1 - y) // feather)
        if opacity not in rows:
            rows[opacity] = bytes(min(value, opacity) for value in horizontal)
        row = rows[opacity]
        offset = (y1 + y) * width + x1
        frame[offset:offset + box_width] = row
    return bytes(frame)


def write_spatial_face_mask(path: Path, boxes: list[tuple[int, int, int, int]],
                            width: int, height: int, fps: float, log: Path,
                            reference_video: Path | None = None, start_seconds: float = 0.0,
                            landmark_payload: dict | None = None) -> None:
    # Matroska's millisecond timestamps are exact for 25-fps/40-ms samples.
    # Other rates can desynchronize alphamerge even when boxes/render agree;
    # refuse those contracts instead of relabelling/resampling facial proof.
    try:
        canonical_rate = not isinstance(fps, bool) and float(fps) == CANONICAL_FACE_FPS
    except (TypeError, ValueError, OverflowError):
        canonical_rate = False
    if not canonical_rate:
        raise WorkerError("A máscara facial exige o contrato canônico de 25 fps.")
    if reference_video is None:
        raise WorkerError("A máscara perioral exige vídeo original e landmarks; máscara de rosto inteiro bloqueada.")
    binding = perioral_landmark_binding(reference_video, boxes, width, height, start_seconds)
    payload = landmark_payload if landmark_payload is not None else generate_perioral_landmarks(
        reference_video, boxes, width, height, start_seconds, path.parent, log)
    frames = validate_perioral_landmarks(payload, binding, boxes)
    command = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error",
               "-f", "rawvideo", "-pixel_format", "gray", "-video_size", f"{width}x{height}",
               "-framerate", str(CANONICAL_FACE_FPS), "-i", "pipe:0", "-an", "-c:v", "ffv1",
               "-level", "3", "-pix_fmt", "gray", str(path)]
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ perioral alpha mask: real source landmarks, eyes/forehead zero, inward feather\n")
        handle.flush()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=handle)
        try:
            assert process.stdin is not None
            for bbox, frame in zip(boxes, frames):
                process.stdin.write(perioral_mask_frame(width, height, bbox, frame["points"]))
            process.stdin.close()
            code = process.wait(timeout=120)
        except (OSError, subprocess.TimeoutExpired, WorkerError) as error:
            process.kill()
            process.wait()
            if isinstance(error, WorkerError):
                raise
            raise WorkerError("Não foi possível criar a máscara facial segura.") from None
    if code or not path.is_file() or path.stat().st_size <= 0:
        raise WorkerError("A criação da máscara facial falhou; nenhuma composição foi liberada.")
    write_json(path.with_suffix(".perioral.json"), {**binding, "mask_sha256": sha256(path),
               "landmarks_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest(),
               "device": payload["device"], "model_sha256": payload["model_sha256"],
               "eyes_forehead_alpha": 0, "feather": "inward-only", "composition_completed": False,
               "human_visual_review": "pending"})


def compose_face_patch(
    rendered_segment: Path,
    base_video: Path,
    reference_video: Path,
    start_seconds: float,
    segment_duration: float,
    index: int,
    job_dir: Path,
    log: Path,
    face_boxes: object = None,
    box_fps: float = 0.0,
    box_size: tuple[int, int] | None = None,
    landmark_payload: dict | None = None,
    display_window: tuple[float, float] | None = None,
) -> Path:
    """Aplica uma janela MuseTalk sobre o relógio completo do vídeo original.

    Recortar, converter cada parte para FPS e concatenar perde quadros quando
    os limites de 25 fps do TalkNet não coincidem com os quadros do original.
    O overlay conserva os PTS/quadros do vídeo-base, inclusive em fontes VFR;
    somente pixels periorais guiados por landmarks e quadros em [início, fim) recebem
    a imagem renderizada. O resto vem do vídeo-base no PTS original, não do
    quadro normalizado a 25 fps que o MuseTalk devolveu.
    """
    # The reference remains the original video even after preceding patches.
    # A normalized AVI or an accumulated composition cannot extend its timeline.
    source_duration = video_stream_duration(reference_video)
    base_duration = video_stream_duration(base_video)
    end_seconds = start_seconds + segment_duration
    display_start, display_end = start_seconds, end_seconds
    if display_window is not None:
        if (not isinstance(display_window, tuple) or len(display_window) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                       for value in display_window)):
            raise WorkerError("A janela nativa da pessoa não possui limites válidos.")
        display_start, display_end = display_window
        if (display_start < start_seconds - 1e-6 or display_end > end_seconds + 1e-6
                or display_end <= display_start):
            raise WorkerError("A composição tentou cruzar o turno nativo da pessoa correta.")
    reference_meta = ffprobe(reference_video)
    stream = next((item for item in reference_meta["streams"] if item.get("codec_type") == "video"), {})
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    display_frame_window = None
    if display_window is not None:
        try:
            native_rate = Fraction(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate")))
            if not 1 <= native_rate <= 120:
                raise ValueError("rate")
        except (ValueError, TypeError, ZeroDivisionError) as error:
            raise WorkerError("A composição nativa precisa da taxa real dos quadros montados.") from error
        display_frame_window = (math.ceil(display_start * float(native_rate) - 1e-7),
                                math.ceil(display_end * float(native_rate) - 1e-7))
        if display_frame_window[1] <= display_frame_window[0]:
            raise WorkerError("A janela nativa não cobre um quadro completo da pessoa correta.")
    if (
        min(width, height) <= 0 or not math.isfinite(source_duration) or source_duration <= 0
        or not math.isfinite(base_duration) or abs(base_duration - source_duration) > 0.05
        or not math.isfinite(start_seconds) or not math.isfinite(segment_duration)
        or start_seconds < 0 or segment_duration <= 0 or end_seconds > source_duration + 1e-6
    ):
        raise WorkerError("A janela facial confirmada não cabe com segurança no vídeo original.")
    rendered_duration = video_stream_duration(rendered_segment)
    expected_frames = video_frame_count(reference_video)
    if (
        not math.isfinite(rendered_duration) or rendered_duration < segment_duration - 1e-6
        or expected_frames <= 0 or video_frame_count(base_video) != expected_frames
    ):
        raise WorkerError("O trecho facial ou vídeo-base perdeu quadros; composição bloqueada.")
    if box_size is None:
        raise WorkerError("A composição exige caixas faciais rastreadas; overlay completo foi bloqueado.")
    scaled_boxes = spatial_face_boxes(face_boxes, box_fps, segment_duration, box_size, (width, height))
    rendered_meta = ffprobe(rendered_segment)
    rendered_stream = next((item for item in rendered_meta["streams"] if item.get("codec_type") == "video"), {})
    try:
        rates = [Fraction(str(rendered_stream.get(key))) for key in ("r_frame_rate", "avg_frame_rate")]
    except (ValueError, ZeroDivisionError):
        rates = []
    rendered_frames = video_frame_count(rendered_segment)
    if (
        len(rates) != 2 or any(rate != CANONICAL_FACE_FPS for rate in rates)
        or rendered_frames < len(scaled_boxes)
        or abs(rendered_frames / CANONICAL_FACE_FPS - rendered_duration) > 1e-6
    ):
        raise WorkerError(
            "O render facial não corresponde às caixas, à duração e ao contrato canônico de 25 fps; composição bloqueada."
        )
    # MuseTalk can pad its own tail. Canonical extra samples are harmless:
    # trim removes them before alphamerge, and overlay's end stays exclusive.
    # Never extend a short render or relabel/resample an incompatible rate.
    mask = job_dir / f"musetalk-perioral-source-mask-v1-{index:02d}.mkv"
    write_spatial_face_mask(mask, scaled_boxes, width, height, box_fps, log,
                            reference_video=reference_video, start_seconds=start_seconds,
                            landmark_payload=landmark_payload)
    proof_path = mask.with_suffix(".perioral.json")
    try:
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WorkerError("Falta a prova atual da máscara perioral; composição bloqueada.") from error
    binding = perioral_landmark_binding(reference_video, scaled_boxes, width, height, start_seconds)
    if (not isinstance(proof, dict) or any(proof.get(key) != value for key, value in binding.items())
            or proof.get("mask_sha256") != sha256(mask) or proof.get("eyes_forehead_alpha") != 0
            or proof.get("feather") != "inward-only" or proof.get("human_visual_review") != "pending"
            or proof.get("composition_completed") is not False):
        raise WorkerError("A prova perioral não corresponde à máscara atual; composição bloqueada.")
    enable = (f"gte(n,{display_frame_window[0]})*lt(n,{display_frame_window[1]})" if display_frame_window is not None
              else f"gte(t,{display_start:.9f})*lt(t,{display_end:.9f})")
    filters = [
        "[1:v]setpts=PTS-STARTPTS,setsar=1[base]",
        f"[0:v]trim=duration={segment_duration:.6f},"
        "setpts=PTS-STARTPTS,"
        f"scale={width}:{height}:flags=lanczos,setsar=1[face]",
        f"[2:v]trim=duration={segment_duration:.6f},setpts=PTS-STARTPTS,setsar=1[mask]",
        f"[face][mask]alphamerge,setpts=PTS-STARTPTS+{start_seconds:.6f}/TB[patch]",
        # Keep the final 25 fps sample through its own display interval. The
        # explicit exclusive end prevents repeating it outside confirmed time.
        "[base][patch]overlay=eof_action=repeat:repeatlast=1:"
        f"enable='{enable}'[video]",
    ]
    output = job_dir / f"lipsync-segmented-perioral-v1-{index:02d}.mp4"
    run([
        "ffmpeg", "-y", "-i", str(rendered_segment), "-i", str(base_video), "-i", str(mask),
        "-filter_complex", ";".join(filters), "-map", "[video]", "-an", "-t", f"{source_duration:.6f}",
        "-vsync", "0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output),
    ], log=log)
    if video_frame_count(output) != expected_frames:
        raise WorkerError("A composição alterou a contagem de quadros do vídeo original; resultado bloqueado.")
    assert_video_stream_coverage(output, reference_video)
    if sha256(reference_video) != binding["source_sha256"] or sha256(mask) != proof["mask_sha256"]:
        raise WorkerError("Fonte ou máscara mudaram durante a composição perioral; resultado bloqueado.")
    proof.update(render_sha256=sha256(rendered_segment), output_sha256=sha256(output),
                 composition_completed=True, output_frame_count=expected_frames)
    if display_window is not None:
        proof.update(display_window={"start": display_start, "end": display_end},
                     display_window_policy="audio-master-native-person-clamp-v1",
                     display_frame_window={"start_frame": display_frame_window[0],
                                           "end_frame": display_frame_window[1], "fps": float(native_rate)})
    write_json(proof_path, proof)
    return output


def audio_master_native_display(raw_segment: dict, analysis: dict) -> tuple[tuple[float, float], dict]:
    """Clamp canonical patches to actual native frames of the correct person."""
    from dubstudio_audio_master import validate_timeline
    timeline = validate_timeline(analysis["timeline"])
    native_fps = Fraction(timeline["video_base"]["fps_num"], timeline["video_base"]["fps_den"])
    mappings = [item for item in timeline["visual_mapping"]
                if item["source_segment_id"] == raw_segment.get("source_segment_id")
                and item["speaker"] == raw_segment.get("speaker")]
    if len(mappings) != 1:
        raise WorkerError("A inferência facial não corresponde a um turno único da pessoa.")
    mapping = mappings[0]
    canonical_fps = Fraction(str(raw_segment["fps"]))
    left = max(Fraction(raw_segment["start_frame"], 1) / canonical_fps,
               Fraction(mapping["output_start_frame"], 1) / native_fps)
    right = min(Fraction(raw_segment["end_frame"], 1) / canonical_fps,
                Fraction(mapping["output_end_frame"], 1) / native_fps)
    first = max(mapping["output_start_frame"], math.ceil(left * native_fps))
    last = min(mapping["output_end_frame"], math.ceil(right * native_fps))
    if right <= left or last <= first:
        raise WorkerError("O trecho de inferência não cobre nenhum quadro nativo seguro da pessoa.")
    source_group = next((item for item in analysis.get("mapped_source_segments", [])
                         if item.get("source_segment_id") == mapping["source_segment_id"]
                         and item.get("speaker") == mapping["speaker"]), None)
    if source_group is None or source_group["start_frame"] > first or source_group["end_frame"] < last:
        raise WorkerError("A prova de rosto não cobre os quadros nativos realmente compostos.")
    proof = {**{key: raw_segment.get(key) for key in ("source_segment_id", "speaker", "track_id",
               "face_confirmation", "speaker_confirmation", "coordinate_source")},
             "start_frame": first, "end_frame": last, "fps": float(native_fps),
             "boxes": source_group["boxes"][first - source_group["start_frame"]:last - source_group["start_frame"]],
             "canonical_inference_window": {"start_frame": raw_segment["start_frame"],
                                            "end_frame": raw_segment["end_frame"], "fps": float(canonical_fps)},
             "display_window": {"start": float(left), "end": float(right)}}
    return (float(left), float(right)), proof


def audio_master_boundary_preservation(analysis: dict, confirmed: list[dict]) -> list[dict]:
    """Report only unpatched native edge samples, never invent speaker scores."""
    timeline = analysis["timeline"]
    fps = Fraction(timeline["video_base"]["fps_num"], timeline["video_base"]["fps_den"])
    maximum = max(float(1 / fps), 1 / CANONICAL_FACE_FPS)
    records = []
    for mapping in timeline["visual_mapping"]:
        if mapping.get("face_confirmation") == "no_face_confirmed":
            records.append({"start": float(Fraction(mapping["output_start_frame"], 1) / fps),
                            "end": float(Fraction(mapping["output_end_frame"], 1) / fps),
                            "reason": "narração sem rosto explicitamente confirmada", "kind": "confirmed_faceless_narration",
                            "source_segment_id": mapping["source_segment_id"], "speaker": mapping["speaker"]})
            continue
        groups = sorted([item for item in confirmed if item.get("source_segment_id") == mapping["source_segment_id"]
                         and item.get("speaker") == mapping["speaker"]], key=lambda item: item["start_frame"])
        cursor = mapping["output_start_frame"]
        gaps = []
        for item in groups:
            if item["start_frame"] > cursor:
                gaps.append((cursor, item["start_frame"]))
            cursor = max(cursor, item["end_frame"])
        if cursor < mapping["output_end_frame"]:
            gaps.append((cursor, mapping["output_end_frame"]))
        for start, end in gaps:
            duration = float(Fraction(end - start, 1) / fps)
            if (duration > maximum + 1e-7
                    or (start != mapping["output_start_frame"] and end != mapping["output_end_frame"])):
                raise WorkerError("A inferência deixou uma lacuna interna ou longa no turno da pessoa; lip-sync bloqueado.")
            records.append({"start": float(Fraction(start, 1) / fps), "end": float(Fraction(end, 1) / fps),
                            "reason": "Fronteira canônica sem inferência; quadro original da pessoa preservado, sem score positivo",
                            "kind": "canonical_neutral_boundary", "policy": "audio-master-native-boundary-preservation-v1",
                            "source_segment_id": mapping["source_segment_id"], "speaker": mapping["speaker"],
                            "maximum_seconds": maximum, "machine_confirmed": False})
    return records


def perioral_composition_summary(job_dir: Path, source: Path, count: int) -> dict:
    """A new report must come from completed perioral masks, not a legacy claim."""
    if type(count) is not int or not 1 <= count <= 30000:
        raise WorkerError("A composição perioral não possui número de janelas válido.")
    records = []
    source_hash = sha256(source)
    for index in range(1, count + 1):
        mask = job_dir / f"musetalk-perioral-source-mask-v1-{index:02d}.mkv"
        output = job_dir / f"lipsync-segmented-perioral-v1-{index:02d}.mp4"
        try:
            proof = json.loads(mask.with_suffix('.perioral.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            raise WorkerError("Falta prova física da composição perioral; relatório bloqueado.") from error
        if (not isinstance(proof, dict) or proof.get('contract') != PERIORAL_MASK_CONTRACT
                or proof.get('source_sha256') != source_hash
                or proof.get('composition_completed') is not True
                or proof.get('eyes_forehead_alpha') != 0 or type(proof.get('eyes_forehead_alpha')) is not int
                or proof.get('feather') != 'inward-only' or proof.get('human_visual_review') != 'pending'
                or type(proof.get('frame_count')) is not int or proof['frame_count'] <= 0
                or not mask.is_file() or proof.get('mask_sha256') != sha256(mask)
                or not output.is_file() or proof.get('output_sha256') != sha256(output)
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(proof.get(key, '')))
                       for key in ('model_sha256', 'landmarks_sha256'))):
            raise WorkerError("A prova física perioral não confere; relatório bloqueado.")
        records.append({'index': index, 'frame_count': proof['frame_count'],
                        **{key: proof[key] for key in ('source_sha256', 'mask_sha256', 'output_sha256',
                                                     'model_sha256', 'landmarks_sha256')}})
    return {'policy': PERIORAL_MASK_CONTRACT, 'outside_face_source': 'original-frame-clock',
            'mask_outside_tracked_box': 0, 'mask_expands_box': False, 'segments': count,
            'region': 'mouth-and-lower-nose', 'source_landmarks': 'dwpose68-pinned',
            'eyes_forehead_alpha': 0, 'human_visual_review': 'pending', 'evidence': records,
            'encoding_note': 'H.264 lossy encoding may alter decoded pixels; generated content is spatially confined.'}


def render_confirmed_talknet_segments(
    video: Path,
    audio: Path,
    job_dir: Path,
    analysis: dict,
    log: Path,
    fast_profile: bool,
) -> Path:
    """Aplica MuseTalk a todas — e somente às — janelas faciais confirmadas."""
    raw_segments = analysis.get("confirmed_face_segments") or []
    if not isinstance(raw_segments, list) or not raw_segments:
        raise WorkerError("TalkNet não forneceu trechos faciais confirmados para sincronizar.")
    if analysis.get("ambiguity") or analysis.get("ambiguous_intervals"):
        raise WorkerError("A análise facial contém ambiguidades; nenhuma janela será omitida para renderizar.")
    mapped_clock = analysis.get("timing_mode") == "audio_master"
    if mapped_clock:
        validate_audio_master_visual_binding(video, analysis)
        normalized = Path(analysis["normalized_video"])
    else:
        talknet_root = Path(os.environ.get("DUBSTUDIO_TALKNET_ROOT", "/content/TalkNet-ASD"))
        normalized = talknet_root / "demo" / video.stem / "pyavi" / "video.avi"
    root = Path(os.environ.get("DUBSTUDIO_MUSETALK_ROOT", "/content/MuseTalk"))
    if not normalized.exists() or not root.exists():
        raise WorkerError("Os arquivos do TalkNet ou MuseTalk necessários para os trechos faciais não estão disponíveis.")
    analysis["musetalk_audio_frame_clock"] = patch_musetalk_audio_frame_clock(root)
    required_weights = [
        root / "models/musetalkV15/musetalk.json", root / "models/musetalkV15/unet.pth",
        root / "models/sd-vae/config.json", root / "models/sd-vae/diffusion_pytorch_model.bin",
        root / "models/whisper/config.json", root / "models/whisper/pytorch_model.bin",
        root / "models/dwpose/dw-ll_ucoco_384.pth", root / "models/face-parse-bisent/79999_iter.pth",
    ]
    missing = [str(path) for path in required_weights if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise WorkerError(f"Pesos obrigatórios do MuseTalk ausentes: {missing}")
    normalized_meta = ffprobe(normalized)
    normalized_stream = next((item for item in normalized_meta["streams"] if item.get("codec_type") == "video"), {})
    normalized_duration = video_stream_duration(normalized)
    source_duration = video_stream_duration(video)
    # A mapped turn can be shorter than the original detected donor: identity
    # was already proved on the source and the natural audio must not be
    # lengthened merely to meet the detector's source-window minimum.
    minimum_frames = 1 if mapped_clock else 20
    bounded_segments, timeline = bound_confirmed_talknet_segments(
        raw_segments, source_duration, normalized_duration, minimum_frames=minimum_frames)
    base_video = video
    confirmed_segments = []
    native_proofs = []
    total = len(bounded_segments)
    os.environ["MPLBACKEND"] = "Agg"
    for index, raw_segment in enumerate(bounded_segments, start=1):
        segment_minimum = confirmed_face_minimum_frames(raw_segment, minimum_frames)
        try:
            start_frame = int(raw_segment.get("start_frame", -1))
            end_frame = int(raw_segment.get("end_frame", -1))
            fps = float(raw_segment.get("fps", 0))
            boxes = [tuple(int(value) for value in box) for box in raw_segment.get("boxes", [])]
        except (AttributeError, TypeError, ValueError):
            raise WorkerError(f"O trecho facial {index} não possui coordenadas válidas.") from None
        if start_frame < 0 or fps <= 0 or len(boxes) != end_frame - start_frame or len(boxes) < segment_minimum:
            raise WorkerError(f"O trecho facial {index} não cobre uma janela segura.")
        start_seconds = start_frame / fps
        segment_duration = len(boxes) / fps
        if start_seconds + segment_duration > min(normalized_duration, source_duration) + 1e-6:
            raise WorkerError(f"O trecho facial {index} excede a duração normalizada do TalkNet.")
        segment_video = job_dir / f"musetalk-confirmed-face-{index:02d}.mp4"
        segment_audio = job_dir / f"musetalk-confirmed-face-{index:02d}.wav"
        run([
            "ffmpeg", "-y", "-i", str(normalized), "-ss", f"{start_seconds:.6f}", "-t", f"{segment_duration:.6f}",
            "-map", "0:v:0", "-an", "-r", f"{fps:.6f}", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "16", "-pix_fmt", "yuv420p", str(segment_video),
        ], log=log)
        run([
            "ffmpeg", "-y", "-i", str(audio), "-ss", f"{start_seconds:.6f}", "-t", f"{segment_duration:.6f}",
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(segment_audio),
        ], log=log)
        if abs(video_stream_duration(segment_video) - segment_duration) > 0.35:
            raise WorkerError(f"A preparação do trecho facial {index} perdeu duração.")
        segment_meta = ffprobe(segment_video)
        segment_stream = next((item for item in segment_meta["streams"] if item.get("codec_type") == "video"), {})
        coordinates, reason = resample_face_boxes(
            boxes, video_frame_count(segment_video),
            int(normalized_stream.get("width") or 0), int(normalized_stream.get("height") or 0),
            int(segment_stream.get("width") or 0), int(segment_stream.get("height") or 0),
        )
        if not coordinates:
            raise WorkerError(f"As coordenadas do trecho facial {index} foram recusadas: {reason}")
        result_dir = job_dir / f"musetalk-result-{LIPSYNC_CACHE_REVISION}-face-{index:02d}"
        with (result_dir.parent / f"{segment_video.stem}.pkl").open("wb") as handle:
            pickle.dump(coordinates, handle)
        config = job_dir / f"musetalk-confirmed-face-{index:02d}.yaml"
        config.write_text(f"task_0:\n  video_path: {segment_video}\n  audio_path: {segment_audio}\n", encoding="utf-8")
        before = {path.resolve() for path in result_dir.rglob("*.mp4")} if result_dir.exists() else set()
        command = [
            os.environ.get("DUBSTUDIO_MUSETALK_PYTHON", sys.executable), "-m", "scripts.inference",
            "--inference_config", str(config), "--result_dir", str(result_dir),
            "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
            "--version", "v15", "--ffmpeg_path", shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
            "--use_saved_coord", "--saved_coord",
        ]
        if fast_profile:
            command.extend(["--use_float16", "--batch_size", "24"])
        write_progress(job_dir, 82, "Aplicando lip-sync", f"Sincronizando rosto confirmado {index} de {total}")
        result = run_streaming(command, cwd=root, log=log)
        rendered = select_musetalk_output(result_dir, before, result, f"MuseTalk não criou o trecho facial {index}.")
        assert_video_stream_coverage(rendered, segment_video)
        composition_options = {}
        if mapped_clock:
            display, native_proof = audio_master_native_display(raw_segment, analysis)
            composition_options["display_window"] = display
            native_proofs.append(native_proof)
        base_video = compose_face_patch(
            rendered, base_video, video, start_seconds, segment_duration, index, job_dir, log,
            face_boxes=coordinates, box_fps=fps,
            box_size=(int(segment_stream.get("width") or 0), int(segment_stream.get("height") or 0)),
            **composition_options,
        )
        confirmed_segments.append({"start": native_proof["start_frame"] / native_proof["fps"],
                                   "end": native_proof["end_frame"] / native_proof["fps"]}
                                  if mapped_clock else {"start": start_seconds, "end": raw_segment["end_frame"] / fps})
    preserved = []
    cursor = 0.0
    for segment in confirmed_segments:
        if segment["start"] - cursor > 1e-6:
            preserved.append({"start": cursor, "end": segment["start"], "reason": "TalkNet não confirmou rosto falante; imagem original preservada"})
        cursor = segment["end"]
    if source_duration - cursor > 1e-6:
        preserved.append({"start": cursor, "end": source_duration, "reason": "TalkNet não confirmou rosto falante; imagem original preservada"})
    if mapped_clock:
        preserved = audio_master_boundary_preservation(analysis, native_proofs)
    analysis["coordinate_source"] = "TalkNet completo em trechos faciais confirmados"
    # runner_v2 derives its exported proof directly from these frame indices.
    # They must describe exactly the same bounded boxes used by MuseTalk.
    analysis["confirmed_face_segments"] = native_proofs if mapped_clock else bounded_segments
    analysis["source_timeline"] = timeline
    spatial = perioral_composition_summary(job_dir, video, len(confirmed_segments))
    if mapped_clock:
        spatial.update(policy="perioral-mapped-source-dwpose68-inner-feather-v1",
                       outside_face_source="mapped-source-frame-clock",
                       source_video_sha256=analysis["source_video_sha256"],
                       timeline_sha256=analysis["timeline_sha256"],
                       output_base_sha256=analysis["output_base_sha256"],
                       audio_master_sha256=analysis["audio_master_sha256"])
        for record, native_proof in zip(spatial["evidence"], native_proofs):
            record.update(inference_frame_count=record["frame_count"],
                          frame_count=native_proof["end_frame"] - native_proof["start_frame"],
                          native_start_frame=native_proof["start_frame"], native_end_frame=native_proof["end_frame"],
                          native_fps=native_proof["fps"])
    analysis["spatial_composition"] = spatial
    analysis["lipsync_confirmed_segments"] = confirmed_segments
    analysis["preserved_no_speaker_visual_segments"] = preserved
    analysis["musetalk_input"] = {"video": "trechos TalkNet", "audio": "trechos do mix", "coordinate_source": analysis["coordinate_source"]}
    assert_video_stream_coverage(base_video, video)
    return base_video


def validate_audio_master_visual_binding(video: Path, analysis: dict) -> dict:
    """Bind inferred boxes to the retimed base and actual portable timeline."""
    from dubstudio_audio_master import AudioMasterError, validate_timeline
    try:
        timeline = validate_timeline(analysis.get("timeline"))
    except AudioMasterError as error:
        raise WorkerError(f"Montagem de áudio natural recusada: {error}") from error
    timeline_path = Path(str(analysis.get("timeline_path") or ""))
    normalized = Path(str(analysis.get("normalized_video") or ""))
    if (not timeline_path.is_file() or sha256(timeline_path) != analysis.get("timeline_sha256")
            or not normalized.is_file() or sha256(normalized) != analysis.get("normalized_video_sha256")
            or sha256(video) != analysis.get("output_base_sha256")
            or analysis.get("output_base_sha256") != timeline["video_base"].get("sha256")
            or analysis.get("source_video_sha256") != timeline.get("source_video_sha256")
            or analysis.get("audio_master_sha256") != timeline.get("audio_master", {}).get("sha256")):
        raise WorkerError("Fonte, áudio, vídeo montado ou relógio de lip-sync mudou; prova mapeada recusada.")
    try:
        on_disk = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise WorkerError("A linha de montagem está ilegível.") from error
    if on_disk != timeline or video_frame_count(video) != timeline["video_base"]["frame_count"]:
        raise WorkerError("A linha de montagem não corresponde aos quadros efetivamente renderizados.")
    if not analysis.get("mapped_normalization") or analysis["mapped_normalization"].get("policy") != "retimed-frame-center-fps25-v1":
        raise WorkerError("Falta o mapa canônico do vídeo montado para o MuseTalk.")
    return timeline


def musetalk(
    video: Path,
    audio: Path,
    job_dir: Path,
    analysis: dict,
    log: Path,
    require_lipsync: bool = True,
    fast_profile: bool = False,
) -> tuple[Path, list[dict]]:
    skipped = []
    coordinate_mode = lipsync_coordinate_mode(analysis, fast_profile)
    if coordinate_mode == "rejected":
        if require_lipsync:
            raise WorkerError("O detector não confirmou um rosto falante seguro para aplicar lip-sync. Nenhum vídeo sem sincronização labial será entregue.")
        skipped.append({"start": 0.0, "end": ffprobe(video)["duration"], "reason": "TalkNet encontrou baixa confiança, nenhum rosto ou múltiplos falantes visíveis"})
        output = job_dir / "video-preserved.mp4"
        run(["ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output)], log=log)
        return output, skipped
    if isinstance(analysis.get("confirmed_face_segments"), list) and analysis["confirmed_face_segments"]:
        return render_confirmed_talknet_segments(video, audio, job_dir, analysis, log, fast_profile), skipped
    if isinstance(analysis.get("confirmed_face_segment"), dict) and analysis["confirmed_face_segment"].get("boxes"):
        # A cena mudou depois do locutor confirmado. Renderize apenas a janela
        # com boca verificável; a composição preserva as imagens posteriores
        # e valida novamente a duração total antes de exportar.
        return render_confirmed_face_segment(video, audio, job_dir, analysis, log, fast_profile), skipped
    if require_lipsync and analysis.get("safe_for_single_face_lipsync") and not any(
        analysis.get(key) for key in ("fast_coordinates", "s3fd_coordinates", "locked_face_coordinates", "optical_coordinates")
    ):
        detail = str(analysis.get("confirmed_face_segments_status") or analysis.get("confirmed_face_segment_status") or "TalkNet não retornou uma janela facial segura")
        raise WorkerError(
            "TalkNet não confirmou trechos faciais suficientes para aplicar lip-sync estrito: "
            f"{detail}. Nenhum vídeo sem sincronização labial será entregue."
        )
    root = Path(os.environ.get("DUBSTUDIO_MUSETALK_ROOT", "/content/MuseTalk"))
    if not root.exists():
        raise WorkerError("MuseTalk não foi instalado no runtime.")
    analysis["musetalk_audio_frame_clock"] = patch_musetalk_audio_frame_clock(root)
    required_weights = [
        root / "models/musetalkV15/musetalk.json",
        root / "models/musetalkV15/unet.pth",
        root / "models/sd-vae/config.json",
        root / "models/sd-vae/diffusion_pytorch_model.bin",
        root / "models/whisper/config.json",
        root / "models/whisper/pytorch_model.bin",
        root / "models/whisper/preprocessor_config.json",
        root / "models/dwpose/dw-ll_ucoco_384.pth",
        root / "models/face-parse-bisent/79999_iter.pth",
        root / "models/face-parse-bisent/resnet18-5c106cde.pth",
    ]
    missing_weights = [str(path) for path in required_weights if not path.exists() or path.stat().st_size == 0]
    if missing_weights:
        raise WorkerError(f"Pesos obrigatórios do MuseTalk ausentes: {missing_weights}")
    config = job_dir / "musetalk.yaml"
    s3fd_cache_name = str(analysis.get("face_cache") or analysis.get("s3fd_cache") or "")
    optical_cache_name = str(analysis.get("optical_cache") or "")
    prefer_s3fd_coordinates = bool(
        fast_profile
        and (analysis.get("s3fd_coordinates") or analysis.get("locked_face_coordinates") or analysis.get("optical_coordinates"))
        and (s3fd_cache_name or optical_cache_name)
    )
    # Com caixas S3FD já validadas, ampliar o vídeo só piora a composição por
    # CPU e não melhora a detecção (que não será usada). Mantemos os pixels
    # originais nessa rota; o fallback do detector completo continua ampliado.
    musetalk_video, musetalk_audio = prepare_musetalk_media(
        video, audio, job_dir, log, upscale_for_detector=not prefer_s3fd_coordinates,
    )
    # O diretório inclui a revisão. Assim, uma tentativa antiga (inclusive
    # uma que gerou MP4 curto) nunca é confundida com a renderização atual.
    result_dir = job_dir / f"musetalk-result-{LIPSYNC_CACHE_REVISION}"
    use_talknet_coordinates = False
    coordinate_source = "MuseTalk completo"
    # O S3FD do TalkNet foi executado para todos os frames normalizados antes
    # de o locutor ser confirmado. Usamos suas caixas somente se a análise
    # declarou um único rosto falante e cada frame tem uma única caixa válida.
    # Assim, nenhum placeholder do detector do MuseTalk pode encolher o vídeo.
    if prefer_s3fd_coordinates:
        talknet_root = Path(os.environ.get("DUBSTUDIO_TALKNET_ROOT", "/content/TalkNet-ASD"))
        normalized = talknet_root / "demo" / video.stem / "pyavi" / "video.avi"
        try:
            normalized_meta = ffprobe(normalized)
            normalized_stream = next((item for item in normalized_meta["streams"] if item.get("codec_type") == "video"), {})
            musetalk_meta = ffprobe(musetalk_video)
            musetalk_stream = next((item for item in musetalk_meta["streams"] if item.get("codec_type") == "video"), {})
            if analysis.get("optical_coordinates"):
                optical_cache = job_dir / Path(optical_cache_name).name
                with optical_cache.open("rb") as handle:
                    tracked = pickle.load(handle)
                if not isinstance(tracked, list) or not all(isinstance(box, tuple) and len(box) == 4 for box in tracked):
                    raise WorkerError("a trilha óptica facial não é válida")
                coordinates, reason = resample_face_boxes(
                    tracked,
                    video_frame_count(musetalk_video),
                    int(normalized_stream.get("width") or 0),
                    int(normalized_stream.get("height") or 0),
                    int(musetalk_stream.get("width") or 0),
                    int(musetalk_stream.get("height") or 0),
                )
                coordinate_source = "TalkNet S3FD com rastreamento óptico validado"
            else:
                s3fd_cache = job_dir / Path(s3fd_cache_name).name
                with s3fd_cache.open("rb") as handle:
                    faces = pickle.load(handle)
                if analysis.get("s3fd_coordinates"):
                    coordinates, reason = build_s3fd_coordinate_cache(
                        faces,
                        video_frame_count(musetalk_video),
                        int(normalized_stream.get("width") or 0),
                        int(normalized_stream.get("height") or 0),
                        int(musetalk_stream.get("width") or 0),
                        int(musetalk_stream.get("height") or 0),
                    )
                    coordinate_source = "TalkNet S3FD completo reamostrado"
                else:
                    raw_box = analysis.get("locked_face_box") or []
                    if not isinstance(raw_box, list) or len(raw_box) != 4:
                        raise WorkerError("a caixa do retrato estático não é válida")
                    coordinates, reason = scale_locked_face_box(
                        tuple(int(value) for value in raw_box),
                        video_frame_count(musetalk_video),
                        int(normalized_stream.get("width") or 0),
                        int(normalized_stream.get("height") or 0),
                        int(musetalk_stream.get("width") or 0),
                        int(musetalk_stream.get("height") or 0),
                    )
                    coordinate_source = "TalkNet S3FD de retrato estático validado"
            if not coordinates:
                raise WorkerError(reason)
            # scripts.inference procura o cache no pai de result_dir usando o
            # stem do vídeo configurado, por exemplo musetalk-input.pkl.
            with (result_dir.parent / f"{musetalk_video.stem}.pkl").open("wb") as handle:
                pickle.dump(coordinates, handle)
            use_talknet_coordinates = True
            analysis["coordinate_cache"] = reason
        except (OSError, pickle.PickleError, TypeError, ValueError, WorkerError) as exc:
            analysis["coordinate_cache"] = f"cache facial validado recusado: {exc}"
    if prefer_s3fd_coordinates and not use_talknet_coordinates:
        # O cache seguro foi recusado. Para o fallback oficial, restaura a
        # entrada ampliada que dá a melhor chance ao detector DWPose; a saída
        # ainda será bloqueada se não cobrir a duração inteira.
        musetalk_video, musetalk_audio = prepare_musetalk_media(video, audio, job_dir, log, upscale_for_detector=True)
    if not use_talknet_coordinates and coordinate_mode == "talknet_cache":
        talknet_root = Path(os.environ.get("DUBSTUDIO_TALKNET_ROOT", "/content/TalkNet-ASD"))
        normalized = talknet_root / "demo" / video.stem / "pyavi" / "video.avi"
        coordinate_cache = job_dir / "video.pkl"
        if normalized.exists() and coordinate_cache.exists() and coordinate_cache.stat().st_size:
            musetalk_video = normalized
            use_talknet_coordinates = True
            coordinate_source = "TalkNet completo"
    if not use_talknet_coordinates and fast_profile:
        # Um track do TalkNet pode confirmar o locutor, porém ter menos caixas
        # que os frames normalizados. Nesse caso não inventamos coordenadas e
        # não caímos para renderização por CPU: o MuseTalk faz a detecção
        # completa e mantém sua inferência float16 na GPU.
        analysis["coordinate_cache"] = "coordenadas TalkNet incompletas; usando pré-processamento completo do MuseTalk na GPU"
    if use_talknet_coordinates:
        write_progress(job_dir, 82, "Aplicando lip-sync", f"Usando {coordinate_source} para todos os quadros")
    # O relatório precisa registrar a rota que de fato renderizou o resultado,
    # inclusive quando o cache S3FD validado evita uma segunda detecção.
    analysis["coordinate_source"] = coordinate_source
    config.write_text(f"task_0:\n  video_path: {musetalk_video}\n  audio_path: {musetalk_audio}\n", encoding="utf-8")
    musetalk_python = os.environ.get("DUBSTUDIO_MUSETALK_PYTHON", sys.executable)
    # Colab exports an inline Matplotlib backend that is not installed inside
    # the isolated MuseTalk environment.  A headless backend is sufficient.
    os.environ["MPLBACKEND"] = "Agg"
    before = {path.resolve() for path in result_dir.rglob("*.mp4")} if result_dir.exists() else set()
    command = [
        musetalk_python, "-m", "scripts.inference", "--inference_config", str(config), "--result_dir", str(result_dir),
        "--unet_model_path", "models/musetalkV15/unet.pth", "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15", "--ffmpeg_path", shutil.which("ffmpeg") or "/usr/bin/ffmpeg",
    ]
    # A A100 executa a inferência do MuseTalk de forma muito mais eficiente
    # em float16.  Lotes maiores diminuem a espera entre a GPU e o preparo de
    # frames; a detecção de rosto continua obrigatória e não há atalho visual.
    if fast_profile:
        command.extend(["--use_float16", "--batch_size", "24"])
    # Mantém o cache bruto de coordenadas para auditoria se o DWPose não
    # reconhecer todos os quadros. Não o reutilizamos sem validação completa.
    if not use_talknet_coordinates:
        command.append("--saved_coord")
    if use_talknet_coordinates:
        # O MuseTalk aceita um cache de coords. Ele vem apenas do rosto que o
        # TalkNet confirmou como falante e elimina uma segunda detecção lenta
        # quadro a quadro, preservando o modo estrito de lip-sync.
        command.extend(["--use_saved_coord", "--saved_coord"])

    last_progress = 82

    def report_musetalk_progress(line: str) -> None:
        nonlocal last_progress
        normalized = line.lower()
        stage = "Aplicando lip-sync"
        message = (
            "Preparando quadros com as coordenadas TalkNet confirmadas"
            if use_talknet_coordinates
            else "Detectando o rosto completo com MuseTalk na GPU"
        )
        target = 83
        if any(token in normalized for token in ("inference", "batch", "generating")):
            target, message = 86, "Inferência labial em execução na GPU"
        elif any(token in normalized for token in ("blend", "paste", "combine", "merge", "padding generated", "padding")):
            target, message = 90, "Compondo o rosto sincronizado no vídeo"
        elif any(token in normalized for token in ("ffmpeg", "mux", "encode", "video generation")):
            target, stage, message = 92, "Exportando lip-sync", "Codificando o vídeo sincronizado"
        match = re.search(r"(\d+)\s*/\s*(\d+)", normalized)
        if match:
            current, total = (int(value) for value in match.groups())
            if total >= 20 and 0 <= current <= total:
                # O contador é emitido pelo próprio MuseTalk para uma etapa
                # concluída; usamos apenas esse avanço medido dentro da faixa
                # reservada ao lip-sync.
                target = max(target, min(92, 83 + round((current / total) * 9)))
        if target > last_progress:
            last_progress = target
            write_progress(job_dir, target, stage, message)

    result = run_streaming(command, cwd=root, log=log, on_output=report_musetalk_progress)
    rendered = select_musetalk_output(result_dir, before, result, "MuseTalk terminou sem criar o vídeo esperado.")
    assert_video_stream_coverage(rendered, video)
    analysis["musetalk_input"] = {
        "video": musetalk_video.name,
        "audio": musetalk_audio.name,
        "video_duration": round(video_stream_duration(musetalk_video), 3),
        "coordinate_source": coordinate_source,
    }
    return rendered, skipped


def finalize(video: Path, mix: Path, source: Path, job_dir: Path, duration: float, log: Path,
             timing_mode: str = "source", timeline: dict | None = None) -> Path:
    master_mode = timing_mode == "audio_master"
    if master_mode:
        from dubstudio_audio_master import AudioMasterError, validate_timeline
        try:
            validated = validate_timeline(timeline)
        except AudioMasterError as error:
            raise WorkerError(f"Exportação do áudio natural bloqueada: {error}") from error
        expected = validated["sample_count"] / validated["sample_rate"]
        if (abs(duration - expected) > 1 / validated["sample_rate"]
                or sha256(source) != validated.get("source_video_sha256")
                or video_frame_count(video) != validated["video_base"]["frame_count"]):
            raise WorkerError("A exportação diverge do relógio do áudio natural ou perdeu quadros.")
        # Physical PCM is checked before encoding. No guessed container
        # duration or -t can hide a missing last speech sample.
        try:
            import wave
            with wave.open(str(mix), "rb") as pcm:
                if (pcm.getframerate() != validated["sample_rate"]
                        or pcm.getnframes() != validated["sample_count"]):
                    raise WorkerError("A mixagem perdeu amostras do áudio mestre; exportação bloqueada.")
        except (wave.Error, EOFError) as error:
            raise WorkerError("A mixagem final do áudio natural precisa ser PCM WAV íntegro.") from error
    source_meta = ffprobe(source)
    stream = next(item for item in source_meta["streams"] if item.get("codec_type") == "video")
    width, height = int(stream["width"]), int(stream["height"])
    max_width, max_height = ((1920, 1080) if width >= height else (1080, 1920))
    scale = f"scale='min({max_width},iw)':'min({max_height},ih)':force_original_aspect_ratio=decrease,setsar=1"
    output = job_dir / "dublagem-final.mp4"
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    cpu_args = ["-c:v", "libx264", "-preset", "medium", "-crf", "19"]

    def encode(video_args: list[str]) -> None:
        run([
            "ffmpeg", "-y", "-i", str(video), "-i", str(mix), "-map", "0:v:0", "-map", "1:a:0",
            "-vf", scale, *video_args, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
            "-ac", "2", *([] if master_mode else ["-t", str(duration)]),
            "-movflags", "+faststart", str(output),
        ], log=log)

    nvenc_available = encoders.returncode == 0 and "h264_nvenc" in encoders.stdout
    if not nvenc_available:
        encode(cpu_args)
    else:
        try:
            encode(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0"])
        except WorkerError as exc:
            # Algumas imagens anunciam o encoder NVENC, mas a build do FFmpeg não
            # consegue abrir uma sessão no driver do Pod. O resultado de MuseTalk
            # já está pronto neste ponto: repetir somente a exportação em libx264
            # é seguro e evita descartar toda a dublagem/lip-sync validada.
            failure = str(exc).lower()
            nvenc_failure = any(marker in failure for marker in (
                "h264_nvenc", "openencodesessionex", "no capable devices", "unsupported device",
            ))
            if not nvenc_failure:
                raise
            output.unlink(missing_ok=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write("\nNVENC indisponível neste Pod; repetindo apenas a exportação final com libx264.\n")
            encode(cpu_args)
    assert_video_stream_coverage(output, video if master_mode else source)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--checkpoint-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.package_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or not manifest.get("authorization", {}).get("voice_clone_confirmed"):
        raise WorkerError("Manifesto incompatível ou sem autorização de clonagem vocal.")
    input_video = args.package_root / manifest["input_file"]
    if sha256(input_video) != manifest["input_sha256"]:
        raise WorkerError("O checksum do vídeo não corresponde ao manifesto.")
    job_dir = args.checkpoint_root / manifest["job_id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    local_video = job_dir / input_video.name
    if not local_video.exists() or sha256(local_video) != manifest["input_sha256"]:
        shutil.copy2(input_video, local_video)
    checkpoints = Checkpoints(job_dir / "checkpoint.json")
    lipsync_cache_reset = reset_incompatible_lipsync_cache(checkpoints, job_dir)
    log = job_dir / "pipeline.log"
    performance_profile = str(manifest.get("performance_profile", "fast_lipsync"))
    fast_profile = performance_profile == "fast_lipsync"
    stage_seconds: dict[str, float] = {}
    cache_usage: dict[str, bool] = {}
    report = {
        "schema_version": SCHEMA_VERSION, "job_id": manifest["job_id"], "started_at": int(time.time()),
        "source_language": manifest["source_language"], "target_language": manifest["target_language"],
        "models": {"asr": "faster-whisper distil-large-v3" if fast_profile else "faster-whisper large-v3", "separation": "Demucs htdemucs CUDA", "diarization": "SpeechBrain ECAPA", "translation": "MADLAD-400 3B", "active_speaker": "TalkNet", "lipsync": "MuseTalk 1.5"},
        "performance_profile": performance_profile,
        "lipsync_cache_reset": lipsync_cache_reset,
    }
    write_progress(job_dir, 4, "Validando vídeo", "Conferindo vídeo, áudio, GPU e espaço disponível")
    meta = ffprobe(local_video)
    duration = meta["duration"]
    if duration <= 0 or duration > 1200:
        raise WorkerError("Duração inválida ou acima de 20 minutos.")
    try:
        gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"]).stdout.strip()
    except Exception as exc:
        raise WorkerError("Este pipeline requer um runtime Colab com GPU NVIDIA.") from exc
    free_bytes = shutil.disk_usage(job_dir).free
    if free_bytes < 15 * 1024**3:
        raise WorkerError("São necessários pelo menos 15 GB livres no Drive/runtime.")
    report["preflight"] = {"gpu": gpu, "free_gb": round(free_bytes / 1024**3, 2)}
    checkpoints.mark("preflight")

    vocals, background = job_dir / "vocals.wav", job_dir / "background.wav"
    cache_usage["separation"] = checkpoints.done("separation", [vocals, background])
    write_progress(job_dir, 12, "Separando áudio", "Reutilizando separação de áudio" if cache_usage["separation"] else "Separando voz, música e ambiente na GPU")
    stage_started = time.monotonic()
    if not cache_usage["separation"]:
        demucs_vocals, demucs_background = separate_audio(local_video, job_dir, log)
        shutil.copy2(demucs_vocals, vocals); shutil.copy2(demucs_background, background)
        checkpoints.mark("separation")
    stage_seconds["separation"] = round(time.monotonic() - stage_started, 3)

    transcript_json = job_dir / "transcript.json"
    cache_usage["transcription"] = checkpoints.done("transcription", [transcript_json])
    write_progress(job_dir, 24, "Transcrevendo", "Reutilizando transcrição" if cache_usage["transcription"] else "Reconhecendo a fala na GPU")
    stage_started = time.monotonic()
    if not cache_usage["transcription"]:
        segments, detected_language, language_confidence, asr_model = transcribe(vocals, manifest["source_language"], job_dir, fast_profile)
        # Em áudio difícil, o modo rápido reprocessa com o modelo de maior
        # precisão antes de continuar. Assim não entrega uma transcrição fraca.
        transcript_data = json.loads(transcript_json.read_text(encoding="utf-8"))
        if fast_profile and (language_confidence < 0.80 or float(transcript_data.get("word_probability", 0.0)) < 0.60):
            write_progress(job_dir, 27, "Revisando transcrição", "Baixa confiança detectada; aplicando reconhecimento de maior precisão")
            segments, detected_language, language_confidence, asr_model = transcribe(vocals, manifest["source_language"], job_dir, False)
        checkpoints.mark("transcription")
    else:
        data = json.loads(transcript_json.read_text(encoding="utf-8")); segments = data["segments"]
        detected_language, language_confidence, asr_model = data["language"], float(data.get("probability", 0)), str(data.get("model", "large-v3"))
    stage_seconds["transcription"] = round(time.monotonic() - stage_started, 3)
    report["detected_language"] = {"code": detected_language, "confidence": language_confidence}
    report["asr"] = {"model": asr_model, "fast_profile": fast_profile}

    diarization_json = job_dir / "diarization.json"
    cache_usage["diarization"] = checkpoints.done("diarization", [diarization_json])
    write_progress(job_dir, 34, "Identificando vozes", "Reutilizando identificação de locutores" if cache_usage["diarization"] else "Separando locutores na GPU")
    stage_started = time.monotonic()
    if not cache_usage["diarization"]:
        segments = diarize(vocals, segments, manifest["speaker_count"], job_dir)
        checkpoints.mark("diarization")
    else:
        segments = json.loads(diarization_json.read_text(encoding="utf-8"))["segments"]
    stage_seconds["diarization"] = round(time.monotonic() - stage_started, 3)

    segments_json = job_dir / "segments.json"
    cache_usage["translation"] = checkpoints.done("translation", [segments_json])
    write_progress(job_dir, 44, "Traduzindo", "Reutilizando tradução revisada" if cache_usage["translation"] else "Traduzindo mantendo tempos e locutores")
    stage_started = time.monotonic()
    if not cache_usage["translation"]:
        translation_source = detected_language if manifest["source_language"] == "auto" else manifest["source_language"]
        segments = translate_segments(segments, translation_source, manifest["target_language"], job_dir)
        checkpoints.mark("translation")
    else:
        segments = json.loads(segments_json.read_text(encoding="utf-8"))["segments"]
    stage_seconds["translation"] = round(time.monotonic() - stage_started, 3)
    original_srt, translated_srt = job_dir / "transcript-original.srt", job_dir / "translation.srt"
    write_srt(original_srt, segments, "text"); write_srt(translated_srt, segments, "translated_text")

    dub = job_dir / "dub-dialogue.wav"
    voice_proof_path = job_dir / "voice-proof.json"
    cache_usage["synthesis"] = checkpoints.done("synthesis", [dub, voice_proof_path])
    write_progress(job_dir, 57, "Gerando vozes", "Reutilizando vozes geradas" if cache_usage["synthesis"] else "Criando a dublagem com a referência autorizada")
    stage_started = time.monotonic()
    if not cache_usage["synthesis"]:
        dub = synthesize(
            segments,
            manifest["target_language"],
            vocals,
            job_dir,
            report,
            bool(manifest.get("require_authorized_clone", False)),
        )
        write_json(voice_proof_path, {
            "voice_engine": report.get("voice_engine", ""),
            "voice_clone": report.get("voice_clone", {}),
        })
        write_json(segments_json, {"target_language": manifest["target_language"], "segments": segments})
        checkpoints.mark("synthesis")
    else:
        try:
            voice_proof = json.loads(voice_proof_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WorkerError("A prova da voz gerada em cache não pôde ser validada.") from exc
        report["voice_engine"] = voice_proof.get("voice_engine", "")
        report["voice_clone"] = voice_proof.get("voice_clone", {})
    if bool(manifest.get("require_authorized_clone", False)):
        voice_clone = report.get("voice_clone")
        if (
            not str(report.get("voice_engine") or "").startswith("Chatterbox Multilingual V3 authorized clone")
            or not isinstance(voice_clone, dict)
            or voice_clone.get("confirmed") is not True
        ):
            raise WorkerError("A voz em cache não confirma a clonagem autorizada exigida; o vídeo não será exportado.")
    stage_seconds["synthesis"] = round(time.monotonic() - stage_started, 3)
    mix = job_dir / "dub-mix.wav"
    cache_usage["mix"] = checkpoints.done("mix", [mix])
    write_progress(job_dir, 67, "Mixando áudio", "Reutilizando mixagem" if cache_usage["mix"] else "Preservando música e efeitos no áudio final")
    stage_started = time.monotonic()
    if not cache_usage["mix"]:
        mix = mix_audio(background, dub, duration, job_dir, log)
        checkpoints.mark("mix")
    stage_seconds["mix"] = round(time.monotonic() - stage_started, 3)

    analysis_path = job_dir / "active-speaker.json"
    cache_usage["active_speaker"] = checkpoints.done("active_speaker", [analysis_path])
    write_progress(job_dir, 74, "Confirmando rosto falante", "Reutilizando análise de falante" if cache_usage["active_speaker"] else "Analisando rosto e voz antes do lip-sync")
    stage_started = time.monotonic()
    if not cache_usage["active_speaker"]:
        analysis = talknet_analysis(local_video, job_dir, log)
        write_json(analysis_path, analysis); checkpoints.mark("active_speaker")
    else:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    stage_seconds["active_speaker"] = round(time.monotonic() - stage_started, 3)
    report["active_speaker"] = analysis

    synced = job_dir / "synced.mp4"
    skipped_path = job_dir / "skipped-lipsync.json"
    cache_usage["lipsync"] = checkpoints.done("lipsync", [synced, skipped_path])
    lipsync_message = "Reutilizando renderização labial" if cache_usage["lipsync"] else "Alterando somente o rosto confirmado"
    if not cache_usage["lipsync"] and analysis.get("confirmed_face_segments"):
        lipsync_message = "Sincronizando todos os trechos faciais confirmados; preservando cenas sem locutor"
    elif not cache_usage["lipsync"] and analysis.get("confirmed_face_segment"):
        lipsync_message = "Sincronizando o trecho facial confirmado; preservando cenas sem locutor"
    elif not cache_usage["lipsync"] and analysis.get("optical_coordinates"):
        lipsync_message = "Usando a trilha facial dinâmica validada em todos os quadros"
    elif not cache_usage["lipsync"] and analysis.get("safe_for_single_face_lipsync"):
        lipsync_message = f"Trechos faciais recusados: {analysis.get('confirmed_face_segments_status') or analysis.get('confirmed_face_segment_status') or 'motivo não informado'}"
    elif not cache_usage["lipsync"] and analysis.get("safe_for_single_face_lipsync") and not analysis.get("s3fd_coordinates") and not analysis.get("locked_face_coordinates"):
        lipsync_message = f"Trilha facial dinâmica recusada: {analysis.get('optical_status') or 'motivo não informado'}"
    write_progress(job_dir, 82, "Aplicando lip-sync", lipsync_message)
    stage_started = time.monotonic()
    if not cache_usage["lipsync"]:
        rendered, skipped = musetalk(
            local_video,
            mix,
            job_dir,
            analysis,
            log,
            bool(manifest.get("require_lipsync", True)),
            fast_profile,
        )
        shutil.copy2(rendered, synced); write_json(skipped_path, skipped)
        # A decisão de coordenadas pode ter sido refinada pelo MuseTalk. Ela
        # precisa ir para o relatório e para o cache, não ficar só em memória.
        write_json(analysis_path, analysis)
        checkpoints.mark("lipsync")
    else:
        skipped = json.loads(skipped_path.read_text(encoding="utf-8"))
    stage_seconds["lipsync"] = round(time.monotonic() - stage_started, 3)
    final = job_dir / "dublagem-final.mp4"
    cache_usage["finalize"] = checkpoints.done("finalize", [final])
    write_progress(job_dir, 94, "Exportando vídeo", "Reutilizando exportação final" if cache_usage["finalize"] else "Codificando o vídeo final")
    stage_started = time.monotonic()
    if not cache_usage["finalize"]:
        final = finalize(synced, mix, local_video, job_dir, duration, log); checkpoints.mark("finalize")
    stage_seconds["finalize"] = round(time.monotonic() - stage_started, 3)
    final_meta = ffprobe(final)
    audio_stream = next((item for item in final_meta["streams"] if item.get("codec_type") == "audio"), {})
    speeds = [float(item.get("speed", 1.0)) for item in segments]
    speaker_confidences = [float(item.get("speaker_confidence", 0.0)) for item in segments]
    report.update({
        "finished_at": int(time.time()), "speaker_count": len({item["speaker"] for item in segments}),
        "segments": len(segments), "no_lipsync_segments": skipped,
        "lipsync": {
            "required": bool(manifest.get("require_lipsync", True)),
            "confirmed": not bool(skipped),
            "precision": "float16" if fast_profile else "float32",
            "batch_size": 24 if fast_profile else 8,
            "coordinate_source": str(analysis.get("coordinate_source") or "MuseTalk completo"),
            "confirmed_segments": analysis.get("lipsync_confirmed_segments", []),
            "preserved_no_speaker_visual_segments": analysis.get("preserved_no_speaker_visual_segments", []),
        },
        "translation": {
            "target_language": manifest["target_language"],
            "segments_validated": len(segments),
        },
        "cache": cache_usage,
        "stages_seconds": stage_seconds,
        "metrics": {
            "processing_seconds": int(time.time()) - report["started_at"],
            "speed_min": round(min(speeds), 4), "speed_max": round(max(speeds), 4),
            "speed_average": round(sum(speeds) / len(speeds), 4),
            "speaker_confidence_min": round(min(speaker_confidences), 4),
            "speaker_confidence_average": round(sum(speaker_confidences) / len(speaker_confidences), 4),
        },
        "summary": {
            "duration": final_meta["duration"],
            "timing_tolerance_seconds": abs(final_meta["duration"] - duration),
            "audio_channels": audio_stream.get("channels", 0),
            "normalized_lufs_target": -14,
        },
    })
    report["active_speaker"] = analysis
    report_path = job_dir / "report.json"; write_json(report_path, report)
    write_progress(job_dir, 98, "Empacotando resultado", "Gerando relatório, legendas e vídeo final")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for path in [final, report_path, segments_json, original_srt, translated_srt, log]:
            if path.exists():
                archive.write(path, path.name)
    checkpoints.mark("package")
    write_progress(job_dir, 100, "Resultado pronto", "Dublagem e lip-sync concluídos")
    print(f"RESULTADO={args.output}")


if __name__ == "__main__":
    main()
