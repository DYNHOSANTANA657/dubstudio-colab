"""Natural-voice sample clock and explicit, speaker-bound video editing.

This module never changes speech speed. Source transcript times remain immutable;
each language receives a separate output clock. Repetition is forward at the
source frame rate and occurs before lip-sync, never on a previously rendered mouth.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import wave
from fractions import Fraction
from pathlib import Path


TIMING_POLICY = "audio-master-speaker-turns-v1"
MAPPED_SPATIAL_POLICY = "perioral-mapped-source-dwpose68-inner-feather-v1"


class AudioMasterError(RuntimeError):
    pass


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def source_segments_sha256(segments: list[dict]) -> str:
    """Fingerprint approved source fields, without replacing their timestamps."""
    _source_segments(segments)
    return _json_hash([{key: item.get(key) for key in ("id", "start", "end", "speaker", "text")}
                      for item in segments])


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise AudioMasterError(f"{label} precisa ser um inteiro >= {minimum}.")
    return value


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise AudioMasterError(f"{label} precisa ser um tempo finito não negativo.")
    return float(value)


def _source_segments(value: object) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise AudioMasterError("A montagem requer os turnos originais aprovados.")
    identities = set()
    for item in value:
        if not isinstance(item, dict):
            raise AudioMasterError("Turno original inválido.")
        identity = item.get("id")
        if isinstance(identity, bool) or not isinstance(identity, (str, int)) or str(identity) == "" or str(identity) in identities:
            raise AudioMasterError("Cada turno precisa de identificador único.")
        identities.add(str(identity))
        if not isinstance(item.get("speaker"), str) or not item["speaker"].strip():
            raise AudioMasterError("Cada turno precisa de uma identidade vocal confirmada.")
        start, end = _seconds(item.get("start"), "Início original"), _seconds(item.get("end"), "Fim original")
        if end <= start:
            raise AudioMasterError("Turno original vazio ou invertido.")
    return value


def build_timeline(source_segments: list[dict], pcm_lengths: list[int] | dict,
                   sample_rate: int = 48000) -> dict:
    """Serialise natural PCM clips in source turn order, including A/B/A.

    Overlapping source speech is sequential, as explicitly chosen by the user.
    ``pcm_lengths`` describes complete natural clips, not available video slots.
    """
    segments = _source_segments(source_segments)
    _integer(sample_rate, "Taxa PCM", 1)
    if sample_rate > 192000:
        raise AudioMasterError("Taxa PCM não suportada.")
    if isinstance(pcm_lengths, dict):
        lengths = []
        for item in segments:
            identity = item["id"]
            if identity in pcm_lengths:
                lengths.append(pcm_lengths[identity])
            elif str(identity) in pcm_lengths:
                lengths.append(pcm_lengths[str(identity)])
            else:
                raise AudioMasterError("Falta áudio natural para um turno.")
        if len(pcm_lengths) != len(segments):
            raise AudioMasterError("Há áudios sem turno correspondente.")
    elif isinstance(pcm_lengths, list) and len(pcm_lengths) == len(segments):
        lengths = list(pcm_lengths)
    else:
        raise AudioMasterError("Os comprimentos PCM não correspondem aos turnos originais.")
    order = sorted(range(len(segments)), key=lambda index: (segments[index]["start"], index))
    cursor, turns = 0, []
    for index in order:
        count = _integer(lengths[index], "Amostras da fala completa", 1)
        source = segments[index]
        turns.append({"source_segment_id": source["id"], "speaker": source["speaker"],
                      "source_start": source["start"], "source_end": source["end"],
                      "output_start_sample": cursor, "output_end_sample": cursor + count,
                      "pcm_samples": count})
        cursor += count
    value = {"timing_mode": "audio_master", "policy": TIMING_POLICY,
             "sample_rate": sample_rate, "sample_count": cursor,
             "source_segments": copy.deepcopy(segments),
             "source_segments_sha256": source_segments_sha256(segments), "turns": turns}
    return validate_timeline(value, source_segments)


def validate_timeline(value: object, source_segments: list[dict] | None = None) -> dict:
    """Reject gaps, dropped/duplicated speech, clock drift and changed sources."""
    if not isinstance(value, dict) or value.get("timing_mode") != "audio_master" or value.get("policy") != TIMING_POLICY:
        raise AudioMasterError("Contrato de áudio natural desconhecido; prova antiga recusada.")
    rate = _integer(value.get("sample_rate"), "Taxa da linha de áudio", 1)
    if rate > 192000:
        raise AudioMasterError("Taxa PCM não suportada.")
    total = _integer(value.get("sample_count"), "Total de amostras", 1)
    segments = _source_segments(value.get("source_segments"))
    if value.get("source_segments_sha256") != source_segments_sha256(segments):
        raise AudioMasterError("O hash dos turnos originais não corresponde à revisão.")
    if source_segments is not None and source_segments_sha256(source_segments) != value["source_segments_sha256"]:
        raise AudioMasterError("A revisão original foi modificada durante a montagem.")
    expected = sorted(enumerate(segments), key=lambda pair: (pair[1]["start"], pair[0]))
    turns = value.get("turns")
    if not isinstance(turns, list) or len(turns) != len(expected):
        raise AudioMasterError("Faltam falas ou há turnos extras na linha final.")
    cursor = 0
    for turn, (_, source) in zip(turns, expected):
        if not isinstance(turn, dict) or any(turn.get(key) != source.get(source_key) for key, source_key in (
                ("source_segment_id", "id"), ("speaker", "speaker"), ("source_start", "start"), ("source_end", "end"))):
            raise AudioMasterError("A pessoa, a ordem ou os tempos originais de um turno mudaram.")
        start = _integer(turn.get("output_start_sample"), "Início PCM")
        end = _integer(turn.get("output_end_sample"), "Fim PCM", 1)
        count = _integer(turn.get("pcm_samples"), "Amostras completas", 1)
        if start != cursor or end != start + count:
            raise AudioMasterError("A fala final possui lacuna, sobreposição ou amostras perdidas.")
        cursor = end
    if cursor != total:
        raise AudioMasterError("A última fala não alcança o fim do áudio mestre.")
    master = value.get("audio_master")
    if master is not None:
        if (not isinstance(master, dict) or master.get("sample_rate") != rate
                or master.get("sample_count") != total or not re.fullmatch(r"[0-9a-f]{64}", str(master.get("sha256", "")))
                or not isinstance(master.get("path"), str) or not master["path"]):
            raise AudioMasterError("O arquivo de áudio mestre não corresponde à sua linha PCM.")
    mapping, base = value.get("visual_mapping"), value.get("video_base")
    if mapping is not None or base is not None:
        if not isinstance(base, dict) or not isinstance(mapping, list) or len(mapping) != len(turns):
            raise AudioMasterError("A montagem de vídeo está incompleta.")
        fps = Fraction(_integer(base.get("fps_num"), "FPS numerador", 1), _integer(base.get("fps_den"), "FPS denominador", 1))
        frame_count = _integer(base.get("frame_count"), "Quadros finais", 1)
        if frame_count != math.ceil(Fraction(total, rate) * fps):
            raise AudioMasterError("O vídeo não termina no último quadro do áudio mestre.")
        source_count = _integer(value.get("source_video", {}).get("frame_count"), "Quadros originais", 1)
        frame_cursor = 0
        for turn, item in zip(turns, mapping):
            if not isinstance(item, dict) or item.get("source_segment_id") != turn["source_segment_id"] or item.get("speaker") != turn["speaker"]:
                raise AudioMasterError("Uma imagem foi associada a outro locutor.")
            start = _integer(item.get("output_start_frame"), "Quadro de início final")
            end = _integer(item.get("output_end_frame"), "Quadro de fim final", 1)
            donor_start = _integer(item.get("source_start_frame"), "Quadro de início original")
            donor_end = _integer(item.get("source_end_frame"), "Quadro de fim original", 1)
            if (start != frame_cursor or end <= start or donor_end <= donor_start or donor_end > source_count
                    or item.get("method") != ("repeat" if end - start > donor_end - donor_start else "trim")):
                raise AudioMasterError("Corte/repetição de vídeo inválido ou fora da fonte.")
            source_fps = Fraction(_integer(value["source_video"].get("fps_num"), "FPS original", 1),
                                  _integer(value["source_video"].get("fps_den"), "FPS original", 1))
            if source_fps != fps:
                raise AudioMasterError("A montagem mudou a velocidade/taxa original dos quadros.")
            allowed_first = max(0, math.ceil(Fraction(str(turn["source_start"])) * fps - Fraction(1, 2)))
            allowed_last = min(source_count, math.ceil(Fraction(str(turn["source_end"])) * fps - Fraction(1, 2)))
            if donor_start < allowed_first or donor_end > allowed_last:
                raise AudioMasterError("O trecho de imagem não pertence ao turno da pessoa correta.")
            if item.get("face_confirmation") == "no_face_confirmed":
                if item.get("speaker_confirmation") != "narration" or item.get("source_face_track_id") is not None:
                    raise AudioMasterError("A narração não possui prova de ausência de rosto.")
            elif (type(item.get("source_face_track_id")) is not int or item["source_face_track_id"] < 0
                  or item.get("face_confirmation") not in {"detected", "user_selected"}
                  or item.get("speaker_confirmation") not in {"active_speaker", "user_selected", "external_audio"}):
                raise AudioMasterError("A identidade facial do turno não foi confirmada.")
            # Every switch is the nearest source-FPS frame to the natural PCM
            # switch. Reassigning frame spans cannot change who is speaking.
            if abs(Fraction(start, 1) / fps - Fraction(turn["output_start_sample"], rate)) > 1 / fps:
                raise AudioMasterError("O corte de pessoa se afastou da fala natural.")
            frame_cursor = end
        if frame_cursor != frame_count:
            raise AudioMasterError("A montagem não cobre todos os quadros finais.")
    return value


def _rate(metadata: dict) -> Fraction:
    stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), metadata)
    try:
        rate = Fraction(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or stream.get("fps")))
    except (ValueError, TypeError, ZeroDivisionError):
        raise AudioMasterError("Não foi possível confirmar o FPS original.") from None
    if not 1 <= rate <= 120:
        raise AudioMasterError("FPS original fora do intervalo suportado.")
    if stream.get("avg_frame_rate") and stream.get("r_frame_rate"):
        try:
            if Fraction(str(stream["avg_frame_rate"])) != Fraction(str(stream["r_frame_rate"])):
                raise AudioMasterError("Vídeo com FPS variável precisa de normalização explícita antes da montagem.")
        except (ValueError, TypeError, ZeroDivisionError):
            raise AudioMasterError("Taxa original inválida.") from None
    return rate


def enrich_video_mapping(timeline: dict, metadata: dict, analysis: dict | None,
                         require_faces: bool = True) -> tuple[dict, list[dict]]:
    """Select only donor frames confirmed as the current turn's person."""
    validate_timeline(timeline)
    value = copy.deepcopy(timeline)
    stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), metadata)
    fps = _rate(metadata)
    frame_count = int(stream.get("nb_frames") or metadata.get("frame_count") or 0)
    width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
    if min(frame_count, width, height) <= 0:
        raise AudioMasterError("Fonte sem dimensões/quadros verificáveis para montagem.")
    groups = (analysis or {}).get("confirmed_face_segments", [])
    if (analysis or {}).get("ambiguity") or (analysis or {}).get("ambiguous_intervals"):
        raise AudioMasterError("Há associação facial ambígua; corrija rosto e voz antes de montar.")
    if not isinstance(groups, list):
        raise AudioMasterError("Análise facial ilegível.")
    if require_faces and not groups:
        raise AudioMasterError("Falta confirmar qual pessoa corresponde a cada voz.")
    if not require_faces and not groups and (analysis or {}).get("narration_no_face_confirmed") is not True:
        raise AudioMasterError("Narração sem rosto precisa ser explicitamente confirmada.")
    result, group_bindings = [], []
    rate, sample_count = value["sample_rate"], value["sample_count"]
    output_count = math.ceil(Fraction(sample_count, rate) * fps)
    cursor = 0
    for index, turn in enumerate(value["turns"]):
        candidates = []
        for group in groups:
            if not isinstance(group, dict) or group.get("speaker") != turn["speaker"]:
                continue
            canonical = _rate({"fps": group.get("fps")})
            begin = _integer(group.get("start_frame"), "Início facial")
            end = _integer(group.get("end_frame"), "Fim facial", 1)
            boxes = group.get("boxes")
            if not isinstance(boxes, list) or len(boxes) != end - begin or not boxes:
                raise AudioMasterError("As caixas faciais não cobrem o trecho confirmado.")
            if group.get("speaker_confirmation") not in {"active_speaker", "user_selected", "external_audio"}:
                raise AudioMasterError("A trilha facial não confirmou a identidade da voz.")
            if group.get("face_confirmation") not in {"detected", "user_selected"}:
                raise AudioMasterError("O rosto correspondente não foi confirmado.")
            _integer(group.get("track_id"), "Identidade da trilha facial")
            left = max(Fraction(str(turn["source_start"])), Fraction(begin, 1) / canonical)
            right = min(Fraction(str(turn["source_end"])), Fraction(end, 1) / canonical)
            first = max(0, math.ceil(left * fps - Fraction(1, 2)))
            last = min(frame_count, math.ceil(right * fps - Fraction(1, 2)))
            if last > first:
                candidates.append((last - first, first, last, group))
        if candidates:
            _, first, last, group = max(candidates, key=lambda item: (item[0], -item[1]))
            # The frame-center conversion must never select a coordinate from
            # a neighbouring person's canonical track.
            for source_frame in range(first, last):
                moment = Fraction(2 * source_frame + 1, 2) / fps
                face_frame = math.floor(moment * Fraction(str(group["fps"])))
                if not group["start_frame"] <= face_frame < group["end_frame"]:
                    raise AudioMasterError("O limite de cena não confirmou o rosto no quadro original.")
                box = group["boxes"][face_frame - group["start_frame"]]
                if (not isinstance(box, (list, tuple)) or len(box) != 4
                        or any(type(item) is not int for item in box)
                        or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
                    raise AudioMasterError("Caixa facial original inválida; nenhum outro rosto será usado.")
            confirmation = {key: group.get(key) for key in ("track_id", "face_confirmation", "speaker_confirmation", "coordinate_source")}
            group_bindings.append(copy.deepcopy(group))
        elif not require_faces and (analysis or {}).get("narration_no_face_confirmed") is True:
            first = max(0, math.ceil(Fraction(str(turn["source_start"])) * fps - Fraction(1, 2)))
            last = min(frame_count, math.ceil(Fraction(str(turn["source_end"])) * fps - Fraction(1, 2)))
            confirmation = {"track_id": None, "face_confirmation": "no_face_confirmed", "speaker_confirmation": "narration", "coordinate_source": "narração sem rosto confirmada"}
            group_bindings.append({})
        else:
            raise AudioMasterError(f"Nenhum rosto seguro corresponde à voz {turn['speaker']} no turno {turn['source_segment_id']}.")
        boundary = output_count if index == len(value["turns"]) - 1 else math.floor(Fraction(turn["output_end_sample"], rate) * fps + Fraction(1, 2))
        if last <= first or boundary <= cursor:
            raise AudioMasterError("A fala/trecho é curto demais para um corte de pessoa verificável.")
        result.append({"source_segment_id": turn["source_segment_id"], "speaker": turn["speaker"],
                       "source_start_frame": first, "source_end_frame": last,
                       "output_start_frame": cursor, "output_end_frame": boundary,
                       "method": "repeat" if boundary - cursor > last - first else "trim",
                       "source_face_track_id": confirmation.pop("track_id"), **confirmation})
        cursor = boundary
    value.update(source_video={"fps_num": fps.numerator, "fps_den": fps.denominator,
                               "frame_count": frame_count, "width": width, "height": height},
                 video_base={"fps_num": fps.numerator, "fps_den": fps.denominator,
                             "frame_count": output_count, "width": width, "height": height},
                 visual_mapping=result)
    validate_timeline(value)
    # Internal geometry is deliberately separate from the portable timeline.
    return value, group_bindings


def source_frame_for_output(mapping: dict, frame: int) -> int:
    """A compact forward-loop map, with no reverse/ping-pong or frozen tail."""
    _integer(frame, "Quadro final")
    if not mapping["output_start_frame"] <= frame < mapping["output_end_frame"]:
        raise AudioMasterError("Quadro final fora do turno mapeado.")
    length = mapping["source_end_frame"] - mapping["source_start_frame"]
    return mapping["source_start_frame"] + (frame - mapping["output_start_frame"]) % length


def mapped_analysis(timeline: dict, groups: list[dict], source_analysis: dict) -> dict:
    """Remap source-box geometry onto actual retimed source-FPS frame samples."""
    validate_timeline(timeline)
    fps = Fraction(timeline["video_base"]["fps_num"], timeline["video_base"]["fps_den"])
    confirmed = []
    for mapping, group in zip(timeline["visual_mapping"], groups):
        if not group:
            continue
        boxes = []
        for frame in range(mapping["output_start_frame"], mapping["output_end_frame"]):
            source_frame = source_frame_for_output(mapping, frame)
            canonical = math.floor(Fraction(2 * source_frame + 1, 2) / fps * Fraction(str(group["fps"])))
            boxes.append(group["boxes"][canonical - group["start_frame"]])
        confirmed.append({"start_frame": mapping["output_start_frame"], "end_frame": mapping["output_end_frame"],
                          "fps": float(fps), "boxes": boxes, "speaker": mapping["speaker"],
                          "track_id": mapping["source_face_track_id"],
                          "face_confirmation": mapping["face_confirmation"],
                          "speaker_confirmation": mapping["speaker_confirmation"],
                          "coordinate_source": "rosto original confirmado no relógio de áudio natural",
                          "source_segment_id": mapping["source_segment_id"]})
    value = {"status": "complete", "timing_mode": "audio_master", "timing_policy": TIMING_POLICY,
            "safe_for_single_face_lipsync": bool(confirmed), "confirmed_face_segments": confirmed,
            "ambiguity": [], "ambiguous_intervals": [], "normalization_fps": float(fps),
            "source_video_sha256": timeline.get("source_video_sha256"),
            "source_analysis_sha256": _json_hash(source_analysis),
            "output_base_sha256": timeline["video_base"].get("sha256"),
            "audio_master_sha256": timeline.get("audio_master", {}).get("sha256"),
            "source_segments_sha256": timeline["source_segments_sha256"],
            "visual_mapping": timeline["visual_mapping"], "audio_master": timeline.get("audio_master"),
            "timeline_sha256": _json_hash(timeline), "timeline": timeline}
    if not confirmed and source_analysis.get("narration_no_face_confirmed") is True:
        value.update(no_visible_speaking_face=True, scene_analysis_complete=True,
                     narration_no_face_confirmed=True, preserved_no_speaker_visual_segments=[{
                         "start": 0.0, "end": timeline["sample_count"] / timeline["sample_rate"],
                         "reason": "Análise completa: narração sem rosto; montagem segue o áudio natural"}])
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def retime_video(config: dict, legacy) -> dict:
    """Create a checked pre-lip-sync montage; never mutate source/checkpoints."""
    video = Path(config["video"])
    timeline_path = Path(config["timeline"])
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    validate_timeline(timeline)
    master = timeline.get("audio_master", {})
    master_path = Path(config.get("audio") or timeline_path.parent / str(master.get("path") or ""))
    if (not master_path.is_file() or master.get("sha256") != _hash(master_path)):
        raise AudioMasterError("O áudio natural completo não corresponde à montagem aprovada.")
    try:
        with wave.open(str(master_path), "rb") as handle:
            if handle.getframerate() != timeline["sample_rate"] or handle.getnframes() != timeline["sample_count"]:
                raise AudioMasterError("As amostras físicas do áudio natural divergem da sua linha final.")
    except (wave.Error, EOFError) as error:
        raise AudioMasterError("O áudio mestre precisa ser PCM WAV íntegro.") from error
    source_hash = _hash(video)
    if timeline.get("source_video_sha256") != source_hash:
        raise AudioMasterError("A fonte de vídeo difere daquela aprovada no projeto.")
    source_analysis = json.loads(Path(config["analysis"]).read_text(encoding="utf-8")) if config.get("analysis") else {}
    if (source_analysis.get("no_visible_speaking_face") is True
            and source_analysis.get("scene_analysis_complete") is True):
        source_analysis["narration_no_face_confirmed"] = True
    metadata = legacy.ffprobe(video)
    stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), {})
    metadata = copy.deepcopy(metadata)
    next(item for item in metadata["streams"] if item.get("codec_type") == "video")["nb_frames"] = legacy.video_frame_count(video)
    value, groups = enrich_video_mapping(timeline, metadata, source_analysis, config.get("require_faces", True))
    output = Path(config.get("output") or config.get("video_output") or timeline_path.parent / "base-video.mp4")
    if output.resolve() == video.resolve():
        raise AudioMasterError("A montagem não pode substituir o vídeo original.")
    output.parent.mkdir(parents=True, exist_ok=True)
    log = Path(config.get("log") or output.parent / "stage.log")
    fps = Fraction(value["video_base"]["fps_num"], value["video_base"]["fps_den"])
    chunks = []
    for index, mapping in enumerate(value["visual_mapping"]):
        first, last = mapping["source_start_frame"], mapping["source_end_frame"]
        count = mapping["output_end_frame"] - mapping["output_start_frame"]
        # Decode a donor only once. FFmpeg's video loop repeats *forward*; the
        # subsequent trim has an exact frame count and no speed transform.
        filters = f"trim=start_frame={first}:end_frame={last},setpts=PTS-STARTPTS"
        if count > last - first:
            filters += f",loop=loop=-1:size={last-first}:start=0"
        filters += f",trim=end_frame={count},setpts=N/({fps.numerator}/{fps.denominator}*TB),setsar=1"
        chunk = output.parent / f"audio-master-base-turn-{index:04d}.mkv"
        legacy.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", filters, "-map", "0:v:0", "-an", "-r", str(fps), "-frames:v", str(count),
                    "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(chunk)], log=log)
        if legacy.video_frame_count(chunk) != count:
            raise AudioMasterError("Um corte/repetição perdeu quadros; montagem bloqueada.")
        chunks.append(chunk)
    # Binary-safe concat protocol of independently encoded FFV1 chunks is not
    # reliable. Use explicit inputs and concat filter instead of path manifests.
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for chunk in chunks:
        command.extend(["-i", str(chunk)])
    links = "".join(f"[{index}:v:0]" for index in range(len(chunks)))
    command.extend(["-filter_complex", f"{links}concat=n={len(chunks)}:v=1:a=0[video]", "-map", "[video]",
                    "-an", "-r", str(fps), "-frames:v", str(value["video_base"]["frame_count"]),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(output)])
    legacy.run(command, log=log)
    if legacy.video_frame_count(output) != value["video_base"]["frame_count"]:
        raise AudioMasterError("O vídeo montado não cobre todos os turnos naturais.")
    expected = value["sample_count"] / value["sample_rate"]
    actual = legacy.video_stream_duration(output)
    if actual < expected - 1e-6 or actual - expected > float(1 / fps) + 1e-6:
        raise AudioMasterError("O vídeo montado não termina com o áudio natural.")
    if _hash(video) != source_hash:
        raise AudioMasterError("O vídeo original mudou durante a montagem.")
    value["video_base"].update(path=output.name, sha256=_hash(output))
    validate_timeline(value)
    timeline_output = Path(config.get("timeline_output") or timeline_path)
    _write_json(timeline_output, value)
    analysis = mapped_analysis(value, groups, source_analysis)
    analysis["timeline_sha256"] = _hash(timeline_output)
    analysis["timeline_path"] = str(timeline_output.resolve())
    if analysis.get("confirmed_face_segments") and config.get("prepare_lipsync", True):
        analysis = prepare_mapped_canonical_analysis(output, value, analysis, legacy, log)
    _write_json(Path(config.get("mapped_analysis") or output.parent / "mapped-analysis.json"), analysis)
    if config.get("background") and config.get("background_output"):
        retime_background(Path(config["background"]), Path(config["background_output"]), value, legacy, log)
    # Temporary intermediates are ours and regenerated, not source/cache files.
    for chunk in chunks:
        chunk.unlink(missing_ok=True)
    return {"timeline": value, "output": str(output), "mapped_analysis": analysis, "output_duration": expected}


def prepare_mapped_canonical_analysis(video: Path, timeline: dict, analysis: dict, legacy, log: Path) -> dict:
    """Normalise only the inference clock; preserve source FPS in the montage.

    FFmpeg's ``round=near`` selects the last input frame mapped to each 25-fps
    timestamp. Its frame-center rule is recorded explicitly, so coordinates
    are selected from that same retimed frame rather than guessed from audio.
    """
    value = copy.deepcopy(analysis)
    fps = Fraction(timeline["video_base"]["fps_num"], timeline["video_base"]["fps_den"])
    total = timeline["video_base"]["frame_count"]
    canonical_count = math.floor(Fraction(total, 1) / fps * 25)
    if canonical_count < 1:
        raise AudioMasterError("O vídeo final não possui um quadro canônico completo para lip-sync.")
    normalized = video.parent / "audio-master-inference-fps25.mkv"
    legacy.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                "-vf", "setpts=PTS-STARTPTS,fps=25:round=near:start_time=0,setsar=1", "-an",
                "-frames:v", str(canonical_count), "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(normalized)], log=log)
    if legacy.video_frame_count(normalized) != canonical_count:
        raise AudioMasterError("A entrada canônica perdeu quadros da montagem.")
    source_groups = value["confirmed_face_segments"]
    canonical_groups, preserved, source_indices = [], [], []
    for frame in range(canonical_count):
        # The tiny exact subtraction implements FFmpeg's half-away-from-zero
        # boundary without floating-point ties selecting the following face.
        selected = canonical_source_frame(frame, fps, total)
        source_indices.append(selected)
        candidates = [group for group in source_groups if group["start_frame"] <= selected < group["end_frame"]]
        if len(candidates) > 1:
            raise AudioMasterError("Mais de um rosto foi associado ao quadro montado.")
        if not candidates:
            if not preserved or preserved[-1]["end_frame"] != frame:
                preserved.append({"start_frame": frame, "end_frame": frame + 1, "fps": 25.0,
                                  "reason": "narração sem rosto confirmada"})
            else:
                preserved[-1]["end_frame"] += 1
            continue
        group = candidates[0]
        box = group["boxes"][selected - group["start_frame"]]
        if (not canonical_groups or canonical_groups[-1]["end_frame"] != frame
                or canonical_groups[-1]["source_segment_id"] != group["source_segment_id"]):
            canonical_groups.append({**{key: group[key] for key in ("speaker", "track_id", "face_confirmation",
                                       "speaker_confirmation", "coordinate_source", "source_segment_id")},
                                     "start_frame": frame, "end_frame": frame + 1, "fps": 25.0, "boxes": [box],
                                     "audio_master_mapped": True})
        else:
            canonical_groups[-1]["end_frame"] += 1
            canonical_groups[-1]["boxes"].append(box)
    value.update(confirmed_face_segments=canonical_groups, normalized_video=str(normalized.resolve()),
                 normalized_video_sha256=_hash(normalized), normalization_fps=25.0,
                 mapped_source_segments=source_groups, preserved_no_speaker_visual_segments=[
                     {"start": item["start_frame"] / 25, "end": item["end_frame"] / 25,
                      "reason": item["reason"]} for item in preserved],
                 mapped_normalization={"policy": "retimed-frame-center-fps25-v1", "frame_count": canonical_count,
                                       "source_frame_indices_sha256": _json_hash(source_indices),
                                       "source_fps_num": fps.numerator, "source_fps_den": fps.denominator,
                                       "source_frame_count": total})
    if {item["source_segment_id"] for item in source_groups} != {item["source_segment_id"] for item in canonical_groups}:
        raise AudioMasterError("Um turno facial não cabe no relógio de 25 fps; lip-sync não será declarado completo.")
    return value


def canonical_source_frame(frame: int, source_fps: Fraction, source_count: int) -> int:
    """Exact source sample selected by FFmpeg fps=25:round=near."""
    return min(source_count - 1, max(0, math.ceil(Fraction(2 * frame + 1, 2) * source_fps / 25) - 1))


def retime_background(source: Path, output: Path, timeline: dict, legacy, log: Path) -> None:
    """Extend/cut the separated bed independently; never loop original vocals."""
    validate_timeline(timeline)
    fps = Fraction(timeline["source_video"]["fps_num"], timeline["source_video"]["fps_den"])
    rate = timeline["sample_rate"]
    chunks = []
    for index, (turn, mapping) in enumerate(zip(timeline["turns"], timeline["visual_mapping"])):
        first = math.floor(Fraction(mapping["source_start_frame"], 1) / fps * rate)
        last = math.ceil(Fraction(mapping["source_end_frame"], 1) / fps * rate)
        count = turn["pcm_samples"]
        chunk = output.parent / f"audio-master-background-turn-{index:04d}.wav"
        filters = f"aresample={rate},atrim=start_sample={first}:end_sample={last},asetpts=PTS-STARTPTS"
        if count > last - first:
            fade = min(rate // 100, (last - first) // 4)
            if fade > 0:
                filters += (f",afade=t=in:start_sample=0:nb_samples={fade}"
                            f",afade=t=out:start_sample={last-first-fade}:nb_samples={fade}")
            filters += f",aloop=loop=-1:size={last-first}:start=0"
        filters += f",atrim=end_sample={count},asetpts=PTS-STARTPTS"
        fade = min(rate // 100, count // 4)
        if fade > 0:
            filters += (f",afade=t=in:start_sample=0:nb_samples={fade}"
                        f",afade=t=out:start_sample={count-fade}:nb_samples={fade}")
        legacy.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-af", filters, "-ac", "2", "-ar", str(rate), "-c:a", "pcm_s16le", str(chunk)], log=log)
        with wave.open(str(chunk), "rb") as handle:
            if handle.getnframes() != count:
                raise AudioMasterError("A trilha separada não cobre o turno natural.")
        chunks.append(chunk)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for chunk in chunks:
        command.extend(["-i", str(chunk)])
    links = "".join(f"[{index}:a:0]" for index in range(len(chunks)))
    command.extend(["-filter_complex", f"{links}concat=n={len(chunks)}:v=0:a=1[audio]", "-map", "[audio]",
                    "-ac", "2", "-ar", str(rate), "-c:a", "pcm_s16le", str(output)])
    legacy.run(command, log=log)
    with wave.open(str(output), "rb") as handle:
        if handle.getnframes() != timeline["sample_count"]:
            raise AudioMasterError("O fundo editado termina fora do áudio mestre.")
    for chunk in chunks:
        chunk.unlink(missing_ok=True)
