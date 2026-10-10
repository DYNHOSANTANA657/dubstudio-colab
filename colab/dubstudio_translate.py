#!/usr/bin/env python3
"""Tradutor MADLAD isolado das dependências do Chatterbox."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

# Mantido somente para leitura de checkpoints do fluxo legado. O fluxo atual
# executa MADLAD primeiro e registra a validação; nunca troca modelo sem avisar.
NLLB_CODES = {
    "ar": "arb_Arab", "bg": "bul_Cyrl", "cs": "ces_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn", "es": "spa_Latn",
    "fi": "fin_Latn", "fr": "fra_Latn", "he": "heb_Hebr", "hi": "hin_Deva",
    "hu": "hun_Latn", "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "nl": "nld_Latn", "no": "nob_Latn", "pl": "pol_Latn",
    "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sv": "swe_Latn",
    "tr": "tur_Latn", "uk": "ukr_Cyrl", "vi": "vie_Latn", "zh": "zho_Hans",
}
LATIN_TARGETS = {"cs", "da", "de", "en", "es", "fi", "fr", "hu", "id", "it", "ms", "nl", "no", "pl", "pt", "ro", "sv", "sw", "tr", "vi"}
SCRIPT_TARGETS = {"ar": "ARABIC", "he": "HEBREW", "ru": "CYRILLIC", "bg": "CYRILLIC", "uk": "CYRILLIC", "hi": "DEVANAGARI", "el": "GREEK", "ko": "HANGUL"}


def is_sane_translation(text: str, target: str) -> bool:
    """Recusa saídas repetitivas ou com escrita incompatível com o idioma."""
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 2:
        return False
    # Respostas curtas como "No", "Ja" ou "Tak" são válidas em polonês.
    # A regra anterior rejeitava qualquer palavra curta com uma letra repetida
    # proporcionalmente, transformando uma tradução boa em erro do pipeline.
    letters = [char for char in compact if unicodedata.category(char).startswith("L")]
    if len(letters) < 1:
        return False
    if len(letters) >= 8 and max(letters.count(char) for char in set(letters)) / len(letters) > 0.65:
        return False
    target_base = target.split("-")[0].lower()
    if target_base in LATIN_TARGETS:
        latin = sum(1 for char in letters if "LATIN" in unicodedata.name(char, ""))
        if latin / len(letters) < 0.60:
            return False
    script = SCRIPT_TARGETS.get(target_base)
    if script and sum(script in unicodedata.name(char, "") for char in letters) / len(letters) < 0.50:
        return False
    return True


def translations_are_valid(translations: list[str], segments: list[dict], target: str) -> bool:
    return len(translations) == len(segments) and all(is_sane_translation(item, target) for item in translations)


def generate_nllb(segments: list[dict], source: str, target: str) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    source_code, target_code = NLLB_CODES[source], NLLB_CODES[target]
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang=source_code)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    try:
        forced_bos = tokenizer.convert_tokens_to_ids(target_code)
        translated = []
        for segment in segments:
            encoded = tokenizer(segment["text"], return_tensors="pt", truncation=True, max_length=512).to(model.device)
            with torch.inference_mode():
                output = model.generate(**encoded, forced_bos_token_id=forced_bos, max_new_tokens=256, num_beams=4)
            translated.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip())
        return translated
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_madlad(segments: list[dict], target: str) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("A tradução MADLAD exige a GPU concedida pelo Colab.")
    tokenizer = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/madlad400-3b-mt", torch_dtype=torch.float16, device_map={"": 0}, low_cpu_mem_usage=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )
    try:
        translated = []
        for index, segment in enumerate(segments, 1):
            print(f"MADLAD {target}: segmento {index}/{len(segments)}", flush=True)
            pieces = split_text(str(segment["text"]))
            output_pieces = []
            for piece in pieces:
                encoded = tokenizer(f"<2{target}> {piece}", return_tensors="pt", truncation=False).to(model.device)
                if encoded["input_ids"].shape[1] > 512:
                    raise RuntimeError("Um segmento excede o limite do tradutor. Divida o texto antes de continuar.")
                with torch.inference_mode():
                    output = model.generate(**encoded, max_new_tokens=512, num_beams=4, repetition_penalty=1.05)
                if output.shape[1] >= 512:
                    raise RuntimeError("O tradutor atingiu o limite de saída; segmento não será aceito incompleto.")
                output_pieces.append(tokenizer.batch_decode(output, skip_special_tokens=True)[0].strip())
            translated.append(" ".join(output_pieces))
        return translated
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def split_text(text: str, limit: int = 320) -> list[str]:
    """Divide nos limites de palavras, sem perder nem truncar o texto original."""
    words = text.split()
    pieces, current = [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > limit:
            pieces.append(current)
            current = ""
        current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def detected_output_language(translations: list[str], target: str) -> dict:
    text = " ".join(translations)
    letters = sum(unicodedata.category(char).startswith("L") for char in text)
    if letters < 120:
        return {"method": "script_and_structure", "reason": "texto curto para identificação estatística confiável"}
    from langid.langid import LanguageIdentifier, model
    detector = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    language, confidence = detector.classify(text)
    confidence = float(confidence)
    if language != target and confidence >= 0.85:
        raise RuntimeError(f"Idioma de saída divergente: esperado {target}, detectado {language} ({confidence:.0%}). Revise a tradução antes de gerar a voz.")
    return {"method": "langid_script_and_structure", "detected_language": language, "confidence": confidence}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", default="auto")
    parser.add_argument("--target", required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    if not segments:
        raise RuntimeError("Nenhum segmento recebido para tradução.")
    source, target = args.source.split("-")[0].lower(), args.target.split("-")[0].lower()
    source, target = ("no" if item == "nb" else item for item in (source, target))
    engine = "MADLAD-400 3B"
    if source == target:
        translations = [str(segment["text"]) for segment in segments]
        engine = "original no idioma solicitado"
    else:
        translations = generate_madlad(segments, target)
    if not translations_are_valid(translations, segments, target):
        raise RuntimeError(f"A tradução para {args.target} falhou na validação de escrita/estrutura; a dublagem não será gerada.")
    language_validation = detected_output_language(translations, target)
    for segment, translated in zip(segments, translations):
        segment["translated_text"] = translated
    args.output.write_text(
        json.dumps({"source_language": args.source, "target_language": args.target, "translation_engine": engine, "language_validation": language_validation, "segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
