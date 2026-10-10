#!/usr/bin/env python3
"""Prepara o runtime local do Colab e reutiliza pesos/checkpoints do Drive.

Não mantém a sessão conectada, não abre túnel e não controla login. O marcador
de ambiente só é escrito após os imports exigidos passarem. Uma falha preserva
os downloads concluídos, mas nunca é interpretada como instalação pronta.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

PIPELINE_VERSION = "3.0.0"
MUSE_COMMIT = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
TALKNET_COMMIT = "6d6821479af485e251c4991487e40573b42181b4"
PERTH_COMMIT = "ff1c8ac55a976971245cdd53c18d6131ca00d993"
CACHE_LINKS_FILE = ".dubstudio-cache-links.json"


def execute(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def gpu_precheck() -> str:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("GPU indisponível. Selecione GPU em Ambiente de execução → Alterar tipo de ambiente de execução e reconecte.")
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    if not result.stdout.strip():
        raise RuntimeError("O Colab não concedeu uma GPU nesta sessão. Nenhum modelo foi baixado.")
    return result.stdout.strip()


def file_digest(file: Path) -> bytes:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _mirror_roots(source: Path, destination: Path) -> bool:
    """No self-copy or recursively nested mirrors, including path aliases."""
    source_root, destination_root = source.resolve(), destination.resolve()
    if source_root == destination_root:
        return False
    if source_root.is_relative_to(destination_root) or destination_root.is_relative_to(source_root):
        raise RuntimeError("Origem e destino do cache não podem ficar um dentro do outro.")
    return True


def _cache_path(root: Path, relative: str) -> Path:
    """Validate lexical paths and existing parent directories without following the leaf."""
    if not isinstance(relative, str) or not relative or any(char in relative for char in ("\\", ":", "\x00")):
        raise RuntimeError("O cache contém metadados de link com caminho inválido.")
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts or str(path) == ".":
        raise RuntimeError("O cache contém metadados de link fora do diretório autorizado.")
    target = root.joinpath(*path.parts)
    if not target.parent.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("O cache contém diretório apontando para fora do cache autorizado.")
    return target


def _read_cache_links(root: Path) -> dict[str, str]:
    metadata = root / CACHE_LINKS_FILE
    if not metadata.exists():
        return {}
    if metadata.is_symlink() or metadata.stat().st_size > 1024 * 1024:
        raise RuntimeError("Metadados de links do cache inválidos.")
    try:
        payload = json.loads(metadata.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError("Não foi possível validar os links persistidos do cache.") from exc
    links = payload.get("links") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(links, dict) or len(links) > 10000:
        raise RuntimeError("Formato de links persistidos do cache inválido.")
    for relative, reference in links.items():
        link = _cache_path(root, relative)
        canonical = _cache_path(root, reference)
        if relative == CACHE_LINKS_FILE or reference == CACHE_LINKS_FILE or relative == reference:
            raise RuntimeError("O cache contém referência circular ou reservada.")
        if not canonical.is_file() or not canonical.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("O link persistido não referencia um arquivo do cache autorizado.")
        if link.exists() and not link.is_symlink():
            raise RuntimeError("Os metadados de cache conflitam com um arquivo real.")
    return links


def _copy_regular_file(file: Path, target: Path, *, verify: bool) -> None:
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise RuntimeError("O destino do cache não pode ser um link ou diretório inesperado.")
    if target.exists() and target.stat().st_size == file.stat().st_size:
        if not verify or file_digest(target) == file_digest(file):
            return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".copying")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise RuntimeError("Arquivo temporário do cache inseguro.")
    shutil.copy2(file, temporary)
    temporary.replace(target)


def _write_cache_links(root: Path, links: dict[str, str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / CACHE_LINKS_FILE
    temporary = target.with_name(target.name + ".writing")
    if target.is_symlink() or temporary.is_symlink():
        raise RuntimeError("Arquivo de metadados do cache inseguro.")
    temporary.write_text(json.dumps({"schema_version": 1, "links": links}, sort_keys=True), encoding="utf-8")
    temporary.replace(target)


def _restore_cache_link(root: Path, relative: str, reference: str) -> None:
    target, canonical = _cache_path(root, relative), _cache_path(root, reference)
    if not canonical.is_file() or not canonical.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("O arquivo real do cache precisa existir antes de restaurar o link.")
    if target.is_symlink() and target.resolve() == canonical.resolve():
        return
    duplicate = target.exists() and not target.is_symlink()
    if duplicate and (not target.is_file() or file_digest(target) != file_digest(canonical)):
        raise RuntimeError("O snapshot existente diverge do arquivo real do cache.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".link-writing")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("Já existe um arquivo temporário de link no cache.")
    try:
        temporary.symlink_to(os.path.relpath(canonical, target.parent))
    except OSError:
        # Google Drive/FUSE and non-privileged Windows may not support symlinks.
        # The root metadata is authoritative: never materialize another copy of
        # a multi-GB snapshot. Local restoration recreates its relative link.
        if target.is_symlink():
            target.unlink()
        elif duplicate:
            target.unlink()  # Remove only an independently verified cache duplicate.
        return
    temporary.replace(target)


def copy_missing(source: Path, destination: Path, *, verify: bool = False, preserve_cache_links: bool = False) -> None:
    """Incremental mirror; cache snapshots store references, never duplicate blobs."""
    if not _mirror_roots(source, destination):
        return
    if not source.exists():
        return
    links = _read_cache_links(source) if preserve_cache_links else {}
    saved_links = _read_cache_links(destination) if preserve_cache_links else {}
    files = []
    for file in source.rglob("*"):
        if file.name.endswith((".lock", ".incomplete", ".copying", ".writing", ".link-writing")):
            continue
        relative = file.relative_to(source).as_posix()
        if preserve_cache_links and relative == CACHE_LINKS_FILE:
            continue
        if file.is_symlink():
            canonical = file.resolve()
            if not preserve_cache_links or not canonical.is_file() or not canonical.is_relative_to(source.resolve()):
                raise RuntimeError("O cache contém link fora do diretório de arquivos autorizado.")
            links[relative] = canonical.relative_to(source.resolve()).as_posix()
        elif file.is_file():
            files.append((file, relative))
    # Validate all source links before copying anything; parent symlinks cannot
    # redirect writes outside the destination, even in a previously saved cache.
    for relative, reference in links.items():
        _cache_path(destination, relative)
        _cache_path(destination, reference)
    for file, relative in files:
        _copy_regular_file(file, _cache_path(destination, relative), verify=verify)
    if preserve_cache_links and (links or saved_links):
        saved_links.update(links)
        for relative, reference in saved_links.items():
            _restore_cache_link(destination, relative, reference)
        _write_cache_links(destination, saved_links)


def persist_cache(runtime: Path, storage: Path) -> None:
    if runtime.resolve() == storage.resolve():
        return
    for name in ("cache", "models"):
        copy_missing(runtime / name, storage / name, preserve_cache_links=name == "cache")


def clone_at(repository: str, destination: Path, revision: str) -> None:
    if not (destination / ".git").exists():
        execute(["git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination)])
    current = subprocess.run(["git", "rev-parse", "HEAD"], cwd=destination, capture_output=True, text=True)
    if current.returncode or current.stdout.strip() != revision:
        execute(["git", "fetch", "--depth", "1", "origin", revision], cwd=destination)
        execute(["git", "checkout", "--detach", revision], cwd=destination)


def environment(runtime: Path) -> dict[str, str]:
    values = os.environ.copy()
    values.update({
        "HF_HOME": str(runtime / "cache/huggingface"),
        "TORCH_HOME": str(runtime / "cache/torch"),
        "DUBSTUDIO_WHISPER_CACHE": str(runtime / "cache/whisper"),
        "DUBSTUDIO_SPEECHBRAIN_CACHE": str(runtime / "cache/speechbrain"),
        "DUBSTUDIO_MUSETALK_ROOT": str(runtime / "MuseTalk"),
        "DUBSTUDIO_TALKNET_ROOT": str(runtime / "TalkNet-ASD"),
        "DUBSTUDIO_MUSETALK_PYTHON": str(runtime / "musetalk-env/bin/python"),
        "DUBSTUDIO_TRANSLATE_PYTHON": str(runtime / "app-env/bin/python"),
        "DUBSTUDIO_MUSETALK_BATCH": "4",
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    torch_lib = runtime / "app-env/lib/python3.10/site-packages/torch/lib"
    cuda_libs = runtime / "app-env/lib/python3.10/site-packages/nvidia"
    libraries = [str(torch_lib)] + [str(path / "lib") for path in cuda_libs.glob("*") if (path / "lib").exists()]
    values["LD_LIBRARY_PATH"] = ":".join(libraries + [values.get("LD_LIBRARY_PATH", "")])
    values["PATH"] = str(runtime / "app-env/bin") + ":" + values.get("PATH", "")
    return values


def prepare(runtime: Path, storage: Path, needs_lipsync: bool = True) -> dict:
    gpu = gpu_precheck()  # Must run before apt/pip or any model download.
    runtime.mkdir(parents=True, exist_ok=True)
    storage.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(runtime).free < 15 * 1024**3:
        raise RuntimeError("São necessários pelo menos 15 GB livres no runtime para preparar os modelos.")
    print(f"GPU concedida: {gpu}\nRestaurando cache local…", flush=True)
    for directory in ("cache", "models"):
        copy_missing(storage / directory, runtime / directory, preserve_cache_links=directory == "cache")
    if not shutil.which("ffmpeg") or not shutil.which("git"):
        execute(["apt-get", "update", "-qq"])
        execute(["apt-get", "install", "-y", "-qq", "ffmpeg", "git", "libgl1", "libglib2.0-0"])
    if not shutil.which("uv"):
        execute([sys.executable, "-m", "pip", "install", "--quiet", "uv==0.8.22"])
    execute(["uv", "python", "install", "3.10"])
    requirements = Path(__file__).parent / "requirements-app.txt"
    fingerprint = hashlib.sha256(requirements.read_bytes() + PIPELINE_VERSION.encode()).hexdigest()
    app = runtime / "app-env"
    marker = app / ".ready.json"
    ready = marker.exists() and json.loads(marker.read_text()).get("fingerprint") == fingerprint
    if not ready:
        print("Preparando ambiente de transcrição, tradução e timbre…", flush=True)
        if not (app / "bin/python").exists():
            execute(["uv", "venv", "--seed", "--python", "3.10", str(app)])
        pip = [str(app / "bin/python"), "-m", "pip"]
        execute(pip + ["install", "--quiet", "pip==25.2", "wheel==0.45.1", "setuptools==80.9.0"])
        execute(pip + ["install", "--quiet", f"resemble-perth @ git+https://github.com/resemble-ai/Perth.git@{PERTH_COMMIT}"])
        execute(pip + ["install", "--quiet", "-r", str(requirements)])
        execute([str(app / "bin/python"), "-c", "import torch,torchaudio; from chatterbox.mtl_tts import ChatterboxMultilingualTTS; from faster_whisper import WhisperModel; from transformers import AutoModelForSeq2SeqLM; assert torch.cuda.is_available(); print('Imports/GPU de dublagem OK')"])
        marker.write_text(json.dumps({"fingerprint": fingerprint, "pipeline_version": PIPELINE_VERSION}))
    else:
        print("Ambiente de dublagem já preparado nesta sessão.", flush=True)
    values = environment(runtime)
    if needs_lipsync:
        muse = runtime / "musetalk-env"
        requirements = Path(__file__).parent / "requirements-musetalk.txt"
        fingerprint = hashlib.sha256(requirements.read_bytes() + MUSE_COMMIT.encode()).hexdigest()
        marker = muse / ".ready.json"
        ready = marker.exists() and json.loads(marker.read_text()).get("fingerprint") == fingerprint
        clone_at("https://github.com/TMElyralab/MuseTalk.git", runtime / "MuseTalk", MUSE_COMMIT)
        clone_at("https://github.com/TaoRuijie/TalkNet-ASD.git", runtime / "TalkNet-ASD", TALKNET_COMMIT)
        if not ready:
            if not (muse / "bin/python").exists():
                execute(["uv", "venv", "--seed", "--python", "3.10", str(muse)])
            pip = [str(muse / "bin/python"), "-m", "pip"]
            execute(pip + ["install", "--quiet", "pip==25.2", "wheel==0.45.1", "setuptools==80.9.0"])
            execute(pip + ["install", "--quiet", "torch==2.0.1", "torchvision==0.15.2", "torchaudio==2.0.2", "--index-url", "https://download.pytorch.org/whl/cu118"])
            execute(pip + ["install", "--quiet", "-r", str(requirements)])
            execute([str(muse / "bin/mim"), "install", "mmengine==0.10.7"])
            execute([str(muse / "bin/mim"), "install", "mmcv==2.0.1"])
            execute([str(muse / "bin/mim"), "install", "mmdet==3.1.0"])
            execute(pip + ["install", "--quiet", "--no-deps", "mmpose==1.1.0"])
        download_model_weights(runtime, values)
        # Use local models for computation. Drive contains only persisted copies.
        model_link = runtime / "MuseTalk/models"
        if not model_link.is_symlink():
            if model_link.exists():
                copy_missing(model_link, runtime / "models/musetalk")
                # Git clone models only has examples, preserve it alongside the link.
                model_link.rename(runtime / "MuseTalk/models-repository")
            model_link.symlink_to(runtime / "models/musetalk", target_is_directory=True)
        if not ready:
            muse_values = values.copy()
            muse_values.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            muse_values["LD_LIBRARY_PATH"] = str(muse / "lib/python3.10/site-packages/torch/lib")
            execute([str(muse / "bin/python"), "-c", "import torch,cv2,numpy; from musetalk.utils.audio_processor import AudioProcessor; from musetalk.utils.preprocessing import get_landmark_and_bbox; from musetalk.utils.face_parsing import FaceParsing; assert torch.cuda.is_available(); print('Imports/GPU de MuseTalk OK')"], cwd=runtime / "MuseTalk", env=muse_values)
            marker.write_text(json.dumps({"fingerprint": fingerprint, "pipeline_version": PIPELINE_VERSION}))
        else:
            print("Ambiente de lip-sync já preparado nesta sessão.", flush=True)
    persist_cache(runtime, storage)
    print("Ambiente pronto. Versões candidatas registradas; qualidade será validada no resultado real.", flush=True)
    return {"gpu": gpu, "runtime": str(runtime), "storage": str(storage), "pipeline_version": PIPELINE_VERSION}


def download_model_weights(runtime: Path, values: dict) -> None:
    code = """
import os
from pathlib import Path
from huggingface_hub import hf_hub_download
root=Path(os.environ['DUBSTUDIO_MUSETALK_ROOT']).parent/'models/musetalk'
required=[('TMElyralab/MuseTalk','musetalkV15/musetalk.json',root),('TMElyralab/MuseTalk','musetalkV15/unet.pth',root),('stabilityai/sd-vae-ft-mse','config.json',root/'sd-vae'),('stabilityai/sd-vae-ft-mse','diffusion_pytorch_model.bin',root/'sd-vae'),('openai/whisper-tiny','config.json',root/'whisper'),('openai/whisper-tiny','pytorch_model.bin',root/'whisper'),('openai/whisper-tiny','preprocessor_config.json',root/'whisper'),('yzd-v/DWPose','dw-ll_ucoco_384.pth',root/'dwpose')]
for repo,name,directory in required:
    destination=directory/name
    directory.mkdir(parents=True,exist_ok=True)
    if not destination.exists() or destination.stat().st_size==0:
        hf_hub_download(repo_id=repo,filename=name,local_dir=str(directory))
"""
    execute([str(runtime / "app-env/bin/python"), "-c", code], env=values)
    weights = [
        ("https://drive.google.com/uc?id=1KafnHz7ccT-3IyddBsL5yi2xGtxAKypt", runtime / "models/talknet/sfd_face.pth"),
        ("https://drive.google.com/uc?id=1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea", runtime / "models/talknet/pretrain_TalkSet.model"),
        ("https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812", runtime / "models/musetalk/face-parse-bisent/79999_iter.pth"),
    ]
    for url, file in weights:
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists() or not file.stat().st_size:
            execute([str(runtime / "app-env/bin/gdown"), url, "-O", str(file)], env=values)
        if not file.exists() or not file.stat().st_size:
            raise RuntimeError(f"Peso obrigatório não foi baixado: {file.name}")
    resnet = runtime / "models/musetalk/face-parse-bisent/resnet18-5c106cde.pth"
    if not resnet.exists() or not resnet.stat().st_size:
        execute(["curl", "--fail", "--location", "--retry", "3", "https://download.pytorch.org/models/resnet18-5c106cde.pth", "-o", str(resnet)])
    for source, target in (
        (runtime / "models/talknet/sfd_face.pth", runtime / "TalkNet-ASD/model/faceDetector/s3fd/sfd_face.pth"),
        (runtime / "models/talknet/pretrain_TalkSet.model", runtime / "TalkNet-ASD/pretrain_TalkSet.model"),
    ):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, default=Path("/content/dubstudio"))
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--no-lipsync", action="store_true")
    args = parser.parse_args()
    prepare(args.runtime_root, args.storage_root, not args.no_lipsync)


if __name__ == "__main__":
    main()
