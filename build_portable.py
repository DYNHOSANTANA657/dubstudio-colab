"""Build and independently verify a public, source-only DubStudio distribution.

No directory is recursively added. Every packaged source file must appear in the
explicit whitelist below. Existing destination files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
from datetime import datetime, timezone
import zipfile

VERSION = "3.0.0"
APP_DIRECTORY = f"DubStudio-Colab-v{VERSION}"
SOURCE_ROOT = Path(__file__).resolve().parent
PUBLIC_FILES = (
    ".gitignore", "README.md", "README_COLAB.md", "config.example.json",
    "server.py", "studio_server.py", "studio_bridge.py", "pipeline.py", "languages.py",
    "colab_bridge.py", "DubStudio_Colab.ipynb", "start_windows.bat", "start_powershell.ps1",
    "build_portable.py", "RELATORIO_TESTES.md",
    "colab/dubstudio_bootstrap.py", "colab/dubstudio_runner_v2.py",
    "colab/dubstudio_colab_worker.py", "colab/dubstudio_translate.py",
    "colab/requirements-app.txt", "colab/requirements-musetalk.txt",
    "colab/test_runner_v2.py", "colab/test_cache_safety.py", "colab/test_translation_loading.py", "colab/test_review_resume.py", "colab/test_voice_outputs.py", "colab/test_lipsync_contract.py", "colab/test_bootstrap_setup.py", "colab/test_muse_render_errors.py",
    "static/index.html", "static/app.js", "static/styles.css", "static/favicon.svg",
    "tests/test_studio_bridge.py", "tests/test_studio_server.py",
)
GENERATED_FILES = ("SHA256SUMS.txt", "PACKAGING_REPORT.json")
FORBIDDEN_PARTS = {"runtime", "cloud", "__pycache__", "node_modules", ".git"}
FORBIDDEN_NAMES = {"config.json", "runpod-session.json", "server_runpod_legacy.py",
                   "cloud_pipeline.py", "runpod_client.py", "runpod_session.py", "runpod_secrets.py",
                   "README_LEGADO.md", "publish_runpod_image.ps1"}
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".ipynb", ".bat", ".ps1", ".txt",
                    ".html", ".js", ".css", ".svg"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_public_path(name: str) -> None:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe public path: {name!r}")
    if any(part in FORBIDDEN_PARTS or part.startswith(".test") for part in path.parts):
        raise ValueError(f"Private or temporary directory in package: {name!r}")
    if path.name in FORBIDDEN_NAMES or (path.suffix not in ALLOWED_SUFFIXES and name != ".gitignore"):
        raise ValueError(f"Non-public file type in package: {name!r}")


def source_snapshot(source_root: Path) -> dict[str, bytes]:
    root = source_root.resolve(strict=True)
    files = {}
    if len(PUBLIC_FILES) != len(set(PUBLIC_FILES)):
        raise ValueError("Duplicate names in source whitelist")
    for name in PUBLIC_FILES:
        validate_public_path(name)
        path = root / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/unsafe source file: {name}")
        files[name] = path.read_bytes()
    # A used notebook must remain an unexecuted public template, never a session export.
    notebook = json.loads(files["DubStudio_Colab.ipynb"])
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code" and (cell.get("outputs") or cell.get("execution_count") is not None):
            raise ValueError("Notebook contains private execution outputs; clear it before packaging")
    settings = json.loads(files["config.example.json"])
    if "runpod" in settings or "commands" in settings or "api_key" in settings:
        raise ValueError("Config example still contains legacy/private settings")
    return files


def validate_archive(path: Path, source_files: dict[str, bytes]) -> dict:
    """Check exact names, privacy boundaries, source bytes and hash index after ZIP closes."""
    expected = {f"{APP_DIRECTORY}/{name}" for name in (*PUBLIC_FILES, *GENERATED_FILES)}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        actual = [item.filename for item in infos]
        if len(actual) != len(expected) or len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("ZIP does not contain exactly the explicit public whitelist")
        for info in infos:
            if info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("ZIP contains encrypted files or symbolic links")
            relative = PurePosixPath(info.filename).relative_to(APP_DIRECTORY).as_posix()
            validate_public_path(relative)
            if relative in source_files and archive.read(info) != source_files[relative]:
                raise ValueError(f"ZIP source bytes changed: {relative}")
        checksums = archive.read(f"{APP_DIRECTORY}/SHA256SUMS.txt").decode("utf-8")
        expected_checksums = "".join(f"{digest(source_files[name])}  {name}\n" for name in PUBLIC_FILES)
        if checksums != expected_checksums:
            raise ValueError("ZIP checksum index differs from source files")
        report = json.loads(archive.read(f"{APP_DIRECTORY}/PACKAGING_REPORT.json"))
        expected_records = [{"path": name, "sha256": digest(source_files[name]), "bytes": len(source_files[name])}
                            for name in PUBLIC_FILES]
        if report.get("files") != expected_records or report.get("version") != VERSION:
            raise ValueError("Packaging report does not describe this source snapshot")
        return {"public_source_files": len(source_files), "archive_members": len(actual),
                "private_files_included": 0, "source_bytes_verified": True,
                "checksums_verified": True, "privacy_whitelist_verified": True}


def reserve_destination(directory: Path, stem: str, suffix: str) -> Path:
    for number in range(1, 10000):
        candidate = directory / f"{stem}{'' if number == 1 else f'-{number}'}{suffix}"
        try:
            with candidate.open("xb"):
                pass
            return candidate
        except FileExistsError:
            continue
    raise ValueError("No unused destination filename is available")


def build(output_directory: Path, source_root: Path = SOURCE_ROOT) -> dict:
    source_files = source_snapshot(source_root)
    output_directory = output_directory.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        archive_path = reserve_destination(output_directory, APP_DIRECTORY, ".zip")
        created.append(archive_path)
        report = {"version": VERSION, "built_at_utc": datetime.now(timezone.utc).isoformat(),
                  "policy": "Explicit public-source whitelist; no recursive directory packaging",
                  "runtime_data_included": False, "credentials_included": False,
                  "media_included": False, "gpu_validation": "pending",
                  "files": [{"path": name, "sha256": digest(data), "bytes": len(data)}
                            for name, data in source_files.items()]}
        checksums = "".join(f"{digest(data)}  {name}\n" for name, data in source_files.items())
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in source_files.items():
                archive.writestr(f"{APP_DIRECTORY}/{name}", data)
            archive.writestr(f"{APP_DIRECTORY}/SHA256SUMS.txt", checksums.encode("utf-8"))
            archive.writestr(f"{APP_DIRECTORY}/PACKAGING_REPORT.json", json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8"))
        verification = validate_archive(archive_path, source_files)
        if source_snapshot(source_root) != source_files:
            raise ValueError("Source files changed during build; rebuild after edits finish")
        notebook_path = reserve_destination(output_directory, f"DubStudio_Colab-v{VERSION}", ".ipynb")
        created.append(notebook_path)
        notebook_path.write_bytes(source_files["DubStudio_Colab.ipynb"])
        if notebook_path.read_bytes() != source_files["DubStudio_Colab.ipynb"]:
            raise ValueError("Standalone notebook copy failed source verification")
        checksum_path = reserve_destination(output_directory, f"{archive_path.stem}-SHA256SUMS", ".txt")
        created.append(checksum_path)
        checksum_path.write_text(f"{digest(archive_path.read_bytes())}  {archive_path.name}\n"
                                 f"{digest(notebook_path.read_bytes())}  {notebook_path.name}\n", encoding="utf-8")
        return {"zip_path": str(archive_path), "notebook_path": str(notebook_path),
                "checksums_path": str(checksum_path), "zip_sha256": digest(archive_path.read_bytes()),
                "verification": verification}
    except BaseException:
        # Only files this invocation exclusively created are removed; never old outputs.
        for path in created:
            if path.parent.resolve() == output_directory and path.exists():
                path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a clean, checksummed DubStudio source-only ZIP")
    parser.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads",
                        help="Destination directory; existing names get a numeric suffix")
    arguments = parser.parse_args()
    try:
        print(json.dumps(build(arguments.output_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Packaging failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
