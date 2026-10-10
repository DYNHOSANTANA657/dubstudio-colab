"""Cache I/O protocol tests; these do not claim a Drive or GPU execution."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dubstudio_bootstrap as bootstrap
import dubstudio_runner_v2 as runner


class CacheSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source, self.destination = self.root / "local", self.root / "saved"
        self.source.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def blob(self, root=None, name="weights", content=b"one canonical model blob"):
        path = (root or self.source) / "huggingface/hub/models--google--model/blobs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def link_metadata(self, root=None, reference=None, relative=None):
        root = root or self.source
        reference = reference or self.blob(root).relative_to(root).as_posix()
        relative = relative or "huggingface/hub/models--google--model/snapshots/revision/model.safetensors"
        (root / bootstrap.CACHE_LINKS_FILE).write_text(json.dumps({"schema_version": 1, "links": {relative: reference}}))
        return relative, reference

    def symlink_or_skip(self, path, reference):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(os.path.relpath(reference, path.parent), target_is_directory=reference.is_dir())
        except OSError as exc:
            self.skipTest(f"Host não autoriza symlink: {exc}")

    def test_same_root_and_alias_do_not_hash_copy_or_overwrite(self):
        file = self.source / "checkpoint.json"
        file.write_bytes(b"preserve original")
        with patch.object(bootstrap, "file_digest", side_effect=AssertionError("must not hash")), patch.object(bootstrap.shutil, "copy2", side_effect=AssertionError("must not copy")):
            bootstrap.copy_missing(self.source, self.source / ".", verify=True, preserve_cache_links=True)
        self.assertEqual(file.read_bytes(), b"preserve original")

    def test_runtime_cache_same_root_does_not_mirror(self):
        with patch.object(bootstrap, "copy_missing", side_effect=AssertionError("must not mirror")):
            bootstrap.persist_cache(self.source, self.source / ".")

    def test_nested_mirrors_rejected_before_writes(self):
        for source, target in ((self.source, self.source / "nested"), (self.source / "nested", self.source)):
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "dentro do outro"):
                bootstrap.copy_missing(source, target)
        self.assertFalse((self.source / "nested").exists())

    def test_stage_same_root_has_no_hash_copy_or_overwrite(self):
        file = self.source / "video.mp4"
        file.write_bytes(b"unchanged media")
        cache = runner.StageCache(self.source, self.source / ".")
        with patch.object(runner, "sha256", side_effect=AssertionError("must not hash")), patch.object(runner.shutil, "copy2", side_effect=AssertionError("must not copy")):
            cache.persist()
        self.assertEqual(file.read_bytes(), b"unchanged media")

    def test_stage_nested_checkpoint_rejected(self):
        cache = runner.StageCache(self.source, self.source / "saved")
        with self.assertRaisesRegex(runner.PipelineError, "dentro do outro"):
            cache.persist()

    def test_symlink_snapshot_copies_blob_once_and_preserves_relative_link(self):
        blob = self.blob()
        snapshot = self.source / "huggingface/hub/models--google--model/snapshots/revision/model.safetensors"
        self.symlink_or_skip(snapshot, blob)
        with patch.object(bootstrap.shutil, "copy2", wraps=bootstrap.shutil.copy2) as copier:
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertEqual(copier.call_count, 1)
        target = self.destination / snapshot.relative_to(self.source)
        self.assertTrue(target.is_symlink())
        self.assertFalse(Path(os.readlink(target)).is_absolute())
        self.assertEqual(target.read_bytes(), blob.read_bytes())
        self.assertEqual(target.resolve(), (self.destination / blob.relative_to(self.source)).resolve())

    def test_drive_without_symlink_keeps_only_blob_and_metadata(self):
        relative, reference = self.link_metadata()
        with patch.object(Path, "symlink_to", side_effect=OSError("Drive does not support symlinks")), patch.object(bootstrap.shutil, "copy2", wraps=bootstrap.shutil.copy2) as copier:
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertEqual(copier.call_count, 1)
        self.assertFalse((self.destination / relative).exists())
        metadata = json.loads((self.destination / bootstrap.CACHE_LINKS_FILE).read_text())
        self.assertEqual(metadata["links"], {relative: reference})
        physical = [path for path in self.destination.rglob("*") if path.is_file() and path.name != bootstrap.CACHE_LINKS_FILE]
        self.assertEqual(physical, [self.destination / reference])

    def test_restore_recreates_link_from_drive_metadata(self):
        relative, reference = self.link_metadata()
        probe = self.root / "symlink-probe"
        self.symlink_or_skip(probe, self.source / reference)
        with patch.object(Path, "symlink_to", side_effect=OSError("Drive does not support symlinks")):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        restored = self.root / "restored"
        bootstrap.copy_missing(self.destination, restored, preserve_cache_links=True)
        self.assertTrue((restored / relative).is_symlink())
        self.assertEqual((restored / relative).read_bytes(), (restored / reference).read_bytes())

    def test_external_source_symlink_rejected_without_copy(self):
        external = self.root / "private.txt"
        external.write_bytes(b"must not leak")
        self.symlink_or_skip(self.source / "snapshot", external)
        with self.assertRaisesRegex(RuntimeError, "fora do diretório"):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertFalse(self.destination.exists())

    def test_linked_destination_parent_cannot_escape(self):
        blob = self.blob()
        self.destination.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        self.symlink_or_skip(self.destination / "huggingface", outside)
        with self.assertRaisesRegex(RuntimeError, "fora do cache"):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertEqual(list(outside.iterdir()), [])

    def test_invalid_link_metadata_rejected(self):
        reference = self.blob().relative_to(self.source).as_posix()
        for relative in ("../private", "/private", "C:/private", "..\\private", bootstrap.CACHE_LINKS_FILE):
            with self.subTest(relative=relative):
                self.link_metadata(reference=reference, relative=relative)
                with self.assertRaises(RuntimeError):
                    bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertFalse(self.destination.exists())

    def test_metadata_reference_traversal_and_missing_blob_rejected(self):
        for reference in ("../private", "missing-blob"):
            with self.subTest(reference=reference):
                self.link_metadata(reference=reference)
                with self.assertRaises(RuntimeError):
                    bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)

    def test_metadata_wrong_type_fails_cleanly(self):
        (self.source / bootstrap.CACHE_LINKS_FILE).write_text("[]")
        with self.assertRaisesRegex(RuntimeError, "Formato"):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)

    def test_stored_metadata_from_other_revisions_is_preserved(self):
        self.destination.mkdir()
        old_blob = self.blob(self.destination, "older", b"older revision")
        old_relative, old_reference = self.link_metadata(self.destination, reference=old_blob.relative_to(self.destination).as_posix(), relative="snapshots/old/model.safetensors")
        relative, reference = self.link_metadata()
        with patch.object(Path, "symlink_to", side_effect=OSError("unsupported")):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        metadata = json.loads((self.destination / bootstrap.CACHE_LINKS_FILE).read_text())
        self.assertEqual(metadata["links"], {old_relative: old_reference, relative: reference})

    def test_verified_duplicate_removed_when_links_unavailable(self):
        relative, reference = self.link_metadata()
        self.destination.mkdir()
        target = self.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((self.source / reference).read_bytes())
        with patch.object(Path, "symlink_to", side_effect=OSError("unsupported")):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertFalse(target.exists())
        self.assertEqual((self.destination / reference).read_bytes(), (self.source / reference).read_bytes())

    def test_different_existing_snapshot_not_deleted_in_fallback(self):
        relative, reference = self.link_metadata()
        self.destination.mkdir()
        target = self.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"different, must preserve")
        with patch.object(Path, "symlink_to", side_effect=OSError("unsupported")), self.assertRaisesRegex(RuntimeError, "diverge"):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        self.assertEqual(target.read_bytes(), b"different, must preserve")
        self.assertFalse((self.destination / bootstrap.CACHE_LINKS_FILE).exists())

    def test_divergent_snapshot_rejected_before_successful_symlink_attempt(self):
        relative, reference = self.link_metadata()
        self.destination.mkdir()
        target = self.destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"different, must preserve")
        with patch.object(Path, "symlink_to") as symlink, self.assertRaisesRegex(RuntimeError, "diverge"):
            bootstrap.copy_missing(self.source, self.destination, preserve_cache_links=True)
        symlink.assert_not_called()
        self.assertEqual(target.read_bytes(), b"different, must preserve")
        self.assertFalse((self.destination / bootstrap.CACHE_LINKS_FILE).exists())

    def test_matching_snapshot_verified_before_successful_atomic_replacement(self):
        canonical = self.blob(self.destination)
        target = self.destination / "snapshots/revision/model.safetensors"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical.read_bytes())
        with patch.object(bootstrap, "file_digest", wraps=bootstrap.file_digest) as digest, patch.object(Path, "symlink_to") as symlink, patch.object(Path, "replace") as replace:
            bootstrap._restore_cache_link(self.destination, target.relative_to(self.destination).as_posix(), canonical.relative_to(self.destination).as_posix())
        self.assertEqual(digest.call_count, 2)
        symlink.assert_called_once()
        replace.assert_called_once_with(target)
        self.assertEqual(target.read_bytes(), canonical.read_bytes())

    def test_copy_incomplete_temporaries_ignored(self):
        for name in ("download.lock", "download.incomplete", "download.copying", "download.writing", "download.link-writing"):
            (self.source / name).write_bytes(b"unfinished")
        (self.source / "ready.json").write_bytes(b"ready")
        bootstrap.copy_missing(self.source, self.destination)
        self.assertEqual([path.name for path in self.destination.iterdir()], ["ready.json"])


if __name__ == "__main__":
    unittest.main()
