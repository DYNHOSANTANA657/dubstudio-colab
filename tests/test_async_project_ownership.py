"""Execute the production JavaScript ownership and async regression scenarios."""
import shutil
import subprocess
import unittest
from pathlib import Path


class AsyncProjectOwnershipUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js required")
    def test_production_async_import_and_review_ownership(self):
        result = subprocess.run(
            [shutil.which("node"), str(Path(__file__).with_suffix(".js"))],
            capture_output=True, text=True, encoding="utf-8", timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS (30 scenarios)", result.stdout)
