"""Optional Node regression for the production multi-language player."""
import shutil
import subprocess
import unittest
from pathlib import Path


class PreviewSelectionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for offline JavaScript UI regression")
    def test_production_preview_state_and_media_switch(self):
        script = Path(__file__).with_suffix(".js")
        result = subprocess.run([shutil.which("node"), str(script)], capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)
