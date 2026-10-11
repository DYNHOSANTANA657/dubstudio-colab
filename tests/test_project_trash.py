"""Offline production-JavaScript regression for recoverable project deletion."""
import shutil
import subprocess
import unittest
from pathlib import Path


class ProjectTrashUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is required for production UI tests")
    def test_project_trash_actions(self):
        result = subprocess.run([shutil.which("node"), str(Path(__file__).with_suffix(".js"))],
                                capture_output=True, text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
