import shutil,subprocess,unittest
from pathlib import Path

class FaceRepairUiTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'),'Node.js required')
    def test_production_face_repair_guards(self):
        process=subprocess.run([shutil.which('node'),str(Path(__file__).with_suffix('.js'))],capture_output=True,text=True,encoding='utf-8',timeout=20)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)
        self.assertIn('PASS',process.stdout)
