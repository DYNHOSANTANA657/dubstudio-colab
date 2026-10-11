"""Honest operation labels only; no speech model, browser or real media."""
import ast
import shutil
import subprocess
import unittest
from pathlib import Path


class LipSyncAudioScopeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node is required for offline JavaScript UI regression")
    def test_audio_mix_scope_and_operation_switch_keep_original_controls(self):
        script = Path(__file__).with_suffix(".js")
        result = subprocess.run([shutil.which("node"), str(script)], capture_output=True,
                                text=True, encoding="utf-8", timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_analysis_stage_label_does_not_claim_uploaded_voice_association(self):
        source = Path(__file__).resolve().parents[1] / "colab/dubstudio_runner_v2.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        process = next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "process")
        labels = [node for node in ast.walk(process) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "analysis_label" for target in node.targets)]
        self.assertEqual(len(labels), 1)
        expression = compile(ast.Expression(labels[0].value), str(source), "eval")
        self.assertEqual(eval(expression, {"__builtins__": {}}, {"operation": "lipsync_video"}),
                         "Detectando rostos para o áudio enviado")
        self.assertEqual(eval(expression, {"__builtins__": {}}, {"operation": "dub_video"}),
                         "Confirmação dos rostos e locutores")
        calls = [node for node in ast.walk(process) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "stage"
                 and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "analysis"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[1], ast.Name)
        self.assertEqual(calls[0].args[1].id, "analysis_label")


if __name__ == "__main__":
    unittest.main()
