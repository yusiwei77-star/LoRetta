import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_help_uses_only_paper_aligned_public_names(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "demo.py"), "--help"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--reference-image", result.stdout)
        self.assertIn("--sensed-image", result.stdout)
        self.assertIn("--matchability-threshold", result.stdout)
        self.assertNotIn("--ref-image", result.stdout)
        self.assertNotIn("--query-image", result.stdout)
        self.assertNotIn("--certainty-threshold", result.stdout)
