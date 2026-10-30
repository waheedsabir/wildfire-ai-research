"""Smoke tests on synthetic data; not wildfire model performance tests."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class TestTrainingPipeline(unittest.TestCase):
    def test_one_cpu_epoch_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for subset in ('train', 'val'):
                for cls, color in (('fire', (230, 100, 50)), ('no_fire', (50, 150, 210))):
                    loc = root / 'data' / subset / cls
                    loc.mkdir(parents=True)
                    for i in range(2):
                        Image.new('RGB', (32, 32), color).save(loc / f'fake-{i}.png')
            proc = subprocess.run([
                sys.executable, str(ROOT / 'training' / 'train_ddp.py'),
                '--data-root', str(root / 'data'), '--model', 'tiny_cnn',
                '--epochs', '1', '--batch-size', '2', '--image-size', '32',
                '--out', str(root / 'outputs')
            ], capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue((root / 'outputs' / 'last.pt').exists())
            self.assertEqual(json.loads((root / 'outputs' / 'config.json').read_text())['world_size'], 1)
            report = json.loads((root / 'outputs' / 'metrics.jsonl').read_text().splitlines()[0])
            self.assertEqual(len(report['val_confusion_matrix']), 2)
            self.assertEqual(sum(map(sum, report['val_confusion_matrix'])), 4)


if __name__ == '__main__':
    unittest.main()
