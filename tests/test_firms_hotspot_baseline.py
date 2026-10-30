"""Unit test using artificial rows, NOT real wildfire observations or model validation."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from src.firms_hotspot_baseline import process

class TestFirmsSummary(unittest.TestCase):
    def test_validates_coordinate_range(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/"synthetic.csv"
            with source.open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=["latitude","longitude","acq_date"])
                writer.writeheader()
                writer.writerows([
                    {"latitude":"34.1","longitude":"-118.2","acq_date":"2025-01-01"},
                    {"latitude":"100","longitude":"-118.2","acq_date":"2025-01-01"}])
            process(source,root/"out")
            result=json.loads((root/"out"/"hotspot_summary.json").read_text())
            self.assertEqual(result["valid_hotspot_points"],1)
            self.assertEqual(result["status"],"descriptive_baseline_not_ai_model")

if __name__ == "__main__":
    unittest.main()
