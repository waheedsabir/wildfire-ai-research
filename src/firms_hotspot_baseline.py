#!/usr/bin/env python3
"""Exploratory NASA FIRMS CSV thermal-anomaly summary; NOT AI wildfire detection."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

def process(source: Path, output: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    features = []
    with source.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"latitude","longitude"}.issubset(reader.fieldnames):
            raise ValueError("CSV requires latitude and longitude columns")
        for row in reader:
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (TypeError, ValueError, KeyError):
                continue
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                continue
            day = row.get("acq_date") or "unknown"
            counts[day] += 1
            features.append({"type":"Feature","geometry":{"type":"Point","coordinates":[lon,lat]},
                "properties":{"acq_date":day,"satellite":row.get("satellite",""),
                              "confidence":row.get("confidence",""),"frp":row.get("frp","")}})
    summary = {"analysis_timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "source_file":source.name, "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "valid_hotspot_points":len(features),"counts_by_date":dict(sorted(counts.items())),
        "status":"descriptive_baseline_not_ai_model",
        "warning":"Thermal anomalies are not exact wildfire perimeters, smoke annotations or an operational warning."}
    (output/"hotspot_summary.json").write_text(json.dumps(summary,indent=2))
    (output/"hotspots.geojson").write_text(json.dumps({"type":"FeatureCollection","features":features}))
    print(f"Processed {len(features)} valid anomaly points")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    process(args.input, args.out)
