"""Inspect visualization index.json entries for Class 4."""

import json
from pathlib import Path

def inspect_viz_indices():
    for split in ["train", "valid", "test"]:
        idx_path = Path(f"data/reports/visualizations/aadhaar/{split}/index.json")
        if not idx_path.exists():
            continue
        with open(idx_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"\n--- Split {split} (Total sheets: {len(data.get('sheets', []))}) ---")
        for sheet in data.get("sheets", []):
            if sheet.get("name") in ["class_4", "anomaly_rare_class_4", "anomaly_polygon_annotation", "anomaly_duplicate_class_4"]:
                print(f"Sheet: {sheet.get('name')} | Tiles count: {len(sheet.get('tiles', []))}")
                for tile in sheet.get("tiles", [])[:5]:
                    print(f"  Tile: {tile.get('stem')} | Classes: {tile.get('classes')} | Anomaly: {tile.get('anomaly_reason')}")

if __name__ == "__main__":
    inspect_viz_indices()
