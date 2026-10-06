"""Perform comprehensive diagnostic analysis on Class 4 consistency."""

from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict, Counter

def analyze_class_4():
    raw_dir = Path("data/raw/aadhaar")
    label_files = sorted(raw_dir.glob("*/**/labels/*.txt"))
    
    records = []
    
    for lbl_path in label_files:
        lines = [l.strip() for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
        img_path = lbl_path.parent.parent / "images" / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            continue
            
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if cls_id == 4:
                records.append({
                    "stem": lbl_path.stem,
                    "split": lbl_path.parent.parent.name,
                    "coords": coords,
                    "is_poly": len(coords) > 4,
                    "img_path": img_path,
                    "all_lines": lines,
                })
                
    print(f"Total Class 4 records: {len(records)}")
    
    # 1. Variance in Box Dimensions
    boxes = [r for r in records if not r["is_poly"]]
    widths = [r["coords"][2] for r in boxes]
    heights = [r["coords"][3] for r in boxes]
    aspects = [w/h for w, h in zip(widths, heights)]
    
    print("\n--- Aspect Ratio and Size Variations ---")
    print(f"Aspect ratio range: {min(aspects):.2f} to {max(aspects):.2f} (Ratio max/min = {max(aspects)/min(aspects):.1f}x)")
    print(f"Aspect ratio std dev: {np.std(aspects):.2f}")
    print(f"Width std dev: {np.std(widths):.4f}")
    print(f"Height std dev: {np.std(heights):.4f}")
    
    # Compare with Class 0 (Aadhaar number) and Class 1 (DOB) and Class 3 (Name)
    # Let's compute variance for classes 0, 1, 2, 3 across the entire dataset to compare
    c0_boxes, c1_boxes, c2_boxes, c3_boxes = [], [], [], []
    for lbl_path in label_files:
        lines = [l.strip() for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) == 4:
                if cls_id == 0: c0_boxes.append(coords)
                elif cls_id == 1: c1_boxes.append(coords)
                elif cls_id == 2: c2_boxes.append(coords)
                elif cls_id == 3: c3_boxes.append(coords)
                
    print("\n--- Comparison of Spatial Stability Across Classes ---")
    for cid, name, c_boxes in [
        (0, "Aadhaar Number", c0_boxes),
        (1, "Date of Birth", c1_boxes),
        (2, "Gender", c2_boxes),
        (3, "Name", c3_boxes),
        (4, "Class 4 (Unknown)", boxes),
    ]:
        cxs = [b[0] if isinstance(b, list) else b["coords"][0] for b in c_boxes]
        cys = [b[1] if isinstance(b, list) else b["coords"][1] for b in c_boxes]
        ws = [b[2] if isinstance(b, list) else b["coords"][2] for b in c_boxes]
        hs = [b[3] if isinstance(b, list) else b["coords"][3] for b in c_boxes]
        asps = [w/h for w, h in zip(ws, hs)]
        
        print(f"Class {cid} ({name:15s}): N={len(c_boxes):4d} | "
              f"cx std={np.std(cxs):.3f} | cy std={np.std(cys):.3f} | "
              f"aspect median={np.median(asps):.2f}, std={np.std(asps):.2f}")

if __name__ == "__main__":
    analyze_class_4()
