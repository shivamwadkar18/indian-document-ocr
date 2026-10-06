"""Analyze what visual elements are present in the Class 4 crop regions."""

from pathlib import Path
from PIL import Image
import numpy as np

def inspect_class_4_crops():
    raw_dir = Path("data/raw/aadhaar")
    label_files = sorted(raw_dir.glob("*/**/labels/*.txt"))
    
    crops_info = []
    
    for lbl_path in label_files:
        lines = [l.strip() for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
        img_path = lbl_path.parent.parent / "images" / f"{lbl_path.stem}.jpg"
        if not img_path.exists():
            continue
            
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if cls_id == 4 and len(coords) == 4:
                cx, cy, w, h = coords
                img = Image.open(img_path)
                iw, ih = img.size
                xmin = int((cx - w/2) * iw)
                ymin = int((cy - h/2) * ih)
                xmax = int((cx + w/2) * iw)
                ymax = int((cy + h/2) * ih)
                
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(iw, xmax)
                ymax = min(ih, ymax)
                
                crop = img.crop((xmin, ymin, xmax, ymax))
                # Check mean brightness, variance, aspect ratio
                arr = np.array(crop.convert("L"))
                if arr.size > 0:
                    mean_val = float(np.mean(arr))
                    std_val = float(np.std(arr))
                else:
                    mean_val, std_val = 0, 0
                    
                crops_info.append({
                    "stem": lbl_path.stem,
                    "box": (cx, cy, w, h),
                    "crop_size": crop.size,
                    "aspect": w / h if h > 0 else 0,
                    "mean_brightness": mean_val,
                    "std_contrast": std_val,
                })
                
    print(f"Examined {len(crops_info)} bounding box crops of Class 4.")
    print(f"Crop width (px): min={min(c['crop_size'][0] for c in crops_info)}, max={max(c['crop_size'][0] for c in crops_info)}, median={np.median([c['crop_size'][0] for c in crops_info]):.1f}")
    print(f"Crop height (px): min={min(c['crop_size'][1] for c in crops_info)}, max={max(c['crop_size'][1] for c in crops_info)}, median={np.median([c['crop_size'][1] for c in crops_info]):.1f}")
    print(f"Aspect ratios (w/h): min={min(c['aspect'] for c in crops_info):.2f}, max={max(c['aspect'] for c in crops_info):.2f}, median={np.median([c['aspect'] for c in crops_info]):.2f}")
    
    # Check extreme aspect ratios
    tall_crops = [c for c in crops_info if c["aspect"] < 1.0]
    wide_crops = [c for c in crops_info if c["aspect"] > 4.0]
    mid_crops = [c for c in crops_info if 1.0 <= c["aspect"] <= 4.0]
    print(f"Tall crops (aspect < 1.0, vertical orientation): {len(tall_crops)}")
    print(f"Mid crops (1.0 <= aspect <= 4.0): {len(mid_crops)}")
    print(f"Wide crops (aspect > 4.0, long horizontal text block): {len(wide_crops)}")

if __name__ == "__main__":
    inspect_class_4_crops()
