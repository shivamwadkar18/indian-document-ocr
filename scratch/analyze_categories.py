"""Detailed spatial and semantic breakdown of Class 4 across the 3 categories."""

from pathlib import Path
from collections import defaultdict

def analyze_categories():
    raw_dir = Path("data/raw/aadhaar")
    label_files = sorted(raw_dir.glob("*/**/labels/*.txt"))
    
    entries = []
    for lbl_path in label_files:
        lines = [l.strip() for l in lbl_path.read_text("utf-8-sig").splitlines() if l.strip()]
        anns = []
        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            anns.append((cls_id, coords))
            
        c4_list = [c for cls, c in anns if cls == 4]
        if c4_list:
            other_classes = [cls for cls, _ in anns if cls != 4]
            entries.append({
                "stem": lbl_path.stem,
                "split": lbl_path.parent.parent.name,
                "c4_list": c4_list,
                "other_classes": other_classes,
                "all_anns": anns,
            })
            
    print(f"Total images with Class 4: {len(entries)}")
    
    cat_full = [e for e in entries if set([0,1,2,3]).issubset(set(e["other_classes"]))]
    cat_zero_only = [e for e in entries if set(e["other_classes"]) == {0}]
    cat_none = [e for e in entries if len(e["other_classes"]) == 0]
    cat_other = [e for e in entries if e not in cat_full and e not in cat_zero_only and e not in cat_none]
    
    print(f"Category 1 (Full card: 0, 1, 2, 3 + 4): {len(cat_full)} images")
    print(f"Category 2 (Number + 4 only: 0 + 4): {len(cat_zero_only)} images")
    print(f"Category 3 (Class 4 only): {len(cat_none)} images")
    print(f"Category 4 (Other combos): {len(cat_other)} images")
    
    print("\n--- Category 1 Analysis (Full card) ---")
    for e in cat_full[:10]:
        print(f"Stem: {e['stem'][:35]} | Other: {e['other_classes']}")
        # Print positions of all classes
        for cls, coords in e["all_anns"]:
            if len(coords) == 4:
                print(f"  Class {cls}: cx={coords[0]:.3f}, cy={coords[1]:.3f}, w={coords[2]:.3f}, h={coords[3]:.3f}")
            else:
                print(f"  Class {cls} (Polygon {len(coords)//2} pts): {coords[:4]}...")
                
    print("\n--- Category 2 Analysis (0 + 4 only) ---")
    for e in cat_zero_only[:10]:
        print(f"Stem: {e['stem'][:35]} | Other: {e['other_classes']}")
        for cls, coords in e["all_anns"]:
            if len(coords) == 4:
                print(f"  Class {cls}: cx={coords[0]:.3f}, cy={coords[1]:.3f}, w={coords[2]:.3f}, h={coords[3]:.3f}")
            else:
                print(f"  Class {cls} (Polygon {len(coords)//2} pts): {coords[:4]}...")
                
    print("\n--- Category 3 Analysis (Class 4 only) ---")
    for e in cat_none:
        print(f"Stem: {e['stem'][:35]}")
        for cls, coords in e["all_anns"]:
            if len(coords) == 4:
                print(f"  Class {cls}: cx={coords[0]:.3f}, cy={coords[1]:.3f}, w={coords[2]:.3f}, h={coords[3]:.3f}")
            else:
                print(f"  Class {cls} (Polygon {len(coords)//2} pts): {coords[:4]}...")

if __name__ == "__main__":
    analyze_categories()
