"""Check for filename collisions and count detector targets."""

from pathlib import Path
from collections import Counter

def check_filenames():
    aadhaar_proc = Path("data/processed/aadhaar")
    pan_proc = Path("data/processed/pan")
    
    a_imgs = set(p.name for p in aadhaar_proc.glob("*/**/images/*.*"))
    p_imgs = set(p.name for p in pan_proc.glob("*/**/images/*.*"))
    
    print(f"Aadhaar processed images: {len(a_imgs)}")
    print(f"PAN processed images: {len(p_imgs)}")
    
    collision = a_imgs & p_imgs
    print(f"Filename collisions between Aadhaar and PAN: {len(collision)}")
    
    # Check split breakdown
    for split in ["train", "valid", "test"]:
        a_split = list((aadhaar_proc / split / "images").glob("*.*"))
        p_split = list((pan_proc / split / "images").glob("*.*"))
        print(f"Split '{split}': Aadhaar={len(a_split)}, PAN={len(p_split)}, Total={len(a_split) + len(p_split)}")

if __name__ == "__main__":
    check_filenames()
