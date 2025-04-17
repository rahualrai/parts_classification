import os
import glob
import cv2
import albumentations as A
from tqdm import tqdm

# === CONFIG ===
INPUT_ROOT  = "data/parts_removed_bg"  # your existing train/ & test/ folders
OUTPUT_ROOT = "data/parts"             # NEW folder where sythetic dataset will live
AUG_PER_IMG = 20                       # # of variants per original

transform = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.GaussNoise(var_limit=(10.0,50.0), p=0.3)
])


# === CREATE OUTPUT FOLDERS ===
for split in os.listdir(INPUT_ROOT):
    split_in = os.path.join(INPUT_ROOT, split)
    if not os.path.isdir(split_in):
        continue
    for cls in os.listdir(split_in):
        out_dir = os.path.join(OUTPUT_ROOT, split, cls)
        os.makedirs(out_dir, exist_ok=True)

# === PROCESS & AUGMENT ===
for split in sorted(os.listdir(INPUT_ROOT)):
    split_in = os.path.join(INPUT_ROOT, split)
    if not os.path.isdir(split_in):
        continue

    for cls in sorted(os.listdir(split_in)):
        cls_in  = os.path.join(split_in, cls)
        cls_out = os.path.join(OUTPUT_ROOT, split, cls)
        if not os.path.isdir(cls_in):
            continue

        # grab all .jpg/.png files
        files = glob.glob(os.path.join(cls_in, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(cls_in, "*.[pP][nN][gG]"))
        for path in tqdm(files, desc=f"{split}/{cls}"):
            # load & convert to RGB
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            base = os.path.splitext(os.path.basename(path))[0]

            # 1) save the original (normalized) image
            orig_out = os.path.join(cls_out, f"{base}_orig.png")
            cv2.imwrite(orig_out, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            # 2) generate augmentations
            for i in range(1, AUG_PER_IMG + 1):
                aug_img = transform(image=img)["image"]
                out_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                fname = f"{base}_aug{i:02d}.png"
                cv2.imwrite(os.path.join(cls_out, fname), out_bgr)
