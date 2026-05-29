import os
import glob
import shutil
import json

ROOT = "/data5/sohyunkang/experiments"

folders = glob.glob(os.path.join(ROOT, "w*_hitranet_2026*"))

deleted = []
kept = []

for folder in folders:
    cv_path = os.path.join(folder, "cv_results.json")

    if not os.path.exists(cv_path):
        print(f"🗑️ Deleting (no cv_results): {folder}")
        shutil.rmtree(folder)
        deleted.append(folder)
        continue

    # ✅ auc_mean 읽기
    try:
        with open(cv_path, "r") as f:
            data = json.load(f)

        auc_mean = data.get("auc_mean", None)

        if auc_mean is None:
            print(f"⚠️ No auc_mean → keep: {folder}")
            kept.append(folder)
            continue

        if auc_mean <= 0.65:
            print(f"🗑️ Deleting (low auc_mean={auc_mean:.4f}): {folder}")
            shutil.rmtree(folder)
            deleted.append(folder)
        else:
            kept.append(folder)

    except Exception as e:
        print(f"⚠️ Error reading {cv_path}: {e}")
        kept.append(folder)

print("\n========== SUMMARY ==========")
print(f"Deleted: {len(deleted)}")
print(f"Kept   : {len(kept)}")