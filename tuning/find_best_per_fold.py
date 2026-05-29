import os
import json
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, required=True, help="fold number (1~5)")
args = parser.parse_args()

FOLD = args.fold

results = []

folders = glob.glob("/data5/sohyunkang/experiments/w*_hitranet_2026*")

for folder in folders:

    json_path = os.path.join(folder, f"fold{FOLD}_metrics.json")

    if not os.path.exists(json_path):
        continue

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        auc = data.get("auc", None)
        sens = data.get("sensitivity", None)
        spec = data.get("specificity", None)

        if auc is None:
            continue

        results.append({
            "folder": folder,
            "auc": auc,
            "sensitivity": sens,
            "specificity": spec
        })

    except Exception as e:
        print(f"❌ Error reading {json_path}: {e}")

# 정렬 (AUC 기준)
results = sorted(results, key=lambda x: x["auc"], reverse=True)

# 출력
print(f"\n===== Fold {FOLD} Ranking =====\n")

for i, r in enumerate(results, 1):
    print(f"[{i:02d}] {r['folder']}")
    print(f"     AUC: {r['auc']:.4f} | Sens: {r['sensitivity']:.4f} | Spec: {r['specificity']:.4f}")