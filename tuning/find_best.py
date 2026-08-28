import os
import json
import glob

results = []

# find folders matching pattern
folders = glob.glob("./experiments/*")

for folder in folders:
    json_path = os.path.join(folder, "cv_results.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            
            avgs = data.get("mean", None)
            auc = avgs.get("auc", None)
            
            if auc is not None:
                results.append((folder, auc))
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

# sort by auc_mean (descending)
results.sort(key=lambda x: x[1], reverse=True)

# print results
for folder, auc in results:
    print(f"{folder}: {auc}")