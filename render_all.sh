#!/usr/bin/env bash

EXP_ROOT="/data5/sohyunkang/experiments2"
SCRIPT="/home/sohyunkang/asd_fmri2/render_hitranet_from_saved_aux.py"
AAL="/home/sohyunkang/asd_fmri2/data/ROI_MNI_V4.txt"
PYTHON="/home/sohyunkang/miniconda3/envs/asd_fmri/bin/python"

find "$EXP_ROOT" \
    -type f \
    -name "fold*_best_aux.pt" \
    -print0 |
while IFS= read -r -d '' aux; do
    aux_dir=$(dirname "$aux")
    aux_name=$(basename "$aux")
    fold_name=${aux_name%_best_aux.pt}
    output_mp4="$aux_dir/${fold_name}_trajectory_all.mp4"

    if [ -f "$output_mp4" ]; then
        echo "[SKIP] Already exists: $output_mp4"
        continue
    fi

    echo "========================================"
    echo "[RUN] $aux"

    if "$PYTHON" "$SCRIPT" \
        --aux "$aux" \
        --aal_txt "$AAL" \
        --split all \
        --seed_aggregation mean
    then
        echo "[DONE] $aux"
    else
        echo "[FAILED] $aux"
    fi
done

echo "========================================"
echo "All rendering jobs finished."