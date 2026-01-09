#!/bin/bash

vision_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"

# You can now point this to a folder OR a specific .pt file
CKPT_INPUT="/lustre/groups/eml/projects/sroschmann/ot_logs/a_dinov2vitl_qwen_cc3m_nsup=10k_nunsup=100k_supsail_a=1.0_semisupot_a=0.0001_validation_deb_1/checkpoints/epoch_best.pt"
DATASET_ROOT_DIR="/lustre/groups/eml/projects/sroschmann/data"

shopt -s nullglob
ckpts=()
SINGLE_FILE_MODE=false

# 1. Determine if input is a File or Directory
if [[ -f "$CKPT_INPUT" ]]; then
    echo "Single checkpoint provided."
    ckpts+=("$CKPT_INPUT")
    SINGLE_FILE_MODE=true
elif [[ -d "$CKPT_INPUT" ]]; then
    echo "Directory provided. searching for epoch_*.pt files..."
    ckpts=("$CKPT_INPUT"/epoch_*.pt)
    # Sort by version number
    IFS=$'\n' ckpts=($(printf '%s\n' "${ckpts[@]}" | sort -V))
    unset IFS
else
    echo "Error: '$CKPT_INPUT' is not a valid file or directory."
    exit 1
fi

freq=20

if [ ${#ckpts[@]} -eq 0 ]; then
    echo "No checkpoints found in $CKPT_INPUT"
    exit 1
fi

for ckpt_path in "${ckpts[@]}"; do
    fname=$(basename "$ckpt_path")
    epoch=${fname#epoch_}
    epoch=${epoch%.pt}

    # 2. Only apply frequency check if we are scanning a directory
    # Also ensures we don't try to do math on "best" if filename is epoch_best.pt
    if [ "$SINGLE_FILE_MODE" = false ]; then
        # Check if epoch is actually a number (handles epoch_best.pt in dir mode)
        if [[ "$epoch" =~ ^[0-9]+$ ]]; then
            if (( epoch % freq != 0 )); then
                continue
            fi
        fi
    fi

    echo "########################################################"
    echo "Evaluating checkpoint: $ckpt_path"

    # segmentation winoground MMVP
    for task in imagenetv1 COCO; do
        echo "Task: $task"

        python /home/eml/simon.roschmann/ot-alignment/eval.py \
            --head-weights-path "$ckpt_path" \
            --task "$task" \
            --vision-model "$vision_model" \
            --text-model "$text_model" \
            --dataset_root_dir "$DATASET_ROOT_DIR" \
            --batch_size 32 \
            --agg_mode concat \
            --linear-type linear \
            --target-dimension 1024 \
            --seg_task_config evaluation/seg_configs/cfg_voc20_SAIL.py \
            --results_dir /home/eml/simon.roschmann/ot-alignment/evaluation/eval_result
    done
done