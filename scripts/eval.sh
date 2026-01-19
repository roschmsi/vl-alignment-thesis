#!/bin/bash

image_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"

# You can now point this to a folder OR a specific .pt file
output_dir="/lustre/groups/eml/projects/sroschmann/ot_logs"
output_name="2026-01-15_21-43-57_dinov2-large_NV-Embed-v2_cc3m_sup=10k_unsup=100k"

BEST_CKPT="${output_dir}/${output_name}/checkpoints/epoch_best.pt"

if [ ! -f "$BEST_CKPT" ]; then
    echo "Error: Could not find best checkpoint at $BEST_CKPT"
    exit 1
fi

echo "########################################################"
echo "Evaluation on: $BEST_CKPT"


# segmentation winoground MMVP
for task in imagenetv1 COCO; do
    echo "Task: $task"

    python /home/eml/simon.roschmann/ot-alignment/eval.py \
        --head-weights-path "$BEST_CKPT" \
        --task "$task" \
        --vision-model "$image_model" \
        --text-model "$text_model" \
        --dataset_root_dir "/lustre/groups/eml/projects/sroschmann/data" \
        --batch_size 32 \
        --agg_mode concat \
        --linear-type linear \
        --target-dimension 1024 \
        --seg_task_config evaluation/seg_configs/cfg_voc20_SAIL.py \
        --results_dir "${output_dir}/${output_name}/results"
done