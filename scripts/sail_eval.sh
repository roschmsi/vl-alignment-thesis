#!/bin/bash

vision_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"

CKPT_DIR="/lustre/groups/eml/projects/sroschmann/ot_logs/dinov2_nv2_cc3m_raw_10000_semisupsail_a=1.0_semisupot_a=0.0001_sh_e=0.1_20_an_e=0.01_100_full_fixrn/checkpoints"
DATASET_ROOT_DIR="/lustre/groups/eml/projects/sroschmann/data"

shopt -s nullglob

# Get all checkpoints matching epoch_*.pt and sort them
ckpts=("$CKPT_DIR"/epoch_*.pt)
IFS=$'\n' ckpts=($(printf '%s\n' "${ckpts[@]}" | sort -V))
unset IFS

if [ ${#ckpts[@]} -eq 0 ]; then
    echo "No checkpoints found in $CKPT_DIR"
    exit 1
fi

for ckpt_path in "${ckpts[@]}"; do
    fname=$(basename "$ckpt_path")
    epoch=${fname#epoch_}
    epoch=${epoch%.pt}

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
