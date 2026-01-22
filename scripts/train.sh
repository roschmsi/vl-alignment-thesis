#!/bin/bash

epoch_num=20
lr=1e-4
bs=32768
d=1024
width_factor=8
logit_scale=20
logit_bias=-10

# image_model="facebook/dinov3-vitl16-pretrain-lvd1689m"
image_model="facebook/dinov2-large"
# text_model="Qwen/Qwen3-Embedding-8B"
# text_model="nvidia/llama-embed-nemotron-8b"
text_model="nvidia/NV-Embed-v2"

base_embedding_dir="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data"

supervised_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat.h5"
supervised_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption.h5"

val_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat_validation.h5"
val_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption_validation.h5"

output_dir="/lustre/groups/eml/projects/sroschmann/ot_logs"

current_time=$(date +%Y-%m-%d_%H-%M-%S)
output_name="${current_time}_${image_model##*/}_${text_model##*/}_cc3m_sup=10k_unsup=1M_ep=100_lr=1e-3" # topk_x=256_topk_y=128"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

python /home/eml/simon.roschmann/vl-alignment-thesis/main.py \
    --supervised_text_embedding $supervised_text_embedding \
    --supervised_image_embedding $supervised_image_embedding \
    --val_image_embedding $val_image_embedding \
    --val_text_embedding $val_text_embedding \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 20 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 8 \
    --optimizer lion \
    --siglip \
    --wd 1e-5 \
    --target-dimension $d \
    --linear-type linear \
    --width-factor $width_factor \
    --log-every-n-steps 5 \
    --wandb-project-name semisupervised_alignment \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias \
    --logs $output_dir \
    --hdf5 \
    --supervised

if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi


BEST_CKPT="${output_dir}/${output_name}/checkpoints/epoch_best.pt"


if [ ! -f "$BEST_CKPT" ]; then
    echo "Error: Could not find best checkpoint at $BEST_CKPT"
    exit 1
fi

echo "########################################################"
echo "Training complete. Starting evaluation on: $BEST_CKPT"


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