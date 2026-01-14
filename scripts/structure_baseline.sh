#!/bin/bash

epoch_num=20
lr=1e-4
bs=24576  # 32768 resulted in OOM
d=1024
width_factor=1
logit_scale=20.0

# image_model="facebook/dinov3-vitl16-pretrain-lvd1689m"
image_model="facebook/dinov2-large"
# text_model="Qwen/Qwen3-Embedding-8B"
# text_model="nvidia/llama-embed-nemotron-8b"
text_model="nvidia/NV-Embed-v2"

base_embedding_dir="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data"

supervised_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat.h5"
unsupervised_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat.h5" # imagenet1k_concat.h5"
supervised_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption.h5" # cc3m_raw_caption.h5"
unsupervised_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption.h5" # wikitext103_raw_caption.h5"

val_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat_validation.h5"
val_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption_validation.h5"

output_dir="/lustre/groups/eml/projects/sroschmann/ot_logs"

current_time=$(date +%Y-%m-%d_%H-%M-%S)
output_name="${current_time}_${image_model##*/}_${text_model##*/}_structure_baseline_lam=1000_warmup=500"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

python /home/eml/simon.roschmann/ot-alignment/main.py \
    --supervised_text_embedding $supervised_text_embedding \
    --supervised_image_embedding $supervised_image_embedding \
    --unsupervised_text_embedding $unsupervised_text_embedding \
    --unsupervised_image_embedding $unsupervised_image_embedding \
    --val_image_embedding $val_image_embedding \
    --val_text_embedding $val_text_embedding \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 10 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 24 \
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
    --logs /lustre/groups/eml/projects/sroschmann/ot_logs \
    --hdf5 \
    --semisupervised \
    --n_supervised_pairs 10000 \
    --batch-size-supervised 10000 \
    --n_unsupervised_text 1000000 \
    --n_unsupervised_image 1000000 \
    --structure \
    --structure_temperature 0.05 \
    --structure_normalize_latents \
    --structure_warmup_steps 500 \
    --structure_lambda 1000 \
    --structure_levels 1
    # --debugging