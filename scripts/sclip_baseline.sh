#!/bin/bash

epoch_num=200
lr=1e-4
bs=32768
d=1024
width_factor=1
logit_scale=20.0

supervised_image_embedding="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat.h5"
unsupervised_image_embedding="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat.h5" # imagenet1k_concat.h5"
supervised_text_embedding="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption.h5" # cc3m_raw_caption.h5"
unsupervised_text_embedding="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption.h5" # wikitext103_raw_caption.h5"
# extra_text_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_shortSV_captions.h5"
# image_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat_first100k.h5"
# text_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption_first100k.h5"
output_name="a_dinov2_nv2_cc3m_nsup=10k_nunsup_100k_sclip_unimodal_pl=ot_unpaired_wimg=1_wtext=0_ep=200_logitsc=20_raw"
# semisupsail_a=1.0_semisupot_a=0.0001_sh_e=0.1_20_an_e=0.01_100_cca_lam=0.1"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

python /home/eml/simon.roschmann/ot-alignment/main.py \
    --supervised_text_embedding $supervised_text_embedding \
    --supervised_image_embedding $supervised_image_embedding \
    --unsupervised_text_embedding $unsupervised_text_embedding \
    --unsupervised_image_embedding $unsupervised_image_embedding \
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
    --wd 1e-4 \
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
    --n_unsupervised_text 100000 \
    --n_unsupervised_image 100000 \
    --sclip \
    --sclip_method pseudo-labels \
    --sclip_unpaired_modality image \
    --sclip_space unimodal \
    --sclip_pseudo_label_type ot \
    --sclip_weight_unpaired_images 1.0 \
    --sclip_weight_unpaired_texts 0.0 \
    --debugging