#!/bin/bash

epoch_num=20
lr=1e-4
bs=32768  # 16384
d=1024
width_factor=1  # 8
logit_scale=20
logit_bias=-10

text_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption.h5"
# extra_text_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_shortSV_captions.h5"
image_embedding_list="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat.h5"
output_name="dinov2_nv2_cc3m_raw_10000_semisupsail_a=1.0_semisupot_a=0.0001_sh_e=0.1_20_an_e=0.01_100_check_after_refactoring_debugging_2"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

python /home/eml/simon.roschmann/ot-alignment/main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 1 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 24 \
    --optimizer lion \
    --siglip \
    --wd 1e-7 \
    --target-dimension $d \
    --linear-type linear \
    --width-factor $width_factor \
    --log-every-n-steps 5 \
    --wandb-project-name sail_train \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias \
    --logs /lustre/groups/eml/projects/sroschmann/ot_logs \
    --hdf5 \
    --ot \
    --semisupervised \
    --n_supervised_pairs 10000 \
    --batch-size-supervised 10000 \
    --alpha_semisupervised_sail 1.0 \
    --alpha_semisupervised_ot 0.0001 \
    --anchor_center \
    --anchor_whiten \
    --anchor_relrenorm \
    --anchor_lam_x 0.4 \
    --anchor_lam_y 0.2 \
    --epsilon_sinkhorn_anchor 0.01 \
    --n_iters_sinkhorn_anchor 100 \
    --epsilon_sinkhorn_shared 0.1 \
    --n_iters_sinkhorn_shared 20 \
    --debugging
    # --alpha_semisupervised_clusters 0.0001 \
    # --semisupervised_clusters 512 \
    # --outlier_fraction 0.01 \
    # --min_cluster_size 5 \

    # --alpha_semisupervised_clusters 0.0001 \
    # --semisupervised_clusters 256 \
    # --outlier_fraction 0.05 \
    # --min_cluster_size 5 \

    # --debugging
    # --unbalanced \
    # --tau_x 2.0 \
    # --tau_y 5.0
    # --anchor_rank_k_x 256 \
    # --anchor_rank_k_y 512

    # --alpha_semisupervised_clusters 0.0001 \
    # --semisupervised_clusters 256 \
    # --outlier_fraction 0.05 \
    # --min_cluster_size 5 \
    # --alpha_semisupervised_ot 0.0001 \
    # --supervised \
    # --alpha_supervised_sail 1.0
    # --n_iters_sinkhorn_shared 5 \
    # --epsilon_sinkhorn_shared 0.02
    # --alpha_supervised_implicit 1.0 \
    # --alpha_semisupervised_ot 0.0 \
    # --n_supervised_pairs 10000 \
    # --batch-size-supervised 10000 \
    # --n_iters_sinkhorn_anchor 20 \
    # --sinkhorn \
    # --epsilon 0.01 \
    # --n_iters_sinkhorn 5 \
    # --alpha_supervised_explicit 0 \
    # --alpha_supervised_implicit 1 \
    # --alpha_marginal 0 \
    # --alpha_unsupervised 0
    # --extra-text-embedding-list $extra_text_embedding_list \

if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
