#!/bin/bash

epoch_num=1000
lr=1e-4
bs=10000 # 32768
d=1024
width_factor=1 # 8
logit_scale=20
logit_bias=-10

# image_model="facebook/dinov3-vitl16-pretrain-lvd1689m"
image_model="facebook/dinov2-large"
# text_model="Qwen/Qwen3-Embedding-8B"
# text_model="nvidia/llama-embed-nemotron-8b"
text_model="nvidia/NV-Embed-v2"

base_embedding_dir="/lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data"

supervised_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat.h5"
supervised_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption.h5" # cc3m_raw_caption.h5"

val_image_embedding="${base_embedding_dir}/image_embedding/${image_model##*/}/cc3m_concat_validation.h5"
val_text_embedding="${base_embedding_dir}/text_embedding/${text_model##*/}/cc3m_raw_caption_validation.h5"

output_dir="/lustre/groups/eml/projects/sroschmann/ot_logs"

current_time=$(date +%Y-%m-%d_%H-%M-%S)
output_name="${current_time}_${image_model##*/}_${text_model##*/}_supervised_baseline"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

python /home/eml/simon.roschmann/ot-alignment/main.py \
    --supervised_text_embedding $supervised_text_embedding \
    --supervised_image_embedding $supervised_image_embedding \
    --val_image_embedding $val_image_embedding \
    --val_text_embedding $val_text_embedding \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 50 \
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
    --logit_bias $logit_bias \
    --logs /lustre/groups/eml/projects/sroschmann/ot_logs \
    --hdf5 \
    --ot \
    --supervised \
    --n_supervised_pairs 10000 \
    --alpha_supervised_sail 1.0
    # --nnclr \
    # --text_neighbors_path /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption_neighbors.npy \
    # --image_neighbors_path /lustre/groups/eml/projects/sroschmann/ot-alignment/tensor_data/image_embedding/dinov2-large/cc3m_concat_neighbors.npy \
    # --text_nn_positives 1 \
    # --image_nn_positives 1 \
    # --text_topk 1 \
    # --image_topk 1 \
    # --w_text_nn 1 \
    # --w_image_nn 0

#     --alpha_supervised_sail 1.0 \
    # --alpha_semisupervised_ot 0.0001 \
    # --epsilon_sinkhorn_shared 0.1 \
    # --n_iters_sinkhorn_shared 20 \
    # --epsilon_sinkhorn_anchor 0.01 \
    # --n_iters_sinkhorn_anchor 100 \
    # --anchor_lam_x 0.1 \
    # --anchor_lam_y 0.1 \
    # --n_unsupervised_image 1000000 \
    # --n_unsupervised_text 1000000
    # --debugging
    # --alpha_semisupervised_div 1.0 \
    # --divergence frobenius

    # --alpha_semisupervised_sail 1.0 \
    # --alpha_semisupervised_double_softmax 0.00001 \
    # --temperature_softmax 0.1 \

    # --semisupervised \
    # --n_supervised_pairs 10000 \
    # --batch-size-supervised 10000 \
    # --alpha_semisupervised_sail 1.0
    
    # --debugging
    # --n_unsupervised_image \
    # --n_unsupervised_text \
    
    # --debugging
    # --anchor_relrenorm
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
