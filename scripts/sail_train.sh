#!/bin/bash

epoch_num=20
lr=1e-5
bs=32768
d=1024
width_factor=8
logit_scale=20
logit_bias=-10

text_embedding_list="/lustre/groups/eml/projects/sroschmann/tensor_data/text_embedding/NV-Embed-v2/cc3m_raw_caption"
# extra_text_embedding_list="/dss/mcmlscratch/07/ga27tus3/mmap_data/NV-Embed-v2/dreamclipcc12m_shortSV_captions.mmap"
image_embedding_list="/lustre/groups/eml/projects/sroschmann/tensor_data/image_embedding/dinov2-large/cc3m_concat"
# metadata_path="/dss/mcmlscratch/07/ga27tus3/mmap_data/metadata.pt"
output_name="sail_dinov2l_nv2_cc3m_raw_full"

python /home/eml/simon.roschmann/ot-alignment/main.py \
    --text-embedding-list $text_embedding_list \
    --image-embedding-list $image_embedding_list \
    --val-frequency 1 \
    --dataset-type embedding \
    --seed 42 \
    --resume latest \
    --save-frequency 2 \
    --report-to wandb \
    --batch-size $bs \
    --lr $lr \
    --epochs $epoch_num \
    --workers 8 \
    --optimizer lion \
    --siglip \
    --wd 1e-7 \
    --target-dimension $d \
    --linear-type star \
    --log-every-n-steps 5 \
    --wandb-project-name sail_train \
    --name $output_name \
    --logit_scale $logit_scale \
    --logit_bias $logit_bias \
    --logs /lustre/groups/eml/projects/sroschmann/logs
    # --extra-text-embedding-list $extra_text_embedding_list \
    # --metadata-path $metadata_path \


if [ $? -ne 0 ]; then
    echo "Training failed. Checking for checkpoints..."
    
    if ls ./logs/$output_name/checkpoints/*.pt 1> /dev/null 2>&1; then
        echo "Checkpoint file(s) found. Keeping the log directory."
    else
        echo "No checkpoint files found. Cleaning up ./logs/${output_name}"
        rm -rf ./logs/$output_name
    fi
fi
