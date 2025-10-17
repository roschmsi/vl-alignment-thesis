#!/bin/bash

# -----------------------MODEL SETTING------------------------
vision_model="facebook/dinov2-large"
# vision_model="facebook/dinov2-large"
# vision_model="ijepa-huge"
# vision_model="facebook/dinov2-giant"
# vision_model="openai/clip-vit-large-patch14"
# vision_model="dinov1-vitb16"
# vision_model="dinov1-resnet"
# vision_model="facebook/dinov2-giant"
# vision_model="mae-large"
# vision_model="aim-1B"
# vision_model="aim-600M"
# vision_model="ibot-large"

# text_model="sentence-transformers/all-mpnet-base-v2"
# text_model="Alibaba-NLP/gte-large-en-v1.5" 
# text_model="openai/clip-vit-large-patch14"
# text_model="Alibaba-NLP/gte-base-en-v1.5" 
# text_model="Alibaba-NLP/gte-Qwen2-1.5B-instruct"
# text_model="Alibaba-NLP/gte-Qwen2-7B-instruct"
text_model="nvidia/NV-Embed-v2"
# ------------------------------------------------------------ 

CKPT="/dss/mcmlscratch/07/ga27tus3/ot-alignment/logs/sail_dinov2l_nv2_cc12m_mmap_raw_shortSV_bs=8192_max_epoch=20/checkpoints/epoch_8.pt"
# CKPT="/dss/mcmlscratch/07/ga27tus3/ot-alignment/logs/dreamclip30m_NV2dinoL_bs_32768_lion_mean_lr_1e-5_star7XL_d1024_scale20_bias-10_multi_postext_s2/checkpoints/sail_dinov2l_nv2.pt"
DATASET_ROOT_DIR="/dss/mcmlscratch/07/ga27tus3/data"

# segmentation
for task in imagenetv1 COCO winoground MMVP
do
    # check if the checkpoint exists
    if [ ! -f $checkpoint_path ]; then
        echo "Checkpoint not found: $checkpoint_path"
        continue
    fi
    echo "########################################################"
    echo "Evaluating checkpoint: $checkpoint_path"

    python /dss/dsshome1/07/ga27tus3/ot-alignment/eval.py \
        --head-weights-path $CKPT \
        --task $task \
        --vision-model $vision_model \
        --text-model $text_model \
        --dataset_root_dir $DATASET_ROOT_DIR \
        --batch_size 16 \
        --agg_mode concat \
        --width_factor 8 \
        --target-dimension 1024 \
        --seg_task_config evaluation/seg_configs/cfg_voc20_SAIL.py
done
