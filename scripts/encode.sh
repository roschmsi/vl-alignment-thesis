vision_model="facebook/dinov2-large"
text_model="nvidia/NV-Embed-v2"
# vision_model="google/siglip2-large-patch16-384"
# text_model="google/siglip2-large-patch16-384"
# data="imagenet1k"
# domain="image"
# data="coco"
# coco_caption_index=4
# domain="image"
# data="diffusion_db"
# domain="text"
data="coco"
domain="image"
batch_size=32
source_caption="raw_caption"
agg_mode="concat"
input_dir="/lustre/groups/eml/datasets/coco"
# input_dir="/lustre/groups/eml/projects/sroschmann/diffusion_db"
output_dir="/lustre/groups/eml/projects/sroschmann/ot-alignment"

python /home/eml/simon.roschmann/ot-alignment/encode.py \
--domain "$domain" \
--vision_model_name "$vision_model" \
--text_model_name "$text_model" \
--batch_size "$batch_size" \
--data "$data" \
--resume \
--source_caption "$source_caption" \
--agg_mode "$agg_mode" \
--num_workers 4 \
--input_dir "$input_dir" \
--output_dir "$output_dir"
# --coco_caption_index "$coco_caption_index"