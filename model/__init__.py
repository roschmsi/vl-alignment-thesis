from model.sclip import SemiSupervisedClipLoss
from optimal_transport.matching import FullMatchingModel
from .sail_model import AlignmentLayer, SAILModel, ShareLockAlignmentLayer
from .loss import ClipLoss, SigLipLoss, SigLipLossWithNNPositives
from typing import Union, Optional
import torch


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ("bf16", "pure_bf16"):
        input_dtype = torch.bfloat16
    elif precision in ("fp16", "pure_fp16"):
        input_dtype = torch.float16
    return input_dtype


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == "bf16":
        cast_dtype = torch.bfloat16
    elif precision == "fp16":
        cast_dtype = torch.float16
    elif precision == "fp32":
        cast_dtype = torch.float32
    return cast_dtype


def create_model(
    text_model_name: Optional[str] = None,
    vision_model_name: Optional[str] = None,
    head_weights_path: Optional[str] = None,
    vision_dimesion: int = 1536,
    text_dimension: int = 768,
    target_dimension: int = 512,
    precision: str = "fp32",
    device: Union[str, torch.device] = "cpu",
    linear_type: str = "star",
    logit_scale: float = 20.0,
    logit_bias: float = -10.0,
    agg_mode: str = "concat",
    width_factor: int = 8,
    sharelock: bool = False,
    hidden_states: bool = False,
    hidden_states_img_idx=None,
    hidden_states_text_idx=None,
    reconstruction: bool = False,
    reconstruction_type="linear",
    downsample=False,
):
    if isinstance(device, str):
        device = torch.device(device)

    LayerClass = ShareLockAlignmentLayer if sharelock else AlignmentLayer

    cast_dtype = get_cast_dtype(precision)
    if vision_model_name is not None and text_model_name is not None:
        model = SAILModel(
            text_model_name=text_model_name,
            vision_model_name=vision_model_name,
            target_dimension=target_dimension,
            vlhead_weights_path=head_weights_path,
            linear_type=linear_type,
            cast_dtype=cast_dtype,
            agg_mode=agg_mode,
            width_factor=width_factor,
            sharelock=sharelock,
            hidden_states=hidden_states,
            hidden_states_img_idx=hidden_states_img_idx,
            hidden_states_text_idx=hidden_states_text_idx,
            downsample=downsample,
        )
    else:
        model = LayerClass(
            vision_dimesion,
            text_dimension,
            target_dimension,
            linear_type=linear_type,
            cast_dtype=cast_dtype,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
            width_factor=width_factor,
            hidden_states=hidden_states,
            reconstruction=reconstruction,
            reconstruction_type=reconstruction_type,
        )
    model.to(device=device)
    return model


def create_loss(args):
    if args.nnclr:
        return SigLipLossWithNNPositives(
            w_text_nn=args.w_text_nn,
            w_image_nn=args.w_image_nn,
        )
    if args.sclip:
        print("Using S-CLIP loss")
        return SemiSupervisedClipLoss(
            method=args.sclip_method,
            unpaired_modality=args.sclip_unpaired_modality,
            space=args.sclip_space,
            pseudo_label_type=args.sclip_pseudo_label_type,
            weight_unpaired_images=args.sclip_weight_unpaired_images,
            weight_unpaired_texts=args.sclip_weight_unpaired_texts,
            rank=args.rank,
            world_size=args.world_size,
        )
    elif args.ot:
        loss_config = {
            "divergence": args.divergence,
            # Loss weights
            "alpha_marginal": args.alpha_marginal,
            "alpha_supervised_sail": args.alpha_supervised_sail,
            "alpha_supervised_explicit": args.alpha_supervised_explicit,
            "alpha_supervised_implicit": args.alpha_supervised_implicit,
            "alpha_semisupervised_ot": args.alpha_semisupervised_ot,
            "alpha_semisupervised_ot_all": args.alpha_semisupervised_ot_all,
            "alpha_semisupervised_clusters": args.alpha_semisupervised_clusters,
            "alpha_semisupervised_sail": args.alpha_semisupervised_sail,
            "alpha_semisupervised_div": args.alpha_semisupervised_div,
            "alpha_semisupervised_double_softmax": args.alpha_semisupervised_double_softmax,
            "alpha_semisupervised_conditional_kl": args.alpha_semisupervised_conditional_kl,
            "alpha_semisupervised_joint_kl": args.alpha_semisupervised_joint_kl,
            "alpha_unsupervised": args.alpha_unsupervised,
            # Sinkhorn params
            "epsilon_sinkhorn_shared": args.epsilon_sinkhorn_shared,
            "n_iters_sinkhorn_shared": args.n_iters_sinkhorn_shared,
            "epsilon_sinkhorn_anchor": args.epsilon_sinkhorn_anchor,
            "n_iters_sinkhorn_anchor": args.n_iters_sinkhorn_anchor,
            # SAIL
            "temperature_sail": args.temperature_sail,
            "bias_sail": args.bias_sail,
            # softmax kl approaches
            "temperature_softmax": args.temperature_softmax,
            # anchors advanced
            "anchor_center": args.anchor_center,
            "anchor_whiten": args.anchor_whiten,
            "anchor_lam_x": args.anchor_lam_x,
            "anchor_lam_y": args.anchor_lam_y,
            "anchor_rank_k_x": args.anchor_rank_k_x,
            "anchor_rank_k_y": args.anchor_rank_k_y,
            "anchor_relrenorm": args.anchor_relrenorm,
            # unbalanced OT
            "unbalanced": args.unbalanced,
            "tau_x": args.tau_x,
            "tau_y": args.tau_y,
        }
        return FullMatchingModel(loss_config)

    elif args.siglip:
        print("Using SigLip loss")
        return SigLipLoss(
            rank=args.rank,
            world_size=args.world_size,
        )
    else:
        print("Using Clip (infoNCE) loss")
        return ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod,
        )
