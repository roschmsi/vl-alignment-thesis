from model.sclip import SemiSupervisedClipLoss
from model.structure import StructureLoss
from optimal_transport.matching import MatchingModel, OptimizedMatchingModel
from .sail_model import AlignmentLayer, SAILModel, ShareLockAlignmentLayer
from .vision_model import ImageEmbedding
from .language_model import SentenceEmbedding
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
    elif args.structure:
        print("Using Structure loss")
        return StructureLoss(
            temperature=args.structure_temperature,
            normalize_latents=args.structure_normalize_latents,
            warmup_steps=args.structure_warmup_steps,
            structure_lambda=args.structure_lambda,
            structure_levels=args.structure_levels,
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
            "alpha_semisupervised_monge_gap": args.alpha_semisupervised_monge_gap,
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
            # softmax kl approaches
            "temperature_softmax": args.temperature_softmax,
            # anchors advanced
            # "anchor_center": args.anchor_center,
            # "anchor_whiten": args.anchor_whiten,
            "cca_lam_x": args.cca_lam_x,
            "cca_lam_y": args.cca_lam_y,
            "cca_topk_x": args.cca_topk_x,
            "cca_topk_y": args.cca_topk_y,
            "eig_eps": args.eig_eps,
            # matching
            "match_all": args.match_all,
            "tol_sinkhorn": args.tol_sinkhorn,
            # kernel CCA
            "kernel_cca": args.kernel_cca,
            "kcca_kappa": args.kcca_kappa,
            "kcca_sigma": args.kcca_sigma,
            "kcca_top_k": args.kcca_top_k,
            # procrustes
            "procrustes": args.procrustes,
            "local_cca": args.local_cca,
            "sparse_cca": args.sparse_cca,
        }
        if args.optimized_matching:
            print("Using Full Matching Model with optimized OT losses")
            return OptimizedMatchingModel(loss_config)
        else:
            return MatchingModel(loss_config)

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
