from transformers import AutoImageProcessor, Dinov2Model
import torch
from PIL import Image
import os
from typing import List, Tuple, Dict, Any, Union, Optional
import torch.nn as nn
from .language_model import SentenceEmbedding
from .vision_model import ImageEmbedding
import torch.nn.functional as F
import torch
from transformers.activations import ACT2FN
from .linear import StarMLP, SiglipMLP, SwiGLU, ShareLockMLP
from torch.cuda.amp import autocast
from functools import partial
import numpy as np
import pdb


def extract_hidden_states_from_batch(features, hidden_states_idx):
    if hidden_states_idx is None:
        n_layers = features.shape[-1]
        third = n_layers // 3
        idx1 = third
        idx2 = 2 * third
        hidden_states_idx = [idx1, idx2, -1]

    features = features[:, :, hidden_states_idx]
    features = features.flatten(start_dim=1)

    return features


class AlignmentLayer(nn.Module):
    def __init__(
        self,
        vision_dimension: int,
        text_dimension: int = None,
        target_dimension: int = 1024,
        linear_type: str = "star",
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = 10.0,
        logit_bias: float = -10.0,
        only_vision: bool = False,
        width_factor: float = 8.0,
        hidden_states=False,
        reconstruction=False,
        reconstruction_type="linear",
    ):
        super(AlignmentLayer, self).__init__()
        assert (
            only_vision or text_dimension is not None
        ), "text_dimension must be provided if only_vision is False"

        self.cast_dtype = cast_dtype
        self.linear_type = linear_type
        if linear_type == "star":
            LinearLayer = partial(
                StarMLP, width_factor=width_factor, activation=nn.ReLU6()
            )
        elif linear_type == "mlp":
            LinearLayer = partial(
                SiglipMLP, intermediate_dim=int(width_factor * target_dimension)
            )
        else:
            LinearLayer = nn.Linear

        self.vision_mapping_network = LinearLayer(vision_dimension, target_dimension)
        self.vision_layer_norm = nn.LayerNorm(vision_dimension)

        if not only_vision:
            self.text_mapping_network = LinearLayer(text_dimension, target_dimension)
            self.text_layer_norm = nn.LayerNorm(text_dimension)

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self._initialize_weights(logit_scale, logit_bias)

        self.hidden_states = hidden_states
        self.reconstruction = reconstruction

        if self.reconstruction:
            self.reconstruction_head = ReconstructionHead(
                vision_dimension=vision_dimension,
                text_dimension=text_dimension,
                target_dimension=target_dimension,
                reconstruction_type=reconstruction_type,
                cast_dtype=cast_dtype,
                width_factor=width_factor,
                hidden_states=hidden_states,
            )

        # if self.second_network:
        #     self.second_network = partial(
        #         StarMLP, width_factor=width_factor, activation=nn.ReLU6()
        #     )

    def _initialize_weights(self, scale: float, bias: float):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(scale))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(bias))

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()

    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(
        self,
        image_features=None,
        text_features=None,
        extra_text_features=None,
        compute_logits=False,
    ):

        # TODO check if there is any difference if features for alignment layer are l2 normalized or not

        if image_features is None and text_features is None:
            raise ValueError(
                "At least one of image_features and text_features should be provided."
            )
        if image_features is not None:
            image_features = image_features.to(self.cast_dtype)
            image_features = self.vision_layer_norm(image_features)
            image_features = self.vision_mapping_network(image_features)
        else:
            image_features = None

        if text_features is not None:
            text_features = text_features.to(self.cast_dtype)
            text_features = self.text_layer_norm(text_features)
            text_features = self.text_mapping_network(text_features)
        else:
            text_features = None

        if extra_text_features is not None:
            extra_text_features_cast = extra_text_features.to(self.cast_dtype)
            if torch.isnan(extra_text_features_cast).any():
                breakpoint()
            extra_text_features_norm = self.text_layer_norm(extra_text_features_cast)
            if torch.isnan(extra_text_features_norm).any():
                breakpoint()
            extra_text_features = self.text_mapping_network(extra_text_features_norm)
        else:
            extra_text_features = None

        if (
            compute_logits
            and image_features is not None
            and text_features is not None
            and image_features.nelement() > 0
            and text_features.nelement() > 0
        ):
            logits_per_text = (
                torch.matmul(
                    F.normalize(text_features, dim=-1),
                    F.normalize(image_features, dim=-1).t(),
                )
                * self.logit_scale.exp()
                + self.logit_bias
            )
        else:
            logits_per_text = None

        return_dict = {
            "image_features": image_features,
            "text_features": text_features,
            "extra_text_features": extra_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.logit_scale.exp(),
            "logit_bias": self.logit_bias,
        }

        if self.reconstruction:
            reconstructed = self.reconstruction_head(
                image_features=image_features,
                text_features=text_features,
            )
            return_dict.update(
                {
                    "reconstructed_image_features": reconstructed[
                        "reconstructed_image_features"
                    ],
                    "reconstructed_text_features": reconstructed[
                        "reconstructed_text_features"
                    ],
                }
            )

        return return_dict


class ShareLockAlignmentLayer(nn.Module):
    def __init__(
        self,
        vision_dimension: int,
        text_dimension: int = None,
        target_dimension: int = None,
        linear_type: str = None,
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = np.log(1 / 0.07),
        logit_bias: float = -10.0,
        *args,
        **kwargs,
    ):
        super(ShareLockAlignmentLayer, self).__init__()
        self.cast_dtype = cast_dtype
        # no vision alignment layer overhead
        target_dimension = vision_dimension
        hidden_dim = 4096
        self.text_mapping_network = ShareLockMLP(
            text_dimension, hidden_dim, target_dimension
        )

        self.text_layer_norm = nn.LayerNorm(text_dimension)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_bias = nn.Parameter(torch.ones([]) * -10.0)
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()

    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(
        self,
        image_features=None,
        text_features=None,
        extra_text_features=None,
        compute_logits=False,
    ):

        if image_features is None and text_features is None:
            raise ValueError(
                "At least one of image_features and text_features should be provided."
            )

        if text_features is not None:
            text_features = text_features.to(self.cast_dtype)
            text_features = self.text_layer_norm(text_features)
            text_features = self.text_mapping_network(text_features)
        else:
            text_features = None

        if (
            compute_logits
            and image_features is not None
            and text_features is not None
            and image_features.nelement() > 0
            and text_features.nelement() > 0
        ):
            logits_per_text = (
                torch.matmul(
                    F.normalize(text_features, dim=-1),
                    F.normalize(image_features, dim=-1).t(),
                )
                * self.logit_scale.exp()
                + self.logit_bias
            )
        else:
            logits_per_text = None

        return {
            "image_features": image_features,
            "text_features": text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.get_logit_scale,
            "logit_bias": self.get_logit_bias,
        }


class ReconstructionHead(nn.Module):
    def __init__(
        self,
        vision_dimension: int,
        text_dimension: int = None,
        target_dimension: int = 1024,
        linear_type: str = "star",
        cast_dtype: Optional[torch.dtype] = None,
        logit_scale: float = 10.0,
        logit_bias: float = -10.0,
        only_vision: bool = False,
        width_factor: int = 8,
        hidden_states=False,
        reconstruction_type="linear",
    ):
        super(ReconstructionHead, self).__init__()
        assert (
            only_vision or text_dimension is not None
        ), "text_dimension must be provided if only_vision is False"

        self.cast_dtype = cast_dtype
        self.reconstruction_type = reconstruction_type
        if reconstruction_type == "star":
            LinearLayer = partial(
                StarMLP, width_factor=width_factor, activation=nn.ReLU6()
            )
        elif reconstruction_type == "mlp":
            LinearLayer = SiglipMLP
        else:
            LinearLayer = nn.Linear

        # TODO add norm before reconstruction

        self.vision_reconstruction_head = LinearLayer(
            target_dimension, vision_dimension
        )

        self.text_reconstruction_head = LinearLayer(
            target_dimension,
            text_dimension,
        )

        self.logit_scale = nn.Parameter(torch.randn(1))
        self.logit_bias = nn.Parameter(torch.randn(1))
        self._initialize_weights(logit_scale, logit_bias)

    def _initialize_weights(self, scale: float, bias: float):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Initialize logit_scale and logit_bias
        logit_scale_init = torch.log(torch.tensor(scale))
        self.logit_scale.data.fill_(logit_scale_init)
        self.logit_bias.data.fill_(torch.tensor(bias))

    @property
    def get_logit_scale(self):
        return self.logit_scale.exp()

    @property
    def get_logit_bias(self):
        return self.logit_bias

    def forward(self, image_features, text_features):
        reconstructed_image_features = self.vision_reconstruction_head(image_features)

        reconstructed_text_features = self.text_reconstruction_head(text_features)

        return {
            "reconstructed_image_features": reconstructed_image_features,
            "reconstructed_text_features": reconstructed_text_features,
        }


class SAILModel(nn.Module):
    def __init__(
        self,
        vision_model_name: str,
        text_model_name: str,
        target_dimension: int,
        vlhead_weights_path: str = None,
        linear_type: str = "linear",
        cast_dtype: Optional[torch.dtype] = None,
        seg: bool = False,
        agg_mode: str = "concat",
        width_factor: int = 8,
        sharelock: bool = False,
        hidden_states=False,
        hidden_states_img_idx=None,
        hidden_states_text_idx=None,
        downsample=False,
    ):
        super(SAILModel, self).__init__()
        self.text_model = SentenceEmbedding(
            text_model_name, output_hidden_states=hidden_states
        )
        self.vision_model = ImageEmbedding(
            vision_model_name,
            seg=seg,
            agg_mode=agg_mode,
            output_hidden_states=hidden_states,
        )

        self.hidden_states = hidden_states
        self.hidden_states_img_idx = hidden_states_img_idx
        self.hidden_states_text_idx = hidden_states_text_idx
        self.downsample = downsample

        if (
            any(
                x in vision_model_name
                for x in ["mae", "ibot", "dinov1", "ml-aim", "ijepa", "clip", "aimv2"]
            )
            or "patch" in agg_mode
            or "cls" in agg_mode
        ):
            if hasattr(self.vision_model.model, "config"):
                vision_dimension = self.vision_model.model.config.hidden_size
            else:
                vision_dimension = self.vision_model.model.embed_dim
        else:
            vision_dimension = self.vision_model.model.config.hidden_size * 2

        LayerClass = ShareLockAlignmentLayer if sharelock else AlignmentLayer

        text_dimension = self.text_model.model.config.hidden_size

        if self.hidden_states:
            if self.hidden_states_img_idx:
                vision_dimension = vision_dimension * len(self.hidden_states_img_idx)
            else:
                vision_dimension = vision_dimension * 3

            if self.hidden_states_text_idx:
                text_dimension = text_dimension * len(self.hidden_states_text_idx)
            else:
                text_dimension = text_dimension * 3

        self.vlhead = LayerClass(
            vision_dimension=vision_dimension,
            text_dimension=text_dimension,
            target_dimension=target_dimension,
            linear_type=linear_type,
            cast_dtype=cast_dtype,
            width_factor=width_factor,
            hidden_states=hidden_states,
        )
        if vlhead_weights_path is not None:
            self.load_vlhead_weights(vlhead_weights_path)

    def freeze_except_vlhead(self):
        self._freeze_parameters(self.vision_model)
        self._freeze_parameters(self.text_model)
        self._unfreeze_parameters(self.vlhead)

    def _freeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def _unfreeze_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = True

    def load_vlhead_weights(self, vlhead_weights_path):
        weights = torch.load(vlhead_weights_path, weights_only=False)
        if "state_dict" in weights:
            weights = weights["state_dict"]
        msg = self.vlhead.load_state_dict(weights, strict=False)
        print(f"Loaded VL head weights from {vlhead_weights_path}, {msg}")

    @property
    def image_processor(self):
        return self.vision_model.image_processor

    @torch.no_grad()
    def encode_image(
        self,
        image,
        normalize: bool = False,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
        patch_mode: bool = False,
        attetion_type: str = "qk",
        ignore_residual: bool = False,
    ):
        if is_pre_encoded:
            features = image
        else:
            features = self.vision_model(
                image,
                patch_mode=patch_mode,
                attetion_type=attetion_type,
                ignore_residual=ignore_residual,
            )

        # TODO ensure that this downsampling is consistent, add if clause
        if self.downsample:
            L = features.shape[-1]
            start = (L - 1) % 3
            features = features[:, :, start::3]

        if self.hidden_states:
            features = extract_hidden_states_from_batch(
                features, self.hidden_states_img_idx
            )

        outputs = self.vlhead(image_features=features)
        image_features = outputs["image_features"]
        if return_encoded:
            return (
                F.normalize(image_features, dim=-1) if normalize else image_features
            ), features
        else:
            return F.normalize(image_features, dim=-1) if normalize else image_features

    @torch.no_grad()
    def encode_text(
        self,
        text,
        text_list: List[str] = None,
        normalize: bool = False,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
    ):

        if self.hidden_states:
            if is_pre_encoded:
                features = text
            else:
                features = self.text_model.get_sentence_embeddings(text_list)

                # TODO ensure that this downsampling is consistent, add if clause
                if self.downsample:
                    L = features.shape[-1]
                    start = (L - 1) % 3
                    features = features[:, :, start::3]

                    features = extract_hidden_states_from_batch(
                        features, self.hidden_states_text_idx
                    )
        else:
            if is_pre_encoded:
                features = text
            else:
                if isinstance(text, str):
                    features = self.text_model.get_sentence_embeddings([text])
                elif (
                    hasattr(self.text_model.model, "config")
                    and "NV" in self.text_model.model.config.name_or_path
                ):
                    features = self.text_model.model.encode(
                        text_list, max_length=1024
                    ).half()
                elif "clip" in self.text_model.model_name:
                    features = self.text_model.get_sentence_embeddings(text_list)
                else:
                    features = self.text_model(text)

        outputs = self.vlhead(text_features=features)
        text_features = outputs["text_features"]
        if return_encoded:
            return (
                F.normalize(text_features, dim=-1) if normalize else text_features
            ), features
        else:
            return F.normalize(text_features, dim=-1) if normalize else text_features

    def forward(
        self,
        images,
        texts,
        text_list: List[str] = None,
        is_pre_encoded: bool = False,
        return_encoded: bool = False,
    ):

        if is_pre_encoded:
            # if its pre-encoded, we do not need to return encoded features
            norm_image_features = self.encode_image(
                images,
                normalize=True,
                is_pre_encoded=True,
                return_encoded=False,
            )
            norm_text_features = self.encode_text(
                texts,
                normalize=True,
                is_pre_encoded=True,
                return_encoded=False,
            )
        elif return_encoded:
            # if we need to return encoded features, then it is not pre-encoded
            norm_image_features, encoded_image_features = self.encode_image(
                images,
                normalize=True,
                is_pre_encoded=False,
                return_encoded=True,
            )
            norm_text_features, encoded_text_features = self.encode_text(
                texts,
                text_list=text_list,
                normalize=True,
                is_pre_encoded=False,
                return_encoded=True,
            )
        else:
            # if we do not need to return encoded features and it is not pre-encoded, usuallu during training
            norm_image_features = self.encode_image(images, normalize=True)
            norm_text_features = self.encode_text(
                texts, text_list=text_list, normalize=True
            )

        # TODO downsampling to align with encoding setup
        # breakpoint()
        # L = batch_embeddings.shape[-1]
        # start = (L - 1) % 3
        # batch_embeddings = batch_embeddings[:, :, start::3]

        # Log the sizes of embeddings
        logits_per_text = (
            torch.matmul(norm_text_features, norm_image_features.t())
            * self.vlhead.logit_scale.exp()
            + self.vlhead.logit_bias
        )

        return_dict = {
            "image_features": norm_image_features,
            "text_features": norm_text_features,
            "logits_per_text": logits_per_text,
            "logit_scale": self.vlhead.get_logit_scale,
            "logit_bias": self.vlhead.get_logit_bias,
        }

        if return_encoded:
            return_dict.update(
                {
                    "encoded_image_features": encoded_image_features,
                    "encoded_text_features": encoded_text_features,
                }
            )

        return return_dict


if __name__ == "__main__":
    import torch
    from fvcore.nn import FlopCountAnalysis, parameter_count
    from thop import profile

    # 初始化模型
    model = AlignmentLayer(
        vision_dimension=2048,
        text_dimension=1024,
        target_dimension=1024,
        linear_type="raw",
    )

    print(model)
    # 假设输入图像和文本特征F
    image_features = torch.randn(1, 2048)
    text_features = torch.randn(1, 1024)

    # output = model(image_features=image_features, text_features=text_features)

    flops, params = profile(model, inputs=(image_features, text_features))

    print(f"FLOPs: {flops/1e6}")
    print(f"Params: {params/1e6}")
