# https://github.com/alinlab/s-clip/blob/master/custom/loss.py#L106

import torch
import torch.nn.functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

try:
    import wandb
except ImportError:
    wandb = None

from open_clip.loss import ClipLoss
import ot


class SemiSupervisedClipLoss(ClipLoss):
    def __init__(
        self,
        method,
        unpaired_modality,  # image, text, both
        space,  # unimodal, bimodal
        pseudo_label_type,  # soft, hard, ot
        epsilon=0.01,
        weight_unpaired_images=0.0,
        weight_unpaired_texts=0.0,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod,
        )

        assert method in ["base", "pseudo-labels"]
        self.method = method
        self.pseudo_label_type = pseudo_label_type
        self.space = space
        self.unpaired_modality = unpaired_modality
        self.weight_unpaired_images = weight_unpaired_images
        self.weight_unpaired_texts = weight_unpaired_texts

    def forward(
        self,
        texts_paired,
        images_paired,
        f_texts_paired,
        f_images_paired,
        texts_unpaired,
        images_unpaired,
        f_texts_unpaired,
        f_images_unpaired,
        logit_scale,
        output_dict=True,
    ):
        device = f_images_paired.device
        losses = dict()

        # gather tensors over worlds
        if self.world_size > 1:
            dist_kwargs = {
                "local_loss": self.local_loss,
                "gather_with_grad": self.gather_with_grad,
                "rank": self.rank,
                "world_size": self.world_size,
                "use_horovod": self.use_horovod,
            }
            images_paired = gather_features(images_paired, **dist_kwargs)
            texts_paired = gather_features(texts_paired, **dist_kwargs)
            images_unpaired = gather_features(images_unpaired, **dist_kwargs)
            texts_unpaired = gather_features(texts_unpaired, **dist_kwargs)
            f_images_paired = gather_features(f_images_paired, **dist_kwargs)
            f_texts_paired = gather_features(f_texts_paired, **dist_kwargs)
            f_images_unpaired = gather_features(f_images_unpaired, **dist_kwargs)
            f_texts_unpaired = gather_features(f_texts_unpaired, **dist_kwargs)

        # normalize
        images_paired = F.normalize(images_paired, dim=-1)
        texts_paired = F.normalize(texts_paired, dim=-1)
        images_unpaired = F.normalize(images_unpaired, dim=-1)
        texts_unpaired = F.normalize(texts_unpaired, dim=-1)
        f_images_paired = F.normalize(f_images_paired, dim=-1)
        f_texts_paired = F.normalize(f_texts_paired, dim=-1)
        f_images_unpaired = F.normalize(f_images_unpaired, dim=-1)
        f_texts_unpaired = F.normalize(f_texts_unpaired, dim=-1)

        # compute info nce loss
        if self.method == "base":
            logits_per_paired_image = logit_scale * f_images_paired @ f_texts_paired.T
            logits_per_paired_text = logits_per_paired_image.T

            labels = self.get_ground_truth(device, f_images_paired.shape[0])
            losses["contrastive_loss"] = (
                F.cross_entropy(logits_per_paired_image, labels)
                + F.cross_entropy(logits_per_paired_text, labels)
            ) / 2

        else:

            texts_candidates = f_texts_paired
            if (
                self.unpaired_modality in ["text", "both"]
                and f_texts_unpaired is not None
                and f_texts_unpaired.numel() > 0
            ):
                texts_candidates = torch.cat([f_texts_paired, f_texts_unpaired], dim=0)

            images_candidates = f_images_paired
            if (
                self.unpaired_modality in ["image", "both"]
                and f_images_unpaired is not None
                and f_images_unpaired.numel() > 0
            ):
                images_candidates = torch.cat(
                    [f_images_paired, f_images_unpaired], dim=0
                )

            logits_img = logit_scale * (f_images_paired @ texts_candidates.T)
            logits_txt = logit_scale * (f_texts_paired @ images_candidates.T)

            labels = self.get_ground_truth(device, f_images_paired.shape[0])
            losses["info_nce"] = (
                F.cross_entropy(logits_img, labels)
                + F.cross_entropy(logits_txt, labels)
            ) / 2

            total_loss = losses["info_nce"]

            # caption-level loss for unpaired images
            if self.unpaired_modality in ["image", "both"]:
                logits_per_unpaired_image = logit_scale * (
                    f_images_unpaired @ f_texts_paired.T
                )

                if self.space == "unimodal":
                    query = images_unpaired
                    labeled_data = images_paired
                elif self.space == "bimodal":
                    query = f_images_unpaired
                    labeled_data = f_images_paired
                else:
                    raise NotImplementedError

                plan = get_assignments(
                    query,
                    labeled_data,
                    logit_scale,
                    self.pseudo_label_type,
                )
                pseudo_labels = plan @ F.one_hot(labels).float()

                losses["caption_loss_unpaired_images"] = soft_cross_entropy(
                    logits_per_unpaired_image, pseudo_labels
                )

                total_loss += (
                    self.weight_unpaired_images * losses["caption_loss_unpaired_images"]
                )

            # caption-level loss for unpaired texts
            if self.unpaired_modality in ["text", "both"]:
                logits_per_unpaired_text = logit_scale * (
                    f_texts_unpaired @ f_images_paired.T
                )

                if self.space == "unimodal":
                    query = texts_unpaired
                    labeled_data = texts_paired
                elif self.space == "bimodal":
                    query = f_texts_unpaired
                    labeled_data = f_texts_paired
                else:
                    raise NotImplementedError

                plan = get_assignments(
                    query,
                    labeled_data,
                    logit_scale,
                    self.pseudo_label_type,
                )
                pseudo_labels = plan @ F.one_hot(labels).float()

                losses["caption_loss_unpaired_texts"] = soft_cross_entropy(
                    logits_per_unpaired_text, pseudo_labels
                )

                total_loss += (
                    self.weight_unpaired_texts * losses["caption_loss_unpaired_texts"]
                )

            losses["total_loss"] = total_loss

        return losses if output_dict else sum(losses.items())

    # def forward(
    #     self,
    #     image_features,
    #     text_features,
    #     logit_scale,
    #     output_dict=False,
    #     query_features=None,
    #     keyword_features=None,
    #     keyword_labels=None,
    # ):
    #     device = image_features.device
    #     losses = dict()  # dict of losses

    #     # gather tensors over worlds
    #     if self.world_size > 1:
    #         dist_kwargs = {
    #             "local_loss": self.local_loss,
    #             "gather_with_grad": self.gather_with_grad,
    #             "rank": self.rank,
    #             "world_size": self.world_size,
    #             "use_horovod": self.use_horovod,
    #         }
    #         image_features = gather_features(image_features, **dist_kwargs)
    #         text_features = gather_features(text_features, **dist_kwargs)
    #         query_features = gather_features(query_features, **dist_kwargs)
    #         keyword_labels = gather_features(keyword_labels, **dist_kwargs)

    #     # normalize
    #     image_features = F.normalize(image_features, dim=-1)
    #     text_features = F.normalize(text_features, dim=-1)
    #     if query_features is not None:
    #         query_features = F.normalize(query_features, dim=-1)

    #     # compute loss
    #     if self.method == "base":
    #         logits_per_image = logit_scale * image_features @ text_features.T
    #         logits_per_text = logits_per_image.T

    #         labels = self.get_ground_truth(device, image_features.shape[0])
    #         losses["contrastive_loss"] = (
    #             F.cross_entropy(logits_per_image, labels)
    #             + F.cross_entropy(logits_per_text, labels)
    #         ) / 2

    #     else:
    #         logits_per_image = logit_scale * image_features @ text_features.T
    #         logits_per_query = logit_scale * query_features @ text_features.T
    #         logits_per_text = torch.cat([logits_per_image, logits_per_query]).T

    #         # supervised CLIP loss
    #         labels = self.get_ground_truth(device, image_features.shape[0])
    #         losses["contrastive_loss"] = (
    #             F.cross_entropy(logits_per_image, labels)
    #             + F.cross_entropy(logits_per_text, labels)
    #         ) / 2

    #         # caption-level loss
    #         plan = get_assignments(
    #             query_features,
    #             image_features,
    #             text_features,
    #             logit_scale,
    #             self.pseudo_label_type,
    #         )
    #         pseudo_labels = plan @ F.one_hot(labels).float()

    #         losses["caption_loss"] = (
    #             soft_cross_entropy(logits_per_query, pseudo_labels)
    #         ) / 2

    # # keyword-level loss
    # selected = []
    # pseudo_labels_keyword = torch.zeros(
    #     len(query_features), len(keyword_features), device=device
    # )
    # for query_id, q in enumerate(query_features):
    #     sample_id = int(plan[query_id].max(dim=0)[1])  # nearest one
    #     candidates = (
    #         keyword_labels[sample_id, :, 0].nonzero().flatten().tolist()
    #     )

    #     if len(candidates) > 0:
    #         selected.append(query_id)
    #         if len(candidates) == 1:
    #             pseudo_labels_keyword[query_id, candidates[0]] = 1
    #         else:
    #             k = torch.stack([keyword_features[i] for i in candidates])
    #             sim = (q @ k.T * logit_scale).detach()
    #             prob = sim / sim.sum()
    #             for i in range(len(sim)):
    #                 pseudo_labels_keyword[query_id, candidates[i]] = prob[i]

    # logits_per_query_keyword = logit_scale * query_features @ keyword_features.T
    # losses["keyword_loss"] = (
    #     soft_cross_entropy(logits_per_query_keyword, pseudo_labels_keyword)
    # ) / 2

    #     losses["total_loss"] = sum(losses.values())

    # return losses if output_dict else sum(losses.items())


# TODO discuss bug in their assignment function
# def get_assignments(query, image, text, logit_scale, pseudo_label_type):
#     if pseudo_label_type == "hard-image":
#         plan = hard_nn(query, image)
#     elif pseudo_label_type == "hard-text":
#         plan = hard_nn(query, image)
#     elif pseudo_label_type == "soft-image":
#         plan = soft_nn(query, image, logit_scale)
#     elif pseudo_label_type == "soft-text":
#         plan = soft_nn(query, text, logit_scale)
#     elif pseudo_label_type == "ot-image":
#         plan = ot_plan(query, image, logit_scale)
#     elif pseudo_label_type == "ot-text":
#         plan = ot_plan(query, image, logit_scale)
#     else:
#         raise NotImplementedError
#     return plan


def get_assignments(query, labeled_data, logit_scale, pseudo_label_type):
    if pseudo_label_type == "hard":
        plan = hard_nn(query, labeled_data)
    elif pseudo_label_type == "soft":
        plan = soft_nn(query, labeled_data, logit_scale)
    elif pseudo_label_type == "ot":
        plan = ot_plan(query, labeled_data, logit_scale)
    else:
        raise NotImplementedError
    return plan


def hard_nn(query, support):
    _, idx = (query @ support.T).max(dim=1)
    plan = F.one_hot(idx, len(support)).float()
    return plan


def soft_nn(query, support, logit_scale):
    plan = F.softmax(query @ support.T * logit_scale, dim=1)
    return plan


def ot_plan(query, support, logit_scale):
    C = 1 - query @ support.T  # (query, batch)
    reg = 1 / logit_scale  # learned temperature

    dim_p, dim_q = C.shape
    p = torch.ones(dim_p, device=C.device, dtype=torch.double) / dim_p
    q = torch.ones(dim_q, device=C.device, dtype=torch.double) / dim_q
    P = ot.bregman.sinkhorn(p, q, C, reg=reg, numItermax=10)

    plan = P / P.sum(dim=1, keepdim=True)
    plan = plan.type_as(support)
    return plan


def soft_cross_entropy(outputs, targets, weight=1.0):
    loss = -targets * F.log_softmax(outputs, dim=1)
    return (loss * weight).sum(dim=1).mean()


def gather_features(
    features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
):
    if features is None:
        return features

    assert (
        has_distributed
    ), "torch.distributed did not import correctly, please use a PyTorch version with support."
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_features = hvd.allgather(features)
        else:
            with torch.no_grad():
                all_features = hvd.allgather(features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features = list(all_features.chunk(world_size, dim=0))
                gathered_features[rank] = features
                all_features = torch.cat(gathered_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
        else:
            gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
            dist.all_gather(gathered_features, features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_features[rank] = features
            all_features = torch.cat(gathered_features, dim=0)

    return all_features
