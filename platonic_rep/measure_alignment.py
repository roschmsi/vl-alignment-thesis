import argparse
import os
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import metrics
import utils


def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing
    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    if isinstance(feats, torch.Tensor):
        feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
        return feats.cuda()
    elif isinstance(feats, list):
        return [
            metrics.remove_outliers(f.float(), q=q, exact=exact).cuda() for f in feats
        ]
    else:
        raise ValueError(f"Unsupported input type for prepare_features: {type(feats)}")


def compute_score(
    x_feats,
    y_feats,
    metric="mutual_knn",
    topk=10,
    kernel="ip",
    rbf_sigma=1.0,
    normalize=True,
):
    """
    Uses different layer combinations of x_feats and y_feats to find the best alignment.

    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
        metric: alignment metric to use (e.g., "mutual_knn")
        topk: number of nearest neighbors if using knn-based metrics
        normalize: whether to normalize the features before comparison

    Returns:
        best_alignment_score: the highest alignment score
        best_alignment_indices: tuple (i, j) of the best x_feat and y_feat layer indices
        all_scores: list of tuples (score, i, j) for each x-y layer pair
    """
    if isinstance(x_feats, torch.Tensor):
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

    if isinstance(y_feats, torch.Tensor):
        y_feats = [y_feats[:, j, :] for j in range(y_feats.shape[1])]

    best_alignment_indices = None
    best_alignment_score = float("-inf")  # in case scores can be negative
    all_scores = []

    for i, x in enumerate(x_feats):
        for j, y in enumerate(y_feats):
            if normalize:
                x_aligned = F.normalize(x, p=2, dim=-1)
                y_aligned = F.normalize(y, p=2, dim=-1)
            else:
                x_aligned = x
                y_aligned = y

            kwargs = {}
            if "knn" in metric:
                kwargs["topk"] = topk
            elif "cka" in metric:
                kwargs["kernel_metric"] = kernel
                kwargs["rbf_sigma"] = rbf_sigma

            score = metrics.AlignmentMetrics.measure(
                metric, x_aligned, y_aligned, **kwargs
            )

            all_scores.append((i, j, score))

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)

    return best_alignment_score, best_alignment_indices, all_scores


def compute_alignment(
    x_feat_paths, y_feat_paths, metric, topk, kernel, rbf_sigma, save_path, precise=True
):
    """
    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """

    os.makedirs(save_path, exist_ok=True)

    symmetric_metric = x_feat_paths == y_feat_paths
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))

    for i, x_fp in enumerate(x_feat_paths):
        x_dict = torch.load(x_fp, map_location="cuda:0")
        x_feats = x_dict["feats"]
        x_feats = prepare_features(x_feats.float(), exact=precise)
        x_feats = [x_feats[:, i, :] for i in range(x_feats.shape[1])]

        if "projected" in x_dict.keys():
            x_projected = x_dict["projected"]
            x_projected = prepare_features(x_projected.float(), exact=precise)
            x_feats.append(x_projected.squeeze())

        for j, y_fp in enumerate(y_feat_paths):
            # if symmetric_metric:
            #     if i > j:
            #         pbar.update(1)
            #         continue

            y_dict = torch.load(y_fp, map_location="cuda:0")
            y_feats = y_dict["feats"]
            y_feats = prepare_features(y_feats.float(), exact=precise)
            y_feats = [y_feats[:, i, :] for i in range(y_feats.shape[1])]

            if "projected" in y_dict.keys():
                y_projected = y_dict["projected"]
                y_projected = prepare_features(y_projected.float(), exact=precise)
                y_feats.append(y_projected.squeeze())

            # turn tensors into list
            best_score, best_indices, all_scores = compute_score(
                x_feats,
                y_feats,
                metric=metric,
                topk=topk,
                kernel=kernel,
                rbf_sigma=rbf_sigma,
                # normalize=False,
            )

            x_fp_parts = x_fp.strip().split("/")
            x_fp_model_name = x_fp_parts[-3]

            y_fp_parts = y_fp.strip().split("/")
            y_fp_model_name = y_fp_parts[-3]

            df = pd.DataFrame(
                all_scores, columns=["model_x_layer", "model_y_layer", "score"]
            )
            df.to_csv(
                f"{save_path}/model_x={x_fp_model_name}_model_y={y_fp_model_name}.csv"
            )
            pbar.update(1)

            torch.cuda.empty_cache()

    return


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="wit_1024")
    parser.add_argument(
        "--modality_x", type=str, default="all", choices=["vision", "language", "all"]
    )
    parser.add_argument("--pool_x", type=str, default=None, choices=["avg", "cls"])
    parser.add_argument(
        "--modality_y", type=str, default="all", choices=["vision", "language", "all"]
    )
    parser.add_argument("--pool_y", type=str, default=None, choices=["avg", "cls"])
    parser.add_argument(
        "--metric",
        type=str,
        default="mutual_knn",
        choices=metrics.AlignmentMetrics.SUPPORTED_METRICS,
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--kernel", type=str, default="ip", choices=["ip", "rbf"])
    parser.add_argument("--rbf_sigma", type=float, default=1.0)

    parser.add_argument("--input_dir", type=str, default="./platonic_results/features")
    parser.add_argument(
        "--output_dir", type=str, default="./platonic_results/alignment"
    )
    parser.add_argument("--precise", action="store_true")
    parser.add_argument("--force_remake", action="store_true")
    parser.add_argument(
        "--language_models",
        nargs="+",
        default=[
            "bigscience/bloomz-7b1",
            "huggyllama/llama-7b",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "google/siglip2-so400m-patch14-224",
            "nvidia/NV-Embed-v2",
        ],
        metavar="LLM",
        help="Space-separated list of LLM model names.",
    )

    parser.add_argument(
        "--vision_models",
        nargs="+",
        default=[
            "facebook/vit-mae-large",
            "facebook/dinov2-large",
            "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
            "google/siglip2-so400m-patch14-224",
        ],
        metavar="LVM",
        help="Space-separated list of vision model names.",
    )

    args = parser.parse_args()

    if not args.precise:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    save_path = utils.to_alignment_filename(
        args.output_dir,
        args.dataset,
        args.modality_x,
        args.pool_x,
        args.modality_y,
        args.pool_y,
        args.metric,
        args.topk,
        args.kernel,
        args.rbf_sigma,
    )

    models_x = (
        args.language_models if args.modality_x == "language" else args.vision_models
    )
    models_y = (
        args.language_models if args.modality_y == "language" else args.vision_models
    )

    models_x_paths = [
        utils.to_feature_filename(
            args.input_dir,
            args.dataset,
            modality=args.modality_x,
            model_name=m,
            pool=args.pool_x,
        )
        for m in models_x
    ]
    models_y_paths = [
        utils.to_feature_filename(
            args.input_dir,
            args.dataset,
            modality=args.modality_y,
            model_name=m,
            pool=args.pool_y,
        )
        for m in models_y
    ]

    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn

    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if "knn" in args.metric:
        print(f"topk:\t{args.topk}")

    print(f"models_x_paths:")
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)

    print("\nmeasuring alignment")
    compute_alignment(
        models_x_paths,
        models_y_paths,
        args.metric,
        topk=args.topk,
        kernel=args.kernel,
        rbf_sigma=args.rbf_sigma,
        save_path=save_path,
        precise=args.precise,
    )
