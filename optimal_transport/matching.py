import torch.nn as nn
import torch
from optimal_transport.ot_simplified import (
    basic_marginal_loss,
    sinkhorn,
    optimized_quad_loss,
)
from optimal_transport.utils import cosine_similarity_matrix
from math import log
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import torch.nn.functional as F
from math import log as log_fct
import math
import os


class IncrementalCovariance:
    def __init__(
        self,
        device,
        hidden_dim=1024,
        use_fp64: bool = True,
    ):
        self.device = device
        self.n = 0

        dtype = torch.float64 if use_fp64 else torch.float32
        self.running_mean = torch.zeros(1, hidden_dim, device=device, dtype=dtype)
        self.running_scatter_matrix = torch.zeros(
            hidden_dim, hidden_dim, device=device, dtype=dtype
        )

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        batch = batch.to(self.device, non_blocking=True).float()
        batch = F.normalize(batch, p=2, dim=1)
        batch = batch.to(self.running_mean.dtype)

        nb = batch.size(0)
        if nb == 0:
            return

        batch_mean = batch.mean(dim=0, keepdim=True)
        batch_centered = batch - batch_mean
        batch_scatter_matrix = batch_centered.T @ batch_centered

        if self.n == 0:
            self.running_mean.copy_(batch_mean)
            self.running_scatter_matrix.copy_(batch_scatter_matrix)
            self.n = nb
            return

        n_prev = self.n
        n_new = n_prev + nb
        delta = batch_mean - self.running_mean

        self.running_mean += delta * (nb / n_new)
        self.running_scatter_matrix += batch_scatter_matrix + (delta.T @ delta) * (
            n_prev * nb / n_new
        )

        self.n = n_new

    @torch.no_grad()
    def compute(self):
        denominator = max(self.n - 1, 1)
        covariance = (self.running_scatter_matrix / denominator).to(torch.float32)
        mean = self.running_mean.to(torch.float32)
        return covariance, mean


class MatchingModel(nn.Module):

    def __init__(self, config):
        super(MatchingModel, self).__init__()
        # Loss weights
        self.alpha_supervised_sail = config["alpha_supervised_sail"]
        self.alpha_semisupervised_sail = config["alpha_semisupervised_sail"]

        self.alpha_semisupervised_ot = config["alpha_semisupervised_ot"]
        self.alpha_semisupervised_ot_all = config["alpha_semisupervised_ot_all"]
        self.alpha_semisupervised_softmax = config["alpha_semisupervised_softmax"]
        self.alpha_semisupervised_monge_gap = config["alpha_semisupervised_monge_gap"]

        # Sinkhorn parameters
        self.epsilon_sinkhorn_shared = config["epsilon_sinkhorn_shared"]
        self.n_iters_sinkhorn_shared = config["n_iters_sinkhorn_shared"]
        self.epsilon_sinkhorn_anchor = config["epsilon_sinkhorn_anchor"]
        self.n_iters_sinkhorn_anchor = config["n_iters_sinkhorn_anchor"]

        # KL softmax unsupervised loss
        self.temperature_softmax = config.get("temperature_softmax", 0.1)

        self.cca_lam_x = config.get("cca_lam_x", None)
        self.cca_lam_y = config.get("cca_lam_y", None)
        self.cca_topk_x = config.get("cca_topk_x", None)
        self.cca_topk_y = config.get("cca_topk_y", None)
        self.eig_eps = config.get("eig_eps", 1e-6)

    def precompute_procrustes_projections(self, epsilon=1e-3):
        """
        Computes ZCA Whitening matrices + Procrustes Rotation.
        """
        device = self.Sxx.device

        def compute_zca_matrix(Cov, eps):
            Cov_reg = Cov + eps * torch.eye(Cov.shape[0], device=device)

            L, V = torch.linalg.eigh(Cov_reg)
            L = torch.clamp(L, min=1e-6)

            inv_sqrt = torch.diag(1.0 / torch.sqrt(L))
            W_zca = V @ inv_sqrt @ V.T
            return W_zca

        self.W_zca_x = compute_zca_matrix(self.Sxx, eps=self.cca_lam_x)
        self.W_zca_y = compute_zca_matrix(self.Syy, eps=self.cca_lam_y)

        C_white = self.W_zca_x @ self.Sxy @ self.W_zca_y

        # Orthogonal Procrustes Problem
        U, _, Vh = torch.linalg.svd(C_white, full_matrices=False)

        # rotation from X_white to Y_white
        self.R_procrustes = U @ Vh

    def loss_supervised_sail(
        self,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ):
        """
        Sigmoid loss from SAIL paper.
        Intuition: Contrastive learning.
        """
        cosine_latent = cosine_similarity_matrix(fX_pairs, fY_pairs)
        logits = cosine_latent * logit_scale + logit_bias
        target = -1 + 2 * torch.eye(len(fX_pairs), device=fX_pairs.device)
        loss = -torch.mean(torch.nn.functional.logsigmoid(target * logits))
        return loss

    def loss_semisupervised_sail(
        self,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        fX: torch.Tensor,
        fY: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
    ):
        """
        Computes symmetric Sigmoid loss using two rectangular matrices.
        This avoids the expensive O(M^2) computation of the unsupervised-unsupervised block
        while preserving the bidirectional learning signal of the supervised pairs.
        """
        # 1. Prepare the full sets of targets (Supervised + Unsupervised)
        # Check if unsupervised data exists; if not, cat is skipped
        X_all = torch.cat([fX_pairs, fX], dim=0) if fX is not None else fX_pairs
        Y_all = torch.cat([fY_pairs, fY], dim=0) if fY is not None else fY_pairs

        # N = number of supervised pairs (anchors)
        N = fX_pairs.shape[0]

        # image anchors --> all texts
        # Shape: (N_supervised) x (N_supervised + M_unsupervised)
        # This covers: [Sup_X vs Sup_Y] AND [Sup_X vs Unsup_Y]
        loss_img = self.compute_rectangular_loss(
            anchors=fX_pairs,
            targets=Y_all,
            num_positives=N,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        # text anchors --> all images
        # Shape: (N_supervised) x (N_supervised + M_unsupervised)
        # This covers: [Sup_Y vs Sup_X] AND [Sup_Y vs Unsup_X]
        loss_txt = self.compute_rectangular_loss(
            anchors=fY_pairs,
            targets=X_all,
            num_positives=N,
            logit_scale=logit_scale,
            logit_bias=logit_bias,
        )

        # average the two losses
        return (loss_img + loss_txt) / 2

    def compute_rectangular_loss(
        self, anchors, targets, num_positives, logit_scale, logit_bias
    ):
        """
        Helper: Computes Sigmoid loss for a rectangular matrix (N x M).
        Assumes the first 'num_positives' elements on the diagonal are positive pairs.
        """
        # 1. Compute Logits
        logits = cosine_similarity_matrix(anchors, targets) * logit_scale + logit_bias

        # 2. Create Labels (Rectangular)
        # Initialize all as -1 (negatives)
        labels = -torch.ones_like(logits)

        # Set the supervised diagonal to 1 (positives)
        # We assume the first N items in 'targets' correspond to the 'anchors'
        idx = torch.arange(num_positives, device=logits.device)
        labels[idx, idx] = 1.0

        # 3. Compute Sigmoid Loss
        # log_prob = log(sigmoid(labels * logits))
        # This is numerically stable
        log_prob = torch.nn.functional.logsigmoid(labels * logits)

        # 4. Normalize
        # We average over the entire rectangle. This is robust to batch size changes.
        return -log_prob.mean()

    def loss_semisupervised_softmax(self, dist, target_dist):

        epsilon = self.epsilon_sinkhorn_shared  # Could be learnable

        target_softmax_1 = (-target_dist).argmax(dim=1)
        log_softmax_latent_1 = (-dist / epsilon).log_softmax(dim=1)
        loss_semisupervised_softmax_1 = -log_softmax_latent_1[
            torch.arange(len(dist)), target_softmax_1
        ].mean()

        target_softmax_2 = (-target_dist).argmax(dim=0)
        log_softmax_latent_2 = (-dist / epsilon).log_softmax(dim=0)
        loss_semisupervised_softmax_2 = -log_softmax_latent_2[
            target_softmax_2, torch.arange(len(dist))
        ].mean()

        loss_semisupervised_softmax = (
            loss_semisupervised_softmax_1 + loss_semisupervised_softmax_2
        ) / 2
        return loss_semisupervised_softmax

    def loss_semisupervised_ot(self, X, Y, log_plan):
        """
        Intuition: Encourage the plan in the latent space to be the same as the plan in "anchor" space
        Complexity simplified (Nx=Ny=N, dx=dy=d): O( N d (N + d))
        Inputs:
            X: (Nx, dx) unsupervised samples in left space
            Y: (Ny, dy) unsupervised samples in right space
            Sxx: (dx, dx) constant to compute similarities in anchor space
            Syy: (dy, dy) idem
            Sxy: (dx, dy) idem
            log_plan: (Nx, Ny) log of the transport plan in shared space
        """
        if self.kernel_cca:
            dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_kcca(X, Y)
        elif self.procrustes:
            dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_procrustes(
                X, Y
            )
        elif self.local_cca:
            dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_local_cca(
                X, Y
            )
        elif self.sparse_cca:
            dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_sparse_cca(
                X, Y
            )
        else:
            dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_cca(X, Y)

        # Compute KL divergence between plans
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        # TODO normalize by number of unsupervised samples?
        return loss

    def loss_semisupervised_ot_all(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        log_plan_all: torch.Tensor,
        n_pairs: int,
    ):
        """
        Computes KL Divergence on the Full Matrix (Supervised + Unsupervised).
        """
        with torch.no_grad():
            if self.kernel_cca:
                _, plan_anchor, log_plan_anchor = self.match_in_anchor_kcca(X, Y)
            else:
                _, plan_anchor, log_plan_anchor = self.match_in_anchor_cca(X, Y)

        N_total = log_plan_all.shape[0]
        N_u = plan_anchor.shape[0]

        scale_sup = 1.0 / N_total
        scale_unsup = N_u / N_total

        log_scale_sup = log(scale_sup)
        log_scale_unsup = log(scale_unsup)

        log_plan_sup_diag = log_plan_all.diag()[:n_pairs]

        kl_sup = scale_sup * (log_scale_sup - log_plan_sup_diag)
        loss_sup = kl_sup.sum()

        log_plan_unsup = log_plan_all[n_pairs:, n_pairs:]

        target_P = plan_anchor * scale_unsup
        target_log_P = log_plan_anchor + log_scale_unsup

        kl_unsup = target_P * (target_log_P - log_plan_unsup)
        loss_unsup = kl_unsup.sum()

        return loss_sup + loss_unsup

    def loss_semisupervised_monge_gap(self, X, Y, plan):
        """
        Intuition: Encourage the plan in the latent space to be the same as the plan in "anchor" space
        Complexity simplified (Nx=Ny=N, dx=dy=d): O( N d (N + d))
        Inputs:
            X: (Nx, dx) unsupervised samples in left space
            Y: (Ny, dy) unsupervised samples in right space
            Sxx: (dx, dx) constant to compute similarities in anchor space
            Syy: (dy, dy) idem
            Sxy: (dx, dy) idem
            plan: (Nx, Ny) transport plan in shared space
        """
        dist_anchor, plan_anchor, _, _ = self.match_in_anchor_cca(X, Y)
        wasserstein_anchor = (plan_anchor * dist_anchor).sum()
        monge_gap = (
            (plan * dist_anchor).sum() - wasserstein_anchor
        ) / wasserstein_anchor
        return monge_gap

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_pairs: torch.Tensor,
        Y_pairs: torch.Tensor,
        fX: torch.Tensor,
        fY: torch.Tensor,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
        epoch=None,
        batch_idx=None,
        save_dist=False,
    ):
        device = fX.device
        zero = torch.zeros((), device=device)

        need_plan = (
            self.alpha_marginal > 0
            or self.alpha_unsupervised > 0
            or self.alpha_semisupervised_ot > 0
            or self.alpha_semisupervised_clusters > 0
            or self.alpha_semisupervised_monge_gap > 0
        )
        need_plan_pairs = self.alpha_marginal > 0 or self.alpha_supervised_implicit > 0
        need_plan_all = self.alpha_semisupervised_ot_all > 0

        plan = log_plan = None
        if need_plan:
            plan, log_plan = self.match_in_latent(fX, fY)

        plan_pairs = log_plan_pairs = None
        if need_plan_pairs:
            plan_pairs, log_plan_pairs = self.match_in_latent(fX_pairs, fY_pairs)

        plan_all = log_plan_all = None
        if need_plan_all:
            fX_all = torch.cat([fX_pairs, fX], dim=0)
            fY_all = torch.cat([fY_pairs, fY], dim=0)
            plan_all, log_plan_all = self.match_in_latent(fX_all, fY_all)

        # Losses (compute only if weight > 0)
        loss_supervised_sail = zero
        loss_semisupervised_sail = zero
        loss_semisupervised_ot = zero
        loss_semisupervised_ot_all = zero
        loss_semisupervised_monge_gap = zero

        if self.alpha_supervised_sail > 0:
            loss_supervised_sail = self.loss_supervised_sail(
                fX_pairs=fX_pairs,
                fY_pairs=fY_pairs,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )

        if self.alpha_semisupervised_sail > 0:
            loss_semisupervised_sail = self.loss_semisupervised_sail(
                fX_pairs=fX_pairs,
                fY_pairs=fY_pairs,
                fX=fX,
                fY=fY,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )

        if self.alpha_semisupervised_ot > 0:
            loss_semisupervised_ot = self.loss_semisupervised_ot(
                X=X, Y=Y, log_plan=log_plan
            )

        if self.alpha_semisupervised_ot_all > 0:
            loss_semisupervised_ot_all = self.loss_semisupervised_ot_all(
                X=X,
                Y=Y,
                n_pairs=len(X_pairs),
                log_plan_all=log_plan_all,
            )

        if self.alpha_semisupervised_monge_gap > 0:
            loss_semisupervised_monge_gap = self.loss_semisupervised_monge_gap(
                X=X, Y=Y, plan=plan
            )

        loss = (
            +self.alpha_supervised_sail * loss_supervised_sail
            + self.alpha_semisupervised_sail * loss_semisupervised_sail
            + self.alpha_semisupervised_ot * loss_semisupervised_ot
            + self.alpha_semisupervised_ot_all * loss_semisupervised_ot_all
            + self.alpha_semisupervised_monge_gap * loss_semisupervised_monge_gap
        )

        log = {
            "loss_supervised_sail": float(loss_supervised_sail.detach().item()),
            "loss_semisupervised_sail": float(loss_semisupervised_sail.detach().item()),
            "loss_semisupervised_ot": float(loss_semisupervised_ot.detach().item()),
            "loss_semisupervised_ot_all": float(
                loss_semisupervised_ot_all.detach().item()
            ),
            "loss_semisupervised_monge_gap": float(
                loss_semisupervised_monge_gap.detach().item()
            ),
            "total_loss": float(loss.detach().item()),
        }
        return loss, log


class OptimizedMatchingModel(MatchingModel):
    def __init__(self, config):
        super(OptimizedMatchingModel, self).__init__(config)

        self.match_all = config["match_all"]

    def match_in_latent(self, fX: torch.Tensor, fY: torch.Tensor):
        """
        Compute the transport plan between the two sets of encoded points in the latent space.
        Return both the plan and its log. Try to use the most stable computation possible (e.g. log-sum-exp trick when available).

        Inputs:
            fX: (nx, d)
            fY: (ny, d)
            metric: 'dot', 'euclidean' or 'cosine'
            epsilon_sinkhorn: regularization parameter for sinkhorn
            n_iters_sinkhorn: number of iterations for sinkhorn # If 0 this is equivalent to no log_softmax_row_col
        Outputs:
            plan: (nx, ny) transport plan between fX and fY
            log_plan: (nx, ny) log of the transport plan
        """
        epsilon_sinkhorn, n_iters_sinkhorn = (
            self.epsilon_sinkhorn_shared,
            self.n_iters_sinkhorn_shared,
        )

        dist = 1 - cosine_similarity_matrix(fX, fY)
        with torch.no_grad():
            res = sinkhorn(
                dist,
                epsilon=epsilon_sinkhorn,
                max_iter=n_iters_sinkhorn,
            )
            plan = res["plan"]
            log_plan = res["log_plan"]
            sinkhorn_err = res["err"]
        return dist, plan, log_plan, sinkhorn_err

    def match_in_anchor(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute the transport plan between the two sets of points via the anchor space.
        Return both the plan and its log. Try to use the most stable computation possible (e.g. log-sum-exp trick when available).
        Note: this does not require any training, just the pre-computed covariance matrices Sxx, Syy, Sxy.

        Inputs:
            X: (nx, dx)
            Y: (ny, dy)
            Sxx: (dx, dx) constant to compute similarities in anchor space
            Syy: (dy, dy) idem
            Sxy: (dx, dy) idem
        Outputs:
            plan: (nx, ny) transport plan between fX and fY
            log_plan: (nx, ny) log of the transport plan
        """

        epsilon_sinkhorn, n_iters_sinkhorn = (
            self.epsilon_sinkhorn_anchor,
            self.n_iters_sinkhorn_anchor,
        )

        # Compute distances in anchor space
        K = self.affinity(X, Y)
        dist_anchor = 1 - K

        with torch.no_grad():
            # Find OT plan in anchor space
            res = sinkhorn(
                dist_anchor,
                epsilon=epsilon_sinkhorn,
                max_iter=n_iters_sinkhorn,
            )
            plan_anchor = res["plan"]
            log_plan_anchor = res["log_plan"]
            sinkhorn_err = res["err"]
        return dist_anchor, plan_anchor, log_plan_anchor, sinkhorn_err

    def loss_semisupervised_ot(
        self, dist, plan, log_plan, plan_anchor, log_plan_anchor
    ):
        """
        only requires dist to be differentiable.
        """
        loss_semisupervised_correct_grad = (plan_anchor - plan).detach() * dist
        loss_semisupervised_correct_grad = (
            loss_semisupervised_correct_grad.sum() / self.epsilon_sinkhorn_shared
        )

        loss_semisupervised_correct_value = (
            plan_anchor * (log_plan_anchor - log_plan)
        ).sum()

        loss_semisupervised = (
            loss_semisupervised_correct_grad
            + (
                loss_semisupervised_correct_value - loss_semisupervised_correct_grad
            ).detach()
        )

        return loss_semisupervised

    def loss_semisupervised_ot_all(
        self, dist_all, plan_all, log_plan_all, plan_anchor, log_plan_anchor, n_pairs
    ):
        """
        Target plan: [I_n,     0      ]
                     [ 0 , plan_anchor]
        """
        target_plan = torch.zeros_like(plan_all)
        target_plan[:n_pairs, :n_pairs] = torch.eye(
            n_pairs, device=plan_all.device
        ) / len(dist_all)
        target_plan[n_pairs:, n_pairs:] = plan_anchor * len(plan_anchor) / len(dist_all)

        loss_semisupervised_correct_grad = (
            target_plan - plan_all
        ).detach() * dist_all  # + dist_all[:n_pairs, :n_pairs].diag().sum() / len(dist_all)
        loss_semisupervised_correct_grad = (
            loss_semisupervised_correct_grad.sum() / self.epsilon_sinkhorn_shared
        )

        loss_semisupervised_no_grad_11 = (
            (-log_fct(n_pairs) - log_plan_all.diag()[:n_pairs].mean())
            * n_pairs
            / len(dist_all)
        )
        loss_semisupervised_no_grad_22 = (
            plan_anchor * (log_plan_anchor - log_plan_all[n_pairs:, n_pairs:])
        ).sum()
        loss_semisupervised_correct_value = (
            loss_semisupervised_no_grad_11 + loss_semisupervised_no_grad_22
        )

        loss_semisupervised = (
            loss_semisupervised_correct_grad
            + (
                loss_semisupervised_correct_value - loss_semisupervised_correct_grad
            ).detach()
        )

        return loss_semisupervised

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        X_pairs: torch.Tensor,
        Y_pairs: torch.Tensor,
        fX: torch.Tensor,
        fY: torch.Tensor,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        logit_scale: torch.Tensor,
        logit_bias: torch.Tensor,
        epoch: int = 0,
        batch_idx: int = 0,
        save_dist: bool = False,
    ):
        device = fX.device
        zero = torch.zeros((), device=device)

        sinkhorn_err_anchor = sinkhorn_err = zero

        if (
            self.alpha_semisupervised_ot > 0
            or self.alpha_semisupervised_softmax > 0
            or self.alpha_semisupervised_monge_gap > 0
        ):
            dist_anchor, plan_anchor, log_plan_anchor, sinkhorn_err_anchor = (
                self.match_in_anchor(X, Y)
            )
            dist_pairs, plan_pairs, log_plan_pairs, _ = self.match_in_latent(
                fX_pairs, fY_pairs
            )
            dist, plan, log_plan, sinkhorn_err = self.match_in_latent(fX, fY)

            if save_dist and batch_idx == 0:
                output_dir = "/lustre/groups/eml/projects/sroschmann/debugging_dist"
                os.makedirs(output_dir, exist_ok=True)

                file_path = os.path.join(output_dir, f"dists_epoch_{epoch}.pt")

                torch.save(
                    {
                        "dist": dist.detach().cpu(),
                        "dist_anchor": dist_anchor.detach().cpu(),
                    },
                    file_path,
                )

                print(f"Saved debug stats to {file_path}")

        if self.match_all == "crossmodal":
            dist_all, plan_all, log_plan_all, sinkhorn_err = self.match_in_latent(
                torch.cat([fX_pairs, fX], dim=0), torch.cat([fY_pairs, fY], dim=0)
            )
        elif self.match_all == "intramodal":
            dist_all, plan_all, log_plan_all, sinkhorn_err = self.match_in_latent(
                torch.cat([fY_pairs, fX], dim=0), torch.cat([fX_pairs, fY], dim=0)
            )

        loss_supervised_sail = zero
        loss_semisupervised_sail = zero
        loss_semisupervised_ot = zero
        loss_semisupervised_softmax = zero

        if self.alpha_supervised_sail > 0:
            loss_supervised_sail = self.loss_supervised_sail(
                fX_pairs=fX_pairs,
                fY_pairs=fY_pairs,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )

        if self.alpha_semisupervised_sail > 0:
            loss_semisupervised_sail = self.loss_semisupervised_sail(
                fX_pairs=fX_pairs,
                fY_pairs=fY_pairs,
                fX=fX,
                fY=fY,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )

        if self.alpha_semisupervised_ot > 0:
            if self.match_all in ["crossmodal", "intramodal"]:
                n_pairs = len(X_pairs)
                loss_semisupervised_ot = self.loss_semisupervised_ot_all(
                    dist_all,
                    plan_all,
                    log_plan_all,
                    plan_anchor,
                    log_plan_anchor,
                    n_pairs,
                )
            else:
                loss_semisupervised_ot = self.loss_semisupervised_ot(
                    dist, plan, log_plan, plan_anchor, log_plan_anchor
                )

        if self.alpha_semisupervised_softmax > 0:
            if self.match_all in ["crossmodal", "intramodal"]:
                loss_semisupervised_softmax = self.loss_semisupervised_softmax(
                    dist_all, dist_anchor, n_pairs
                )
            else:
                loss_semisupervised_softmax = self.loss_semisupervised_softmax(
                    dist, dist_anchor
                )

        loss = (
            self.alpha_supervised_sail * loss_supervised_sail
            + self.alpha_semisupervised_sail * loss_semisupervised_sail
            + self.alpha_semisupervised_ot * loss_semisupervised_ot
            + self.alpha_semisupervised_softmax * loss_semisupervised_softmax
        )

        log = {
            "loss_supervised_sail": loss_supervised_sail.item(),
            "loss_semisupervised_sail": loss_semisupervised_sail.item(),
            "loss_semisupervised_ot": loss_semisupervised_ot.item(),
            "loss_semisupervised_softmax": loss_semisupervised_softmax.item(),
            "sinkhorn_err_anchor": sinkhorn_err_anchor,
            "sinkhorn_err_shared": sinkhorn_err,
            "total_loss": loss.item(),
        }

        return loss, log
