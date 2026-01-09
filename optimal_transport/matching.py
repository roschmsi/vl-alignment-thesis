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


class IncrementalCovariance:
    def __init__(self, device, hidden_dim=1024):
        self.device = device
        self.n = 0
        self.sum_x = torch.zeros(1, hidden_dim, device=device)
        self.sum_sq_x = torch.zeros(hidden_dim, hidden_dim, device=device)

    def update(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        self.n += batch.shape[0]
        self.sum_x += batch.sum(dim=0, keepdim=True)
        self.sum_sq_x += torch.matmul(batch.T, batch)

    def compute(self):
        # TODO check numerical stability
        mean = self.sum_x / self.n
        Sxx = (self.sum_sq_x - (self.n * torch.matmul(mean.T, mean))) / max(
            self.n - 1, 1
        )
        return Sxx, mean


class FullMatchingModel(nn.Module):

    def __init__(self, config):
        super(FullMatchingModel, self).__init__()
        # Metrics
        self.divergence = config["divergence"]

        # Loss weights
        self.alpha_marginal = config["alpha_marginal"]
        self.alpha_supervised_sail = config["alpha_supervised_sail"]
        self.alpha_supervised_explicit = config["alpha_supervised_explicit"]
        self.alpha_supervised_implicit = config["alpha_supervised_implicit"]
        self.alpha_semisupervised_ot = config["alpha_semisupervised_ot"]
        self.alpha_semisupervised_monge_gap = config["alpha_semisupervised_monge_gap"]
        self.alpha_semisupervised_ot_all = config["alpha_semisupervised_ot_all"]
        self.alpha_semisupervised_sail = config["alpha_semisupervised_sail"]
        self.alpha_semisupervised_div = config["alpha_semisupervised_div"]
        self.alpha_semisupervised_double_softmax = config[
            "alpha_semisupervised_double_softmax"
        ]
        self.alpha_semisupervised_conditional_kl = config[
            "alpha_semisupervised_conditional_kl"
        ]
        self.alpha_semisupervised_joint_kl = config["alpha_semisupervised_joint_kl"]
        self.alpha_semisupervised_clusters = config["alpha_semisupervised_clusters"]
        self.alpha_unsupervised = config["alpha_unsupervised"]

        # OT parameters
        self.epsilon_sinkhorn_shared = config["epsilon_sinkhorn_shared"]
        self.n_iters_sinkhorn_shared = config["n_iters_sinkhorn_shared"]
        self.epsilon_sinkhorn_anchor = config["epsilon_sinkhorn_anchor"]
        self.n_iters_sinkhorn_anchor = config["n_iters_sinkhorn_anchor"]

        # KL softmax unsupervised loss
        self.temperature_softmax = config.get("temperature_softmax", 0.1)

        # Anchors
        self.register_buffer("anchor_X", None)
        self.register_buffer("anchor_Y", None)
        self.anchor_center = config.get("anchor_center", False)
        self.anchor_whiten = config.get("anchor_whiten", False)
        self.anchor_lam_x = config.get("anchor_lam_x", None)
        self.anchor_lam_y = config.get("anchor_lam_y", None)
        self.anchor_rank_k_x = config.get("anchor_rank_k_x", None)
        self.anchor_rank_k_y = config.get("anchor_rank_k_y", None)

        # Unbalanced OT
        # TODO implement unbalanced OT
        self.unbalanced = config.get("unbalanced", False)
        self.tau_x = config.get("tau_x", 1.0)
        self.tau_y = config.get("tau_y", 1.0)

        # Centering and whitening
        self.register_buffer("x_mean", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("Wxx", None)
        self.register_buffer("Wyy", None)
        self.register_buffer("Sxx", None)
        self.register_buffer("Syy", None)
        self.register_buffer("Sxy_w", None)

    def precompute_anchor_covariances(
        self,
        X_pairs: torch.Tensor,
        Y_pairs: torch.Tensor,
        Sxx_total: torch.Tensor = None,
        Syy_total: torch.Tensor = None,
        mean_x_total: torch.Tensor = None,
        mean_y_total: torch.Tensor = None,
    ):
        """
        Precompute covariance and whitening matrices.
        """
        self.Sxx = Sxx_total
        self.Syy = Syy_total
        self.x_mean = mean_x_total
        self.y_mean = mean_y_total

        X_p = X_pairs
        Y_p = Y_pairs

        X_p = X_p - self.x_mean
        Y_p = Y_p - self.y_mean

        denom_p = max(X_p.size(0) - 1, 1) if self.anchor_center else X_p.size(0)
        self.Sxy = (X_p.T @ Y_p) / denom_p

    def precompute_cca_projections(self, X, Y, lam_x, lam_y):
        """
        Compute CCA projection matrices.
        X: (n_samples, d_x)
        Y: (n_samples, d_y)
        """
        X = X.float()
        Y = Y.float()

        device = X.device

        Sigma_xx = self.Sxx + torch.eye(self.Sxx.size(0), device=device) * lam_x
        Sigma_yy = self.Syy + torch.eye(self.Syy.size(0), device=device) * lam_y

        # compute A^{-1/2} via eigendecomposition
        def compute_inv_sqrt(Sigma, lam):
            # eigenvalues (L) and eigenvectors (V)
            L, V = torch.linalg.eigh(Sigma)
            L = torch.clamp(L, min=lam)
            L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L))
            return V @ L_inv_sqrt @ V.T

        Sxx_inv_sqrt = compute_inv_sqrt(Sigma_xx, lam_x)
        Syy_inv_sqrt = compute_inv_sqrt(Sigma_yy, lam_y)

        T = Sxx_inv_sqrt @ self.Sxy @ Syy_inv_sqrt
        U, S, Vt = torch.linalg.svd(T, full_matrices=False)

        self.CCA_Wx = Sxx_inv_sqrt @ U
        self.CCA_Wy = Syy_inv_sqrt @ Vt.T

    def init_clusters(
        self, X_pairs: torch.Tensor, Y_pairs: torch.Tensor, n_clusters=128, use_cca=True
    ):
        print(f"Initializing Anchors (k={n_clusters}) | CCA: {use_cca} ...")
        device = X_pairs.device
        dtype = X_pairs.dtype

        # 1. Normalize Raw Inputs (Crucial for concatenation)
        X_norm = F.normalize(X_pairs, p=2, dim=1)
        Y_norm = F.normalize(Y_pairs, p=2, dim=1)

        if use_cca:
            # --- CCA PATH: Project and Average ---
            with torch.no_grad():
                # Center (CCA requires centered data)
                X_cent = X_pairs - self.x_mean
                Y_cent = Y_pairs - self.y_mean

                # Project
                X_proj = X_cent @ self.CCA_Wx
                Y_proj = Y_cent @ self.CCA_Wy

                # Normalize post-projection
                X_proj = F.normalize(X_proj, p=2, dim=1)
                Y_proj = F.normalize(Y_proj, p=2, dim=1)

                # Average to find shared concept
                Z_joint = (X_proj + Y_proj) / 2.0
                Z_joint = F.normalize(Z_joint, p=2, dim=1)

            # Cluster
            Z_np = Z_joint.float().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(Z_np)

            # Anchors are the same for both views
            centroids = torch.tensor(
                kmeans.cluster_centers_, device=device, dtype=dtype
            )
            centroids = F.normalize(centroids, p=2, dim=1)

            self.register_buffer("anchor_X", centroids)
            self.register_buffer("anchor_Y", centroids)
            self.use_cca_anchors = True

        else:
            # --- NO CCA PATH: Concatenate and Split ---
            # We cluster the joint occurrence of (x, y)

            # Concatenate features
            # Shape: (N, dx + dy)
            Z_joint = torch.cat([X_norm, Y_norm], dim=1)

            # Cluster
            Z_np = Z_joint.float().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(Z_np)

            centroids_joint = torch.tensor(
                kmeans.cluster_centers_, device=device, dtype=dtype
            )

            # Split centroids back into X and Y components
            dim_x = X_norm.shape[1]
            cx = centroids_joint[:, :dim_x]
            cy = centroids_joint[:, dim_x:]

            # Re-normalize the split centroids so they act as proper directional anchors
            self.register_buffer("anchor_X", F.normalize(cx, p=2, dim=1))
            self.register_buffer("anchor_Y", F.normalize(cy, p=2, dim=1))
            self.use_cca_anchors = False

        print(f"Created {n_clusters} paired anchors.")

    def match_in_anchor_clusters(self, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            eps = self.epsilon_sinkhorn_anchor
            iters = self.n_iters_sinkhorn_anchor

            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            if self.use_cca_anchors:
                X_cent = X - self.x_mean
                Y_cent = Y - self.y_mean
                X_rep = F.normalize(X_cent @ self.CCA_Wx, p=2, dim=1)
                Y_rep = F.normalize(Y_cent @ self.CCA_Wy, p=2, dim=1)
            else:
                X_rep = X
                Y_rep = Y

            Sim_X = torch.mm(X_rep, self.anchor_X.T)
            Sim_Y = torch.mm(Y_rep, self.anchor_Y.T)

            dist = -cosine_similarity_matrix(Sim_X, Sim_Y)  # (nx, ny)

            if not self.use_dustbin or self.outlier_mass <= 0.0:
                res = sinkhorn(dist, epsilon=eps, max_iter=iters)
                return res["plan"], res["log_plan"], None

            plan, log_plan, plan_aug, log_plan_aug, _, _, _ = self._sinkhorn_dustbin(
                dist,
                epsilon=eps,
                max_iter=iters,
                outlier_mass=self.outlier_mass,
                bin_cost=self.bin_cost,
            )
            return plan, log_plan, (plan_aug, log_plan_aug)

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

        dist = -cosine_similarity_matrix(fX, fY)

        res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)

        return res["plan"], res["log_plan"]

    def match_in_anchor_cca(self, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            eps, iters = self.epsilon_sinkhorn_anchor, self.n_iters_sinkhorn_anchor

            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            Xc = (X - self.x_mean) @ self.CCA_Wx
            Yc = (Y - self.y_mean) @ self.CCA_Wy

            Xc = F.normalize(Xc, p=2, dim=1)
            Yc = F.normalize(Yc, p=2, dim=1)

            sim = torch.mm(Xc, Yc.T)
            dist = 1.0 - sim  # (nx, ny)

            res = sinkhorn(dist, epsilon=eps, max_iter=iters)
            return dist, res["plan"], res["log_plan"]

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
        # Note: Ensure cosine_similarity_matrix handles (N, M) shapes efficiently
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

    def loss_supervised_explicit(self, fX_pairs: torch.Tensor, fY_pairs: torch.Tensor):
        """
        Intuition: Explicitly encourages fX = fY for paired samples by minimizing cosine distance.
        Complexity: O(N d)
        Inputs:
            fX: (N, d)
            fY: (N, d)
        """
        norm_fX = torch.norm(fX_pairs, dim=1).clamp(min=1e-8)
        norm_fY = torch.norm(fY_pairs, dim=1).clamp(min=1e-8)
        fX_normalized = fX_pairs / norm_fX.unsqueeze(-1)
        fY_normalized = fY_pairs / norm_fY.unsqueeze(-1)
        cosine_sim = (fX_normalized * fY_normalized).sum(-1)
        loss = (1 - cosine_sim) / 2
        loss = loss.mean()
        return loss

    def loss_supervised_implicit(self, log_plan_pairs: torch.Tensor):
        """
        Intuition: Encourages pairs to be closer than non-paired samples i.e. KL(plan_pairs || 1/N I_N)
        Complexity: complexity of computing the plan O(d N^2)
        Inputs:
            fX: (N, d)
            fY: (N, d)
        """
        N = len(log_plan_pairs)
        log_plan_diag = torch.diag(log_plan_pairs) + log(N)
        loss = -log_plan_diag
        loss = loss.mean()
        return loss

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
        dist_anchor, plan_anchor, log_plan_anchor = self.match_in_anchor_cca(X, Y)
        # Compute KL divergence between plans
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        # TODO normalize by number of unsupervised samples?
        return loss

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

    def loss_semisupervised_clusters(self, X, Y, log_plan):
        """
        KL Divergence between the latent plan and the anchor-relative plan
        """
        plan_anchor, log_plan_anchor = self.match_in_anchor_clusters(X, Y)
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        return loss

    def loss_semisupervised_div(self, X, Y, fX, fY):
        """
        Computes divergence using the 'Trace Trick'.
        Explicitly disables autocast to force execution in float32.
        """

        # 1. Prepare inputs (still can be float16 here, we cast inside the block)
        # We just ensure they are detached/ready if needed, but the math happens below.

        # 2. Critical Section: Disable Autocast
        # "device_type" should match your device (e.g., 'cuda')
        with torch.autocast(device_type=X.device.type, enabled=False):

            # --- Cast Inputs to Float32 ---
            X_f32 = X.float()
            Y_f32 = Y.float()
            fX_f32 = fX.float()
            fY_f32 = fY.float()

            # CCA Matrices (also need to be float32)
            Wx_f32 = self.CCA_Wx.float()
            Wy_f32 = self.CCA_Wy.float()
            mean_x_f32 = self.x_mean.float()
            mean_y_f32 = self.y_mean.float()

            # --- 3. Prepare Anchor Features (CCA Space) ---
            X_f32 = F.normalize(X_f32, p=2, dim=1)
            Y_f32 = F.normalize(Y_f32, p=2, dim=1)

            # Project and Center
            X_cca = (X_f32 - mean_x_f32) @ Wx_f32
            Y_cca = (Y_f32 - mean_y_f32) @ Wy_f32

            # Normalize
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            # --- 4. Prepare Latent Features ---
            # Normalize
            fX_norm = F.normalize(fX_f32, p=2, dim=1)
            fY_norm = F.normalize(fY_f32, p=2, dim=1)

            # --- 5. Apply Trace Trick (Guaranteed FP32) ---

            # Term 1: || A ||_F^2
            Cxx = torch.mm(X_cca.T, X_cca)
            Cyy = torch.mm(Y_cca.T, Y_cca)
            norm1 = (Cxx * Cyy).sum()

            # Term 2: || B ||_F^2
            Dxx = torch.mm(fX_norm.T, fX_norm)
            Dyy = torch.mm(fY_norm.T, fY_norm)
            norm2 = (Dxx * Dyy).sum()

            # Term 3: < A, B >
            Mx = torch.mm(X_cca.T, fX_norm)
            My = torch.mm(Y_cca.T, fY_norm)
            dot = (Mx * My).sum()

            # --- 6. Compute Loss ---
            if self.divergence == "frobenius":
                loss = norm1 + norm2 - 2 * dot
                cste = X.shape[0] * Y.shape[0]
                loss = loss / cste

            elif self.divergence == "cosine":
                denom = (norm1.sqrt() * norm2.sqrt()).clamp(min=1e-8)
                loss = 1 - (dot / denom)
            else:
                raise ValueError(f"Unknown divergence type: {self.divergence}")

        return loss

    def loss_semisupervised_double_softmax(self, X, Y, fX, fY, temperature=0.1):
        """
        Computes KL divergence against a 'Double Softmax' target.
        Target is the geometric mean of the forward and backward probabilities,
        enforcing cycle consistency (mutual agreement) and filtering noise.
        """
        # --- 1. Compute Anchor Targets (CCA Space) ---
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature

            # Double Softmax: Check agreement in both directions
            prob_x2y = F.softmax(logits_anchor, dim=1)
            prob_y2x = F.softmax(logits_anchor, dim=0)

            # Geometric Mean creates a sharp, clean target
            target_matrix = torch.sqrt(prob_x2y * prob_y2x)
            target_matrix = target_matrix / target_matrix.sum().clamp(min=1e-8)

        # --- 2. Compute Latent Prediction ---
        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)

        logits_latent = torch.mm(fX, fY.T) / temperature

        # Align global structure (Joint Distribution) to the target
        log_prob_latent = F.log_softmax(logits_latent.view(-1), dim=0)

        # --- 3. Loss ---
        loss = F.kl_div(log_prob_latent, target_matrix.view(-1), reduction="sum")

        return loss

    def loss_semisupervised_conditional_kl(self, X, Y, fX, fY, temperature=0.1):
        """
        Computes symmetric row-wise KL divergence (Conditional Probability).
        Forces every sample to find a match (good for coverage, risky for noise).
        """
        # 1. Compute Anchor Logits
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)
            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature

        # 2. Compute Latent Logits
        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)
        logits_latent = torch.mm(fX, fY.T) / temperature

        # 3. Symmetric Loss (Row-wise + Col-wise)
        target_xy = F.softmax(logits_anchor, dim=1)
        pred_xy = F.log_softmax(logits_latent, dim=1)
        loss_xy = F.kl_div(pred_xy, target_xy, reduction="batchmean")

        target_yx = F.softmax(logits_anchor.t(), dim=1)
        pred_yx = F.log_softmax(logits_latent.t(), dim=1)
        loss_yx = F.kl_div(pred_yx, target_yx, reduction="batchmean")

        return (loss_xy + loss_yx) / 2

    def loss_semisupervised_joint_kl(self, X, Y, fX, fY, temperature=0.1):
        """
        Computes global KL divergence (Joint Probability).
        Treats the whole matrix as one distribution using a single global Softmax.
        Similar to Double Softmax but without the noise-filtering geometric mean.
        """
        # 1. Compute Anchor Targets (Global Softmax)
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature
            target_prob = F.softmax(logits_anchor.view(-1), dim=0)

        # 2. Compute Latent Prediction
        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)

        logits_latent = torch.mm(fX, fY.T) / temperature
        log_pred_prob = F.log_softmax(logits_latent.view(-1), dim=0)

        # 3. Loss
        loss = F.kl_div(log_pred_prob, target_prob, reduction="sum")

        return loss

    # def loss_semisupervised_div(self, X, Y, fX, fY):
    #     """
    #     Intuition: Encourage distance in latent space to be the same as the distances in "anchor" space
    #     Complexity simplified (Nx=Ny=N, dx=dy=d): O(d^2 + (N d))
    #     Inputs:
    #         X: (Nx, dx) unsupervised samples in left space
    #         Y: (Ny, dy) unsupervised samples in right space
    #         Sxx: (dx, dx) constant to compute similarities in anchor space
    #         Syy: (dy, dy) idem
    #         Sxy: (dx, dy) idem
    #         fX: (Nx, d) encoded unsupervised samples in shared space
    #         fY: (Ny, d) encoded unsupervised samples in shared space

    #     Affinity in the anchor space: K1 = X Sxy Y^T (+- normalization if cosine
    #     Affinity in the shared space: K2 = fX fY^T (+- normalization if cosine)
    #     Divergence between K1 and K2:
    #         - if divergence = 'frobenius': || K1 - K2 ||_F^2
    #         - if divergence = 'cosine': 1 - cosine_similarity_matrix( K1, K2)
    #     """

    #     with torch.no_grad():
    #         norm_X = (X * (X @ self.Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
    #         norm_Y = (Y * (Y @ self.Syy)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
    #         X = X / norm_X
    #         Y = Y / norm_Y
    #     norm_fX = torch.norm(fX, dim=1, keepdim=True).clamp(min=1e-8)
    #     norm_fY = torch.norm(fY, dim=1, keepdim=True).clamp(min=1e-8)
    #     fX = fX / norm_fX
    #     fY = fY / norm_fY

    #     # Now we compute || X Sxy Y^T - fX fY^T ||_F^2 without instantiating the full N x N matrices
    #     # First compute || X Sxy Y^T ||_F^2 = < Sxy, X^T X Sxy Y^T Y >
    #     norm1 = (
    #         torch.linalg.multi_dot([X.T, X, self.Sxy, Y.T, Y]) * self.Sxy
    #     ).sum()  # = torch.linalg.multi_dot([X, Sxy, Y.T]).norm()**2
    #     # Second compute || fX fY^T ||_F^2 = < fX^T fX , fY^T fY >
    #     norm2 = (
    #         torch.mm(fX.T, fX) * torch.mm(fY.T, fY)
    #     ).sum()  # = torch.mm(fX, fY.T).norm()**2
    #     # Last compute < X Sxy Y^T , fX fY^T > = < Sxy , X^T fX fY^T fY >
    #     dot = (
    #         torch.linalg.multi_dot([X.T, fX, fY.T, Y]) * self.Sxy
    #     ).sum()  # = ( torch.mm(fX, fY.T) * torch.linalg.multi_dot([X, Sxy, Y.T]) ).sum()

    #     cste = X.shape[0] * Y.shape[0]  # to normalize the loss w.r.t. number of pairs
    #     if self.divergence == "frobenius":
    #         loss = norm1 + norm2 - 2 * dot
    #         loss = loss / cste
    #     elif self.divergence == "cosine":
    #         loss = 1 - dot / (norm1.sqrt() * norm2.sqrt()).clamp(min=1e-8)
    #     else:
    #         raise ValueError(f"Unknown divergence type: {self.divergence}")
    #     return loss

    def loss_unsupervised(self, X: torch.Tensor, Y: torch.Tensor, plan: torch.Tensor):
        """
        Intuition: plan should preserve pairwise distances in both spaces. (Gromov-Wasserstein)
        Complexity simplified  (Nx=Ny=N, dx=dy=d): ...
        Inputs:
            X: (Nx, dx) unsupervised samples in left space
            Y: (Ny, dy) unsupervised samples in right space
            plan: (Nx, Ny) transport plan between X and Y
            metric: metric within the left and right spaces: 'dot' or 'cosine'
        """
        with torch.no_grad():
            norm_X = torch.norm(X, dim=1, keepdim=True).clamp(min=1e-8)
            X = X / norm_X
            norm_Y = torch.norm(Y, dim=1, keepdim=True).clamp(min=1e-8)
            Y = Y / norm_Y
        return optimized_quad_loss(X, Y, plan)

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
        loss_supervised_explicit = zero
        loss_supervised_implicit = zero
        loss_marginal = zero
        loss_semisupervised_ot = zero
        loss_semisupervised_ot_all = zero
        loss_semisupervised_clusters = zero
        loss_semisupervised_sail = zero
        loss_semisupervised_monge_gap = zero
        loss_semisupervised_div = zero
        loss_unsupervised = zero
        loss_semisupervised_double_softmax = zero
        loss_semisupervised_conditional_kl = zero
        loss_semisupervised_joint_kl = zero

        if self.alpha_supervised_sail > 0:
            loss_supervised_sail = self.loss_supervised_sail(
                fX_pairs=fX_pairs,
                fY_pairs=fY_pairs,
                logit_scale=logit_scale,
                logit_bias=logit_bias,
            )

        if self.alpha_supervised_explicit > 0:
            loss_supervised_explicit = self.loss_supervised_explicit(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs
            )

        if self.alpha_supervised_implicit > 0:
            loss_supervised_implicit = self.loss_supervised_implicit(
                log_plan_pairs=log_plan_pairs
            )

        if self.alpha_marginal > 0:
            loss_marginal = self.loss_marginal(plan) + self.loss_marginal(plan_pairs)

        if self.alpha_semisupervised_ot > 0:
            loss_semisupervised_ot = self.loss_semisupervised_ot(
                X=X, Y=Y, log_plan=log_plan
            )

        if self.alpha_semisupervised_ot_all > 0:
            loss_semisupervised_ot_all = self.loss_semisupervised_ot(
                X=torch.cat([X_pairs, X], dim=0),
                Y=torch.cat([Y_pairs, Y], dim=0),
                log_plan=log_plan_all,
            )

        if self.alpha_semisupervised_monge_gap > 0:
            loss_semisupervised_monge_gap = self.loss_semisupervised_monge_gap(
                X=X, Y=Y, plan=plan
            )

        if self.alpha_semisupervised_clusters > 0:
            loss_semisupervised_clusters = self.loss_semisupervised_clusters(
                X=X, Y=Y, log_plan=log_plan
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

        if self.alpha_semisupervised_div > 0:
            loss_semisupervised_div = self.loss_semisupervised_div(
                X=X, Y=Y, fX=fX, fY=fY
            )

        if self.alpha_semisupervised_double_softmax > 0:
            loss_semisupervised_double_softmax = (
                self.loss_semisupervised_double_softmax(
                    X=X, Y=Y, fX=fX, fY=fY, temperature=self.temperature_softmax
                )
            )

        if self.alpha_semisupervised_conditional_kl > 0:
            loss_semisupervised_conditional_kl = (
                self.loss_semisupervised_conditional_kl(
                    X=X, Y=Y, fX=fX, fY=fY, temperature=self.temperature_softmax
                )
            )

        if self.alpha_semisupervised_joint_kl > 0:
            loss_semisupervised_joint_kl = self.loss_semisupervised_joint_kl(
                X=X, Y=Y, fX=fX, fY=fY, temperature=self.temperature_softmax
            )

        if self.alpha_unsupervised > 0:
            loss_unsupervised = self.loss_unsupervised(X=X, Y=Y, plan=plan)

        loss = (
            self.alpha_marginal * loss_marginal
            + self.alpha_supervised_sail * loss_supervised_sail
            + self.alpha_supervised_explicit * loss_supervised_explicit
            + self.alpha_supervised_implicit * loss_supervised_implicit
            + self.alpha_semisupervised_ot * loss_semisupervised_ot
            + self.alpha_semisupervised_sail * loss_semisupervised_sail
            + self.alpha_semisupervised_ot_all * loss_semisupervised_ot_all
            + self.alpha_semisupervised_monge_gap * loss_semisupervised_monge_gap
            + self.alpha_semisupervised_clusters * loss_semisupervised_clusters
            + self.alpha_semisupervised_div * loss_semisupervised_div
            + self.alpha_semisupervised_double_softmax
            * loss_semisupervised_double_softmax
            + self.alpha_semisupervised_conditional_kl
            * loss_semisupervised_conditional_kl
            + self.alpha_semisupervised_joint_kl * loss_semisupervised_joint_kl
            + self.alpha_unsupervised * loss_unsupervised
        )

        log = {
            "loss_marginal": float(loss_marginal.detach().item()),
            "loss_supervised_sail": float(loss_supervised_sail.detach().item()),
            "loss_supervised_explicit": float(loss_supervised_explicit.detach().item()),
            "loss_supervised_implicit": float(loss_supervised_implicit.detach().item()),
            "loss_semisupervised_ot": float(loss_semisupervised_ot.detach().item()),
            "loss_semisupervised_ot_all": float(
                loss_semisupervised_ot_all.detach().item()
            ),
            "loss_semisupervised_sail": float(loss_semisupervised_sail.detach().item()),
            "loss_semisupervised_monge_gap": float(
                loss_semisupervised_monge_gap.detach().item()
            ),
            "loss_semisupervised_div": float(loss_semisupervised_div.detach().item()),
            "loss_semisupervised_clusters": float(
                loss_semisupervised_clusters.detach().item()
            ),
            "loss_semisupervised_double_softmax": float(
                loss_semisupervised_double_softmax.detach().item()
            ),
            "loss_semisupervised_conditional_kl": float(
                loss_semisupervised_conditional_kl.detach().item()
            ),
            "loss_semisupervised_joint_kl": float(
                loss_semisupervised_joint_kl.detach().item()
            ),
            "loss_unsupervised": float(loss_unsupervised.detach().item()),
            "total_loss": float(loss.detach().item()),
        }
        return loss, log
