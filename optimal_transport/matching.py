# Define model
import torch.nn as nn
import torch
from optimal_transport.ot_simplified import (
    basic_marginal_loss,
    sinkhorn,
    optimized_quad_loss,
    sinkhorn_unbalanced,
)
from optimal_transport.utils import cosine_similarity_matrix
from math import log
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest


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
        self.alpha_semisupervised_ot_all = config["alpha_semisupervised_ot_all"]
        self.alpha_semisupervised_sail = config["alpha_semisupervised_sail"]
        self.alpha_semisupervised_div = config["alpha_semisupervised_div"]
        self.alpha_semisupervised_clusters = config["alpha_semisupervised_clusters"]
        self.alpha_unsupervised = config["alpha_unsupervised"]

        # OT parameters
        self.epsilon_sinkhorn_shared = config["epsilon_sinkhorn_shared"]
        self.n_iters_sinkhorn_shared = config["n_iters_sinkhorn_shared"]
        self.epsilon_sinkhorn_anchor = config["epsilon_sinkhorn_anchor"]
        self.n_iters_sinkhorn_anchor = config["n_iters_sinkhorn_anchor"]

        # SigLIP/SAIL parameters
        self.temperature_sail = config["temperature_sail"]
        self.bias_sail = config["bias_sail"]

        # Anchors
        self.register_buffer("anchor_X", None)
        self.register_buffer("anchor_Y", None)
        self.anchor_center = config.get("anchor_center", False)
        self.anchor_whiten = config.get("anchor_whiten", False)
        self.anchor_lam_x = config.get("anchor_lam_x", None)
        self.anchor_lam_y = config.get("anchor_lam_y", None)
        self.anchor_rank_k_x = config.get("anchor_rank_k_x", None)
        self.anchor_rank_k_y = config.get("anchor_rank_k_y", None)
        self.anchor_relrenorm = config.get("anchor_relrenorm", True)

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
        If Sxx_total/Syy_total are provided, they are used for the geometry.
        Otherwise, geometry is estimated from the pairs.
        X_pairs/Y_pairs are always used for cross-correlation matrix (Sxy).
        """
        # compute mean and covariance of all unpaired samples
        if (
            (Sxx_total is not None)
            and (Syy_total is not None)
            and (mean_x_total is not None)
            and (mean_y_total is not None)
        ):
            self.Sxx = Sxx_total
            self.Syy = Syy_total
            self.x_mean = mean_x_total
            self.y_mean = mean_y_total
        else:
            # fallback: compute mean and covariance of pairs
            X_g = X_pairs
            Y_g = Y_pairs

            if self.anchor_center:
                self.x_mean = X_g.mean(dim=0, keepdim=True)
                self.y_mean = Y_g.mean(dim=0, keepdim=True)
                X_g = X_g - self.x_mean
                Y_g = Y_g - self.y_mean

            denom_x = max(X_g.size(0) - 1, 1) if self.anchor_center else X_g.size(0)
            denom_y = max(Y_g.size(0) - 1, 1) if self.anchor_center else Y_g.size(0)
            self.Sxx = (X_g.T @ X_g) / denom_x
            self.Syy = (Y_g.T @ Y_g) / denom_y

        # compute Sxy of pairs
        X_p = X_pairs
        Y_p = Y_pairs

        if self.anchor_center:
            X_p = X_p - self.x_mean
            Y_p = Y_p - self.y_mean

        denom_p = max(X_p.size(0) - 1, 1) if self.anchor_center else X_p.size(0)
        self.Sxy = (X_p.T @ Y_p) / denom_p

        if self.anchor_whiten:
            self.Wxx = self.sym_invsqrt(
                self.Sxx, eps=self.anchor_lam_x, rank_k=self.anchor_rank_k_x
            )
            self.Wyy = self.sym_invsqrt(
                self.Syy,
                eps=self.anchor_lam_y,
                rank_k=self.anchor_rank_k_y,
            )
            self.Sxy_w = self.Wxx @ self.Sxy @ self.Wyy
        else:
            self.Wxx = self.Wyy = self.Sxy_w = None

    def init_cluster_anchors(
        self,
        all_X_pairs: torch.Tensor,
        all_Y_pairs: torch.Tensor,
        n_clusters=256,
        outlier_fraction=0.05,
        min_cluster_size=3,
    ):
        print(f"Initializing Joint Anchors (Max {n_clusters})...")

        X_data = all_X_pairs.detach().float()
        Y_data = all_Y_pairs.detach().float()
        device = all_X_pairs.device
        dtype = all_X_pairs.dtype

        dx = X_data.shape[1]
        dy = Y_data.shape[1]

        if self.anchor_whiten:
            if self.anchor_center:
                X_data = X_data - self.x_mean
                Y_data = Y_data - self.y_mean

            X_norm = X_data @ self.Wxx
            Y_norm = Y_data @ self.Wyy

            X_balanced = X_norm / (dx**0.5)
            Y_balanced = Y_norm / (dy**0.5)

        else:
            # z-score (diagonal whitening)
            mean_x, std_x = X_data.mean(0), X_data.std(0).clamp(min=1e-8)
            mean_y, std_y = Y_data.mean(0), Y_data.std(0).clamp(min=1e-8)

            X_norm = (X_data - mean_x) / std_x
            Y_norm = (Y_data - mean_y) / std_y

        X_balanced = X_norm / (dx**0.5)
        Y_balanced = Y_norm / (dy**0.5)

        # concatenate and remove outliers
        Z_joint = torch.cat([X_balanced, Y_balanced], dim=1)
        Z_np = Z_joint.cpu().numpy()

        iso = IsolationForest(
            contamination=outlier_fraction, n_jobs=-1, random_state=42
        )
        preds = iso.fit_predict(Z_np)
        mask_good = preds == 1

        Z_clean = Z_np[mask_good]
        indices_good = torch.tensor(mask_good, device=device)
        X_raw_clean = all_X_pairs[indices_good]
        Y_raw_clean = all_Y_pairs[indices_good]

        print(f"  -> Dropped {len(Z_np) - len(Z_clean)} outlier pairs.")

        # --- STEP 3: Joint Clustering ---
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(
            Z_clean
        )
        labels = kmeans.labels_

        # --- STEP 4: Aggregate Centroids (Original Space) ---
        X_centroids_list = []
        Y_centroids_list = []
        labels_torch = torch.from_numpy(labels).to(device)

        for k in range(n_clusters):
            indices = labels_torch == k
            if indices.sum().item() >= min_cluster_size:
                X_centroids_list.append(X_raw_clean[indices].mean(dim=0))
                Y_centroids_list.append(Y_raw_clean[indices].mean(dim=0))

        if not X_centroids_list:
            raise ValueError("All clusters pruned!")

        X_centroids = torch.stack(X_centroids_list).to(dtype)
        Y_centroids = torch.stack(Y_centroids_list).to(dtype)

        # --- STEP 5: Register Anchors ---
        self.register_buffer("raw_anchor_X", X_centroids)
        self.register_buffer("raw_anchor_Y", Y_centroids)

        # Immediate Projection for use (Consistent with self.anchor_whiten)
        if self.anchor_whiten:
            Ax = X_centroids
            Ay = Y_centroids
            if self.anchor_center:
                Ax = Ax - self.x_mean
                Ay = Ay - self.y_mean

            Ax_proj = Ax @ self.Wxx
            Ay_proj = Ay @ self.Wyy

            self.anchor_X = Ax_proj / Ax_proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.anchor_Y = Ay_proj / Ay_proj.norm(dim=1, keepdim=True).clamp(min=1e-8)
        else:
            # Fallback Normalization
            self.anchor_X = X_centroids / X_centroids.norm(dim=1, keepdim=True).clamp(
                min=1e-8
            )
            self.anchor_Y = Y_centroids / Y_centroids.norm(dim=1, keepdim=True).clamp(
                min=1e-8
            )

        print(f"  -> Final Joint Anchors: {len(X_centroids)} clusters.")
        return X_centroids, Y_centroids

    def match_in_anchor_relative(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute OT plan based on Similarity Profiles in the projected (whitened) space.
        """
        with torch.no_grad():
            epsilon_sinkhorn = self.epsilon_sinkhorn_anchor
            n_iters_sinkhorn = self.n_iters_sinkhorn_anchor
            eps = 1e-8

            # --- 1. Apply Geometry (Whitening) ---
            # If whitening is enabled, we project the batch into the shared spherical space
            # so it matches the geometry of the projected anchors.
            if self.anchor_whiten and (self.Wxx is not None):
                # Center
                if self.anchor_center:
                    X = X - self.x_mean
                    Y = Y - self.y_mean

                # Project (Whiten)
                # Note: self.Wxx/Wyy were computed using the supervised pairs.
                X = X @ self.Wxx
                Y = Y @ self.Wyy

            # --- 2. Normalization (Required for Cosine Similarity) ---
            # Even in whitened space, vectors have different lengths.
            # We want directionality only.
            X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=eps)
            Y_norm = Y / torch.norm(Y, dim=1, keepdim=True).clamp(min=eps)

            # --- 3. Compute Profiles ---
            # self.anchor_X is ALREADY whitened and normalized in 'precompute_anchor_covariances'
            Sim_X = torch.mm(X_norm, self.anchor_X.T)
            Sim_Y = torch.mm(Y_norm, self.anchor_Y.T)

            # --- 4. Compute Distance & Sinkhorn ---
            dist = -cosine_similarity_matrix(Sim_X, Sim_Y)

            res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)

            return res["plan"], res["log_plan"]

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

    def match_in_anchor(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        with torch.no_grad():
            eps = 1e-8
            epsilon_sinkhorn, n_iters_sinkhorn = (
                self.epsilon_sinkhorn_anchor,
                self.n_iters_sinkhorn_anchor,
            )

            if self.anchor_whiten and self.anchor_center:
                Xc = X - self.x_mean
                Yc = Y - self.y_mean

                Xw = Xc @ self.Wxx
                Yw = Yc @ self.Wyy

                if self.anchor_relrenorm:
                    Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp(min=eps)
                    Yc = Yc / Yc.norm(dim=1, keepdim=True).clamp(min=eps)

                sim = Xw @ self.Sxy_w @ Yw.T
                dist = -sim

            else:
                norm_X = (
                    (X * (X @ self.Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=eps)
                )
                norm_Y = (
                    (Y * (Y @ self.Syy)).sum(-1, keepdim=True).sqrt().clamp(min=eps)
                )
                Xn = X / norm_X
                Yn = Y / norm_Y
                sim = Xn @ self.Sxy @ Yn.T
                dist = -sim

            res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)

            return res["plan"], res["log_plan"]

    def sym_invsqrt(self, S: torch.Tensor, eps: float, rank_k: int | None = None):
        """
        Symmetric PSD inverse square root with Adaptive Tikhonov regularization.
        """
        d = S.size(0)

        # use mean of diagonal (trace/d) to estimate the scale of S
        with torch.no_grad():
            trace_mean = S.diagonal().mean()
            scale = torch.maximum(
                trace_mean, torch.tensor(1e-8, device=S.device, dtype=S.dtype)
            )
            damping = eps * scale

        # S_reg = S + damping * I
        S_reg = S + torch.eye(d, device=S.device, dtype=S.dtype) * damping

        # numerical stability
        orig_dtype = S_reg.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            S_reg = S_reg.float()
            damping = damping.float()

        # eigendecomposition
        evals, evecs = torch.linalg.eigh(S_reg)

        # rank truncation
        if rank_k is not None:
            rank_k = int(min(rank_k, d))
            if 0 < rank_k < d:
                evecs = evecs[:, -rank_k:]
                evals = evals[-rank_k:]

        # inverse square root
        inv_sqrt = torch.clamp(evals, min=damping).rsqrt()

        # reconstruction
        W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T

        return W.to(orig_dtype)

    def loss_marginal(self, plan: torch.Tensor):
        """
        Encourages the marginals of the plan to be uniform.
        """
        n_x, n_y = plan.shape
        a = torch.ones(n_x, device=plan.device) / n_x
        b = torch.ones(n_y, device=plan.device) / n_y
        marg_loss = basic_marginal_loss(plan, a, b)
        return marg_loss

    def loss_supervised_sail(self, fX_pairs: torch.Tensor, fY_pairs: torch.Tensor):
        """
        Sigmoid loss from SAIL paper.
        Intuition: Contrastive learning.
        """
        cosine_latent = cosine_similarity_matrix(fX_pairs, fY_pairs)
        logits = cosine_latent * self.temperature_sail + self.bias_sail
        target = -1 + 2 * torch.eye(len(fX_pairs), device=fX_pairs.device)
        loss = -torch.mean(torch.nn.functional.logsigmoid(target * logits))
        return loss

    # def loss_semisupervised_sail(
    #     self,
    #     fX_pairs: torch.Tensor,
    #     fY_pairs: torch.Tensor,
    #     fX: torch.Tensor,
    #     fY: torch.Tensor,
    # ):
    #     """
    #     Sigmoid loss from SAIL paper, but with additional negatives.
    #     Intuition: Contrastive learning.
    #     """
    #     device = fX_pairs.device
    #     N = len(fX_pairs)

    #     # X_all = torch.cat([fX_pairs, fX], dim=0) if fX is not None else fX_pairs
    #     Y_all = torch.cat([fY_pairs, fY], dim=0) if fY is not None else fY_pairs
    #     cosine_latent = cosine_similarity_matrix(fX_pairs, Y_all)
    #     logits = cosine_latent * self.temperature_sail + self.bias_sail
    #     target = -torch.ones((fX_pairs.size(0), Y_all.size(0)), device=device)
    #     idx = torch.arange(N, device=device)
    #     target[idx, idx] = 1.0
    #     loss = -torch.mean(torch.nn.functional.logsigmoid(target * logits))
    #     return loss

    def loss_semisupervised_sail(
        self,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        fX: torch.Tensor,
        fY: torch.Tensor,
    ):
        """
        Sigmoid loss on the full matrix, masking out the [Unpaired X vs Unpaired Y] quadrant.
        """
        device = fX_pairs.device
        N = len(fX_pairs)

        # Concatenate to get full X and Y
        X_all = torch.cat([fX_pairs, fX], dim=0) if fX is not None else fX_pairs
        Y_all = torch.cat([fY_pairs, fY], dim=0) if fY is not None else fY_pairs

        # Compute similarity matrix
        cosine_latent = cosine_similarity_matrix(X_all, Y_all)
        logits = cosine_latent * self.temperature_sail + self.bias_sail

        # Create target matrix, initialize as -1
        target = -torch.ones_like(logits)

        # Set known positive pairs to 1 (diagonal of the top-left quadrant)
        idx = torch.arange(N, device=device)
        target[idx, idx] = 1.0

        # Mask out bottom-right quadrant
        mask = torch.ones_like(logits)
        if fX is not None and fY is not None:
            mask[N:, N:] = 0.0

        # Compute sigmoid loss
        log_prob = torch.nn.functional.logsigmoid(target * logits)

        # Normalize by number of valid elements
        masked_loss = log_prob * mask
        loss = -masked_loss.sum() / (mask.sum() + 1e-8)

        return loss

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
        plan_anchor, log_plan_anchor = self.match_in_anchor(X, Y)
        # Compute KL divergence between plans
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        return loss

    def loss_semisupervised_clusters(self, X, Y, log_plan):
        """
        KL Divergence between the latent plan and the anchor-relative plan
        """
        plan_anchor, log_plan_anchor = self.match_in_anchor_relative(X, Y)
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        return loss

    def loss_semisupervised_div(self, X, Y, fX, fY):
        """
        Intuition: Encourage distance in latent space to be the same as the distances in "anchor" space
        Complexity simplified (Nx=Ny=N, dx=dy=d): O(d^2 + (N d))
        Inputs:
            X: (Nx, dx) unsupervised samples in left space
            Y: (Ny, dy) unsupervised samples in right space
            Sxx: (dx, dx) constant to compute similarities in anchor space
            Syy: (dy, dy) idem
            Sxy: (dx, dy) idem
            fX: (Nx, d) encoded unsupervised samples in shared space
            fY: (Ny, d) encoded unsupervised samples in shared space

        Affinity in the anchor space: K1 = X Sxy Y^T (+- normalization if cosine
        Affinity in the shared space: K2 = fX fY^T (+- normalization if cosine)
        Divergence between K1 and K2:
            - if divergence = 'frobenius': || K1 - K2 ||_F^2
            - if divergence = 'cosine': 1 - cosine_similarity_matrix( K1, K2)
        """

        with torch.no_grad():
            norm_X = (X * (X @ self.Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            norm_Y = (Y * (Y @ self.Syy)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            X = X / norm_X
            Y = Y / norm_Y
        norm_fX = torch.norm(fX, dim=1, keepdim=True).clamp(min=1e-8)
        norm_fY = torch.norm(fY, dim=1, keepdim=True).clamp(min=1e-8)
        fX = fX / norm_fX
        fY = fY / norm_fY

        # Now we compute || X Sxy Y^T - fX fY^T ||_F^2 without instantiating the full N x N matrices
        # First compute || X Sxy Y^T ||_F^2 = < Sxy, X^T X Sxy Y^T Y >
        norm1 = (
            torch.linalg.multi_dot([X.T, X, self.Sxy, Y.T, Y]) * self.Sxy
        ).sum()  # = torch.linalg.multi_dot([X, Sxy, Y.T]).norm()**2
        # Second compute || fX fY^T ||_F^2 = < fX^T fX , fY^T fY >
        norm2 = (
            torch.mm(fX.T, fX) * torch.mm(fY.T, fY)
        ).sum()  # = torch.mm(fX, fY.T).norm()**2
        # Last compute < X Sxy Y^T , fX fY^T > = < Sxy , X^T fX fY^T fY >
        dot = (
            torch.linalg.multi_dot([X.T, fX, fY.T, Y]) * self.Sxy
        ).sum()  # = ( torch.mm(fX, fY.T) * torch.linalg.multi_dot([X, Sxy, Y.T]) ).sum()

        cste = X.shape[0] * Y.shape[0]  # to normalize the loss w.r.t. number of pairs
        if self.divergence == "frobenius":
            loss = norm1 + norm2 - 2 * dot
            loss = loss / cste
        elif self.divergence == "cosine":
            loss = 1 - dot / (norm1.sqrt() * norm2.sqrt()).clamp(min=1e-8)
        else:
            raise ValueError(f"Unknown divergence type: {self.divergence}")
        return loss

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
    ):
        device = fX.device
        zero = torch.zeros((), device=device)

        need_plan = (
            self.alpha_marginal > 0
            or self.alpha_unsupervised > 0
            or self.alpha_semisupervised_ot > 0
            or self.alpha_semisupervised_clusters > 0
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
        loss_semisupervised_div = zero
        loss_unsupervised = zero

        if self.alpha_supervised_sail > 0:
            loss_supervised_sail = self.loss_supervised_sail(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs
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

        if self.alpha_semisupervised_clusters > 0:
            loss_semisupervised_clusters = self.loss_semisupervised_clusters(
                X=X, Y=Y, log_plan=log_plan
            )

        if self.alpha_semisupervised_sail > 0:
            loss_semisupervised_sail = self.loss_semisupervised_sail(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs, fX=fX, fY=fY
            )

        if self.alpha_semisupervised_div > 0:
            loss_semisupervised_div = self.loss_semisupervised_div(
                X=X, Y=Y, fX=fX, fY=fY
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
            + self.alpha_semisupervised_clusters * loss_semisupervised_clusters
            + self.alpha_unsupervised * loss_unsupervised
            # + self.alpha_semisupervised_div * loss_semisupervised_div  # enable if using
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
            "loss_semisupervised_div": float(loss_semisupervised_div.detach().item()),
            "loss_semisupervised_clusters": float(
                loss_semisupervised_clusters.detach().item()
            ),
            "loss_unsupervised": float(loss_unsupervised.detach().item()),
            "total_loss": float(loss.detach().item()),
        }
        return loss, log
