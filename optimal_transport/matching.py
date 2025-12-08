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
        # First moment (Mean)
        self.sum_x = torch.zeros(1, hidden_dim, device=device)
        # Second moment (Uncentered Covariance)
        self.sum_sq_x = torch.zeros(hidden_dim, hidden_dim, device=device)

    def update(self, batch):
        """
        batch: (B, d) tensor
        """
        # Ensure batch is on correct device/dtype
        batch = batch.to(self.device, non_blocking=True)

        # 1. Update Count
        self.n += batch.shape[0]

        # 2. Update Sum
        self.sum_x += batch.sum(dim=0, keepdim=True)

        # 3. Update Scatter Matrix (Batch matrix multiplication)
        # batch.T @ batch -> (d, B) @ (B, d) -> (d, d)
        self.sum_sq_x += torch.matmul(batch.T, batch)

    def compute(self):
        """Returns the centered Covariance Matrix (Sxx) and Mean"""
        mean = self.sum_x / self.n

        # Formula: Sxx = (SumSq - N * mean * mean^T) / (N - 1)
        term2 = self.n * torch.matmul(mean.T, mean)
        Sxx = (self.sum_sq_x - term2) / max(self.n - 1, 1)

        return Sxx, mean


class FullMatchingModel(nn.Module):

    def __init__(self, config):
        super(FullMatchingModel, self).__init__()

        # Dimensions
        # self.dx = config["dx"]
        # self.dy = config["dy"]
        # self.d = config["d"]

        # Metrics
        self.divergence = config["divergence"]
        # self.temperature = config.get(
        #     "temperature", 1.0
        # )  # Only used if divergence = 'sigmoid'

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

        self.epsilon_sinkhorn_shared = config["epsilon_sinkhorn_shared"]
        self.n_iters_sinkhorn_shared = config["n_iters_sinkhorn_shared"]
        self.epsilon_sinkhorn_anchor = config["epsilon_sinkhorn_anchor"]
        self.n_iters_sinkhorn_anchor = config["n_iters_sinkhorn_anchor"]

        self.temperature_sail = config["temperature_sail"]
        self.bias_sail = config["bias_sail"]

        self.register_buffer("anchor_X", None)
        self.register_buffer("anchor_Y", None)

        # advanced anchor options
        self.anchor_center = config.get("anchor_center", False)
        self.anchor_whiten = config.get("anchor_whiten", False)
        self.anchor_lam_x = config.get("anchor_lam_x", 5e-3)
        self.anchor_lam_y = config.get("anchor_lam_y", 1e-2)
        self.anchor_rank_k_x = config.get("anchor_rank_k_x", None)  # e.g., 512
        self.anchor_rank_k_y = config.get("anchor_rank_k_y", None)  # e.g., 1024
        self.anchor_relrenorm = config.get(
            "anchor_relrenorm", True
        )  # l2 renorm after centering

        # unbalanced ot
        self.unbalanced = config.get("unbalanced", False)
        self.tau_x = config.get("tau_x", 1.0)
        self.tau_y = config.get("tau_y", 1.0)

        # store whitening bits as buffers (None until precomputed)
        self.register_buffer("x_mean", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("Wxx", None)
        self.register_buffer("Wyy", None)
        self.register_buffer("Sxy_w", None)

    # def init_cluster_anchors(
    #     self, all_X_pairs: torch.Tensor, all_Y_pairs: torch.Tensor, n_clusters=256
    # ):
    #     """
    #     Computes global centroids using K-Means to serve as fixed anchors.

    #     Inputs:
    #         all_X_pairs: (Total_N, dx) All available supervised image embeddings
    #         all_Y_pairs: (Total_N, dy) All available supervised text embeddings
    #         n_clusters: Number of anchors to generate (e.g., 128, 256, 512)
    #     """
    #     print(f"Initializing {n_clusters} global anchors via K-Means...")

    #     # 1. Cluster the Vision space (X)
    #     X_np = all_X_pairs.cpu().numpy()
    #     kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(X_np)

    #     # These are our Vision Anchors
    #     X_centroids = torch.tensor(kmeans.cluster_centers_, dtype=all_X_pairs.dtype)

    #     # 2. Map Text to these Clusters
    #     # We cannot cluster Y independently; we need the Y that corresponds to the X-cluster.
    #     Y_centroids_list = []
    #     labels = kmeans.labels_  # array of shape (Total_N,)

    #     for k in range(n_clusters):
    #         # Find indices of all pairs where the image belongs to cluster k
    #         indices = labels == k

    #         if indices.sum() > 0:
    #             # Average the text embeddings of these specific pairs
    #             avg_text = all_Y_pairs[indices].mean(dim=0)
    #             Y_centroids_list.append(avg_text)
    #         else:
    #             # Fallback for empty cluster (rare)
    #             Y_centroids_list.append(all_Y_pairs.mean(dim=0))

    #     Y_centroids = torch.stack(Y_centroids_list)

    #     print("Anchors initialized.")
    #     return X_centroids, Y_centroids

    def init_cluster_anchors(
        self,
        all_X_pairs: torch.Tensor,
        all_Y_pairs: torch.Tensor,
        n_clusters=256,
        outlier_fraction=0.05,
        min_cluster_size=3,
    ):
        """
        Computes global centroids using K-Means.
        1. Removes outliers using Isolation Forest.
        2. Prunes clusters that have fewer than `min_cluster_size` samples.

        Returns:
            X_centroids, Y_centroids (may have fewer rows than n_clusters)
        """
        print(
            f"Initializing anchors (Max {n_clusters}, trimming outliers & tiny clusters)..."
        )

        # Move to CPU for Sklearn
        X_np = all_X_pairs.detach().cpu().numpy()
        device = all_X_pairs.device
        dtype = all_X_pairs.dtype

        # --- STEP 1: Outlier Removal (Isolation Forest) ---
        iso = IsolationForest(
            contamination=outlier_fraction, n_jobs=-1, random_state=42
        )
        preds = iso.fit_predict(X_np)
        mask = preds == 1

        # Filter X (numpy for kmeans) and Y (torch for averaging)
        X_clean_np = X_np[mask]
        Y_clean = all_Y_pairs[mask]

        n_dropped = len(X_np) - len(X_clean_np)
        print(f"  -> Dropped {n_dropped} outlier samples.")

        # --- STEP 2: Cluster the Clean Data ---
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42).fit(
            X_clean_np
        )

        # Raw centers from sklearn (numpy)
        raw_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # --- STEP 3: Prune Small Clusters ---
        X_centroids_list = []
        Y_centroids_list = []

        for k in range(n_clusters):
            # Get boolean mask for this cluster
            indices = labels == k
            count = indices.sum()

            # PRUNING LOGIC: Only keep if size >= threshold
            if count >= min_cluster_size:
                # 1. Vision Anchor: Use the geometric center found by KMeans
                # Convert back to torch/gpu
                cx = torch.tensor(raw_centers[k], device=device, dtype=dtype)
                X_centroids_list.append(cx)

                # 2. Text Anchor: Average the corresponding Y samples
                # Convert boolean numpy mask to torch mask
                torch_mask = torch.from_numpy(indices).to(device)
                cy = Y_clean[torch_mask].mean(dim=0)
                Y_centroids_list.append(cy)
            # else:
            #     pass (implicitly delete/skip this anchor)

        # Stack into tensors
        if len(X_centroids_list) == 0:
            raise ValueError(
                "All clusters were pruned! Try reducing n_clusters or min_cluster_size."
            )

        X_centroids = torch.stack(X_centroids_list)
        Y_centroids = torch.stack(Y_centroids_list)

        # --- STEP 4: Normalize ---
        # Normalize to unit sphere for Cosine Similarity usage
        X_centroids = X_centroids / torch.norm(X_centroids, dim=1, keepdim=True).clamp(
            min=1e-8
        )
        Y_centroids = Y_centroids / torch.norm(Y_centroids, dim=1, keepdim=True).clamp(
            min=1e-8
        )

        print(
            f"  -> Final Anchors: {len(X_centroids)} (Pruned {n_clusters - len(X_centroids)} small clusters)."
        )
        return X_centroids, Y_centroids

    def set_anchors(self, X_anchor: torch.Tensor, Y_anchor: torch.Tensor):
        """
        Register the fixed global anchors.
        Normalizes them immediately so we don't do it every iteration.
        """
        with torch.no_grad():
            # Normalize to unit sphere
            self.anchor_X = X_anchor / torch.norm(X_anchor, dim=1, keepdim=True).clamp(
                min=1e-8
            )
            self.anchor_Y = Y_anchor / torch.norm(Y_anchor, dim=1, keepdim=True).clamp(
                min=1e-8
            )

    def match_in_anchor_relative(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Compute OT plan based on Similarity Profiles (Relative Coordinates).

        Logic:
        1. Compute how similar X is to every anchor -> Profile_X
        2. Compute how similar Y is to every anchor -> Profile_Y
        3. Match Profile_X to Profile_Y
        """
        with torch.no_grad():
            epsilon_sinkhorn = self.epsilon_sinkhorn_anchor
            n_iters_sinkhorn = self.n_iters_sinkhorn_anchor

            # 1. Normalize Unsupervised Batch
            X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-8)
            Y_norm = Y / torch.norm(Y, dim=1, keepdim=True).clamp(min=1e-8)

            # 2. Compute Similarity Profiles (The "Relative Coordinates")
            # Shapes: (N_batch, N_anchors)
            # We use Matrix Multiplication because anchors are already normalized
            Sim_X = torch.mm(X_norm, self.anchor_X.T)
            Sim_Y = torch.mm(Y_norm, self.anchor_Y.T)

            # 3. Compute Distance between Profiles
            # We treat the similarity profiles as the new feature vectors.
            # We use Cosine Distance between these profiles.
            dist = -cosine_similarity_matrix(Sim_X, Sim_Y)

            # NOTE: If your anchors are very dense, you could use Euclidean here instead:
            # dist = torch.cdist(Sim_X, Sim_Y, p=2)

            # 4. Solve OT
            if self.unbalanced:
                M = 1.0 + dist

                # Dynamic Rho Heuristic (safer than fixed parameter)
                # Use 'tau_x' as a multiplier if you want control, or hardcode 5.0
                with torch.no_grad():
                    median_cost = torch.median(M)
                    # If median is tiny, clamp to 0.1 to prevent division by zero
                    # strictly we use tau_x from config as the "multiplier" now
                    rho = max(median_cost.item() * self.tau_x, 0.1)

                res = sinkhorn_unbalanced(
                    M, epsilon=epsilon_sinkhorn, reg_m=rho, max_iter=n_iters_sinkhorn
                )
            else:
                res = sinkhorn(
                    dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn
                )
            plan = res["plan"]
            log_plan = res["log_plan"]

        return plan, log_plan

    # def precompute_anchor_covariances(
    #     self, X_anchor: torch.Tensor, Y_anchor: torch.Tensor
    # ):
    #     """
    #     One unified precompute:
    #     - ALWAYS row-ℓ2 normalizes anchors.
    #     - If anchor_center=True: center (and optionally re-ℓ2) before covariance.
    #     - Returns (Sxx, Syy, Sxy) like before (so your call sites don't change).
    #     - If anchor_whiten=True: also caches Wxx, Wyy, Sxy_w, means as buffers for match_in_anchor().
    #     """
    #     # 1) row ℓ2 normalization (cosine geometry)
    #     X = X_anchor / X_anchor.norm(dim=1, keepdim=True).clamp(min=1e-8)
    #     Y = Y_anchor / Y_anchor.norm(dim=1, keepdim=True).clamp(min=1e-8)

    #     # 2) optional centering (and optional re-ℓ2 to keep cosine-like behavior)
    #     if self.anchor_center:
    #         x_mean = X.mean(dim=0, keepdim=True)
    #         y_mean = Y.mean(dim=0, keepdim=True)
    #         Xc = X - x_mean
    #         Yc = Y - y_mean
    #         if self.anchor_relrenorm:
    #             Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    #             Yc = Yc / Yc.norm(dim=1, keepdim=True).clamp(min=1e-8)
    #     else:
    #         x_mean = torch.zeros(1, X.size(1), device=X.device, dtype=X.dtype)
    #         y_mean = torch.zeros(1, Y.size(1), device=Y.device, dtype=Y.dtype)
    #         Xc, Yc = X, Y

    #     # 3) covariances (use N-1 if centered; otherwise N)
    #     N = Xc.size(0)
    #     denom = max(N - 1, 1) if self.anchor_center else max(N, 1)
    #     Sxx = (Xc.T @ Xc) / denom
    #     Syy = (Yc.T @ Yc) / denom
    #     Sxy = (Xc.T @ Yc) / denom

    #     # 4) optional whitening (+ridge, +rank truncation), cache for match_in_anchor
    #     if self.anchor_whiten:
    #         Wxx = self._sym_invsqrt(
    #             Sxx, eps=self.anchor_lam_x, rank_k=self.anchor_rank_k_x
    #         )
    #         Wyy = self._sym_invsqrt(
    #             Syy, eps=self.anchor_lam_y, rank_k=self.anchor_rank_k_y
    #         )
    #         Sxy_w = Wxx @ Sxy @ Wyy
    #         self.Wxx, self.Wyy, self.Sxy_w = Wxx, Wyy, Sxy_w
    #         self.x_mean, self.y_mean = x_mean.squeeze(0), y_mean.squeeze(0)

    #         self.Wxx = self.Wxx.to("cuda")
    #         self.Wyy = self.Wyy.to("cuda")
    #         self.Sxy_w = self.Sxy_w.to("cuda")
    #         self.x_mean = self.x_mean.to("cuda")
    #         self.y_mean = self.y_mean.to("cuda")
    #     else:
    #         self.Wxx = self.Wyy = self.Sxy_w = None
    #         self.x_mean = self.y_mean = None

    #     return Sxx, Syy, Sxy

    def precompute_anchor_covariances(
        self,
        X_pairs: torch.Tensor,
        Y_pairs: torch.Tensor,
        Sxx_unpaired: torch.Tensor = None,
        Syy_unpaired: torch.Tensor = None,
        mean_x_unpaired: torch.Tensor = None,
        mean_y_unpaired: torch.Tensor = None,
    ):
        """
        Precompute covariance and whitening matrices.

        If Sxx_unpaired/Syy_unpaired are provided, they are used for the geometry.
        Otherwise, geometry is estimated from the pairs.
        X_pairs/Y_pairs are ALWAYS used for the alignment (Sxy).
        """
        use_hybrid = (Sxx_unpaired is not None) and (Syy_unpaired is not None)

        # --- 1. Geometry Estimation (Sxx, Syy, Mean) ---
        if use_hybrid:
            self.Sxx = Sxx_unpaired
            self.Syy = Syy_unpaired

            # Store means if provided (needed for centering new data later)
            if self.anchor_center:
                self.x_mean = (
                    mean_x_unpaired
                    if mean_x_unpaired is not None
                    else X_pairs.mean(0, keepdim=True)
                )
                self.y_mean = (
                    mean_y_unpaired
                    if mean_y_unpaired is not None
                    else Y_pairs.mean(0, keepdim=True)
                )
            else:
                self.x_mean = torch.zeros(1, X_pairs.size(1), device=X_pairs.device)
                self.y_mean = torch.zeros(1, Y_pairs.size(1), device=Y_pairs.device)
        else:
            # Fallback: Compute on Pairs (Original Logic)
            X_g = X_pairs
            Y_g = Y_pairs

            if self.anchor_center:
                self.x_mean = X_g.mean(dim=0, keepdim=True)
                self.y_mean = Y_g.mean(dim=0, keepdim=True)
                X_g = X_g - self.x_mean
                Y_g = Y_g - self.y_mean

            denom_x = max(X_g.size(0) - 1, 1) if self.anchor_center else X_g.size(0)
            denom_y = max(Y_g.size(0) - 1, 1) if self.anchor_center else Y_g.size(0)
            Sxx = (X_g.T @ X_g) / denom_x
            Syy = (Y_g.T @ Y_g) / denom_y

        # --- 2. Alignment Estimation (Sxy) from PAIRS ---
        # We must align the pairs using the geometry defined above (centered by the global mean)
        X_p = X_pairs
        Y_p = Y_pairs

        if self.anchor_center:
            X_p = X_p - self.x_mean
            Y_p = Y_p - self.y_mean
            # Note: If you want relrenorm, apply it here

        denom_p = max(X_p.size(0) - 1, 1) if self.anchor_center else X_p.size(0)
        Sxy = (X_p.T @ Y_p) / denom_p

        # --- 3. Compute Whitening Matrices ---
        if self.anchor_whiten:
            self.Wxx = self._sym_invsqrt(
                Sxx, eps=self.anchor_lam_x, rank_k=self.anchor_rank_k_x
            )
            self.Wyy = self._sym_invsqrt(
                Syy_unpaired if use_hybrid else Syy,
                eps=self.anchor_lam_y,
                rank_k=self.anchor_rank_k_y,
            )

            # Project Sxy into the whitened space
            self.Sxy_w = self.Wxx @ Sxy @ self.Wyy
        else:
            self.Wxx = self.Wyy = self.Sxy_w = None

        return Sxx, Syy, Sxy

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

        if self.unbalanced:
            M = 1.0 + dist

            # Dynamic Rho Heuristic (safer than fixed parameter)
            # Use 'tau_x' as a multiplier if you want control, or hardcode 5.0
            with torch.no_grad():
                median_cost = torch.median(M)
                # If median is tiny, clamp to 0.1 to prevent division by zero
                # strictly we use tau_x from config as the "multiplier" now
                rho = max(median_cost.item() * self.tau_x, 0.1)

            res = sinkhorn_unbalanced(
                M, epsilon=epsilon_sinkhorn, reg_m=rho, max_iter=n_iters_sinkhorn
            )
        else:
            res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)

        plan = res["plan"]
        log_plan = res["log_plan"]
        return plan, log_plan

    # def match_in_anchor(
    #     self,
    #     X: torch.Tensor,
    #     Y: torch.Tensor,
    #     Sxx: torch.Tensor,
    #     Syy: torch.Tensor,
    #     Sxy: torch.Tensor,
    # ):
    #     """
    #     Compute the transport plan between the two sets of points via the anchor space.
    #     Return both the plan and its log. Try to use the most stable computation possible (e.g. log-sum-exp trick when available).
    #     Note: this does not require any training, just the pre-computed covariance matrices Sxx, Syy, Sxy.

    #     Inputs:
    #         X: (nx, dx)
    #         Y: (ny, dy)
    #         Sxx: (dx, dx) constant to compute similarities in anchor space
    #         Syy: (dy, dy) idem
    #         Sxy: (dx, dy) idem
    #     Outputs:
    #         plan: (nx, ny) transport plan between fX and fY
    #         log_plan: (nx, ny) log of the transport plan
    #     """
    #     with torch.no_grad():

    #         epsilon_sinkhorn, n_iters_sinkhorn = (
    #             self.epsilon_sinkhorn_anchor,
    #             self.n_iters_sinkhorn_anchor,
    #         )

    #         # Compute distances in anchor space
    #         norm_X = (X * (X @ Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
    #         norm_Y = (Y * (Y @ Syy)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
    #         X_normalized = X / norm_X
    #         Y_normalized = Y / norm_Y
    #         dist = -torch.linalg.multi_dot([X_normalized, Sxy, Y_normalized.T])

    #         # Find OT plan in anchor space
    #         res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)
    #         plan_anchor = res["plan"]
    #         log_plan_anchor = res["log_plan"]

    #     return plan_anchor, log_plan_anchor

    def match_in_anchor(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Sxx: torch.Tensor,
        Syy: torch.Tensor,
        Sxy: torch.Tensor,
    ):
        with torch.no_grad():
            eps = 1e-8
            epsilon_sinkhorn, n_iters_sinkhorn = (
                self.epsilon_sinkhorn_anchor,
                self.n_iters_sinkhorn_anchor,
            )

            if (
                self.anchor_whiten
                and (self.Wxx is not None)
                and (self.Wyy is not None)
                and (self.Sxy_w is not None)
            ):
                # optional centering consistent with precompute
                Xc = (
                    X - self.x_mean
                    if (self.anchor_center and self.x_mean is not None)
                    else X
                )
                Yc = (
                    Y - self.y_mean
                    if (self.anchor_center and self.y_mean is not None)
                    else Y
                )
                if self.anchor_relrenorm:
                    Xc = Xc / Xc.norm(dim=1, keepdim=True).clamp(min=eps)
                    Yc = Yc / Yc.norm(dim=1, keepdim=True).clamp(min=eps)
                Xw = Xc @ self.Wxx
                Yw = Yc @ self.Wyy
                sim = Xw @ self.Sxy_w @ Yw.T
                dist = -sim
            else:
                # original Mahalanobis-length normalization + Sxy
                norm_X = (X * (X @ Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=eps)
                norm_Y = (Y * (Y @ Syy)).sum(-1, keepdim=True).sqrt().clamp(min=eps)
                Xn = X / norm_X
                Yn = Y / norm_Y
                dist = -torch.linalg.multi_dot([Xn, Sxy, Yn.T])

            # Find OT plan in anchor space
            if self.unbalanced:
                M = 1.0 + dist

                # Dynamic Rho Heuristic (safer than fixed parameter)
                # Use 'tau_x' as a multiplier if you want control, or hardcode 5.0
                with torch.no_grad():
                    median_cost = torch.median(M)
                    # If median is tiny, clamp to 0.1 to prevent division by zero
                    # strictly we use tau_x from config as the "multiplier" now
                    rho = max(median_cost.item() * self.tau_x, 0.1)

                res = sinkhorn_unbalanced(
                    M, epsilon=epsilon_sinkhorn, reg_m=rho, max_iter=n_iters_sinkhorn
                )
            else:
                res = sinkhorn(
                    dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn
                )

            return res["plan"], res["log_plan"]

    def _sym_invsqrt(
        self, S: torch.Tensor, eps: float = 1e-3, rank_k: int | None = None
    ):
        """
        Symmetric PSD inverse square root with Tikhonov regularization and optional rank truncation.
        Returns W such that W @ S @ W ≈ I (pseudo-inverse if truncated).
        """
        d = S.size(0)
        I = torch.eye(d, device=S.device, dtype=S.dtype)
        S_reg = S + eps * I

        # For numerical stability if inputs might be fp16/bf16
        orig_dtype = S_reg.dtype
        if orig_dtype in (torch.float16, torch.bfloat16):
            S_reg = S_reg.float()

        evals, evecs = torch.linalg.eigh(S_reg)  # evals: (d,), evecs: (d,d) (ascending)

        if rank_k is not None:
            rank_k = int(min(rank_k, d))
            if rank_k > 0 and rank_k < d:
                evecs = evecs[:, -rank_k:]  # (d, r)
                evals = evals[-rank_k:]  # (r,)

        inv_sqrt = torch.clamp(evals, min=eps).rsqrt()  # (r,) or (d,)

        # Scale eigenvectors columnwise, then reconstruct
        W = (evecs * inv_sqrt.unsqueeze(0)) @ evecs.T  # (d, d), symmetric

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

    def loss_semisupervised_sail(
        self,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        fX: torch.Tensor,
        fY: torch.Tensor,
    ):
        """
        Sigmoid loss from SAIL paper, but with additional negatives.
        Intuition: Contrastive learning.
        """
        device = fX_pairs.device
        N = len(fX_pairs)

        # X_all = torch.cat([fX_pairs, fX], dim=0) if fX is not None else fX_pairs
        Y_all = torch.cat([fY_pairs, fY], dim=0) if fY is not None else fY_pairs
        cosine_latent = cosine_similarity_matrix(fX_pairs, Y_all)
        logits = cosine_latent * self.temperature_sail + self.bias_sail
        target = -torch.ones((fX_pairs.size(0), Y_all.size(0)), device=device)
        idx = torch.arange(N, device=device)
        target[idx, idx] = 1.0
        loss = -torch.mean(torch.nn.functional.logsigmoid(target * logits))
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

    def loss_semisupervised_ot(self, X, Y, Sxx, Syy, Sxy, log_plan):
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
        plan_anchor, log_plan_anchor = self.match_in_anchor(X, Y, Sxx, Syy, Sxy)
        # Compute KL divergence between plans
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        return loss

    def loss_semisupervised_clusters(self, X, Y, log_plan):
        """
        KL Divergence between the Latent Plan and the Anchor-Relative Plan.
        """
        # Calculate target plan using relative anchors
        plan_anchor, log_plan_anchor = self.match_in_anchor_relative(X, Y)

        # KL(Plan_Anchor || Plan_Latent) = sum( P_anchor * (log P_anchor - log P_latent) )
        kl = plan_anchor * (log_plan_anchor - log_plan)
        loss = kl.sum()
        return loss

    def loss_semisupervised_div(self, X, Y, Sxx, Syy, Sxy, fX, fY):
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
            norm_X = (X * (X @ Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            norm_Y = (Y * (Y @ Syy)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            X = X / norm_X
            Y = Y / norm_Y
        norm_fX = torch.norm(fX, dim=1, keepdim=True).clamp(min=1e-8)
        norm_fY = torch.norm(fY, dim=1, keepdim=True).clamp(min=1e-8)
        fX = fX / norm_fX
        fY = fY / norm_fY

        # Now we compute || X Sxy Y^T - fX fY^T ||_F^2 without instantiating the full N x N matrices
        # First compute || X Sxy Y^T ||_F^2 = < Sxy, X^T X Sxy Y^T Y >
        norm1 = (
            torch.linalg.multi_dot([X.T, X, Sxy, Y.T, Y]) * Sxy
        ).sum()  # = torch.linalg.multi_dot([X, Sxy, Y.T]).norm()**2
        # Second compute || fX fY^T ||_F^2 = < fX^T fX , fY^T fY >
        norm2 = (
            torch.mm(fX.T, fX) * torch.mm(fY.T, fY)
        ).sum()  # = torch.mm(fX, fY.T).norm()**2
        # Last compute < X Sxy Y^T , fX fY^T > = < Sxy , X^T fX fY^T fY >
        dot = (
            torch.linalg.multi_dot([X.T, fX, fY.T, Y]) * Sxy
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
        Sxx: torch.Tensor,
        Syy: torch.Tensor,
        Sxy: torch.Tensor,
    ):
        device = fX.device
        zero = torch.zeros((), device=device)

        # Decide which plans are needed
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

        # Losses (compute only if weighted)
        if self.alpha_supervised_sail > 0:
            loss_supervised_sail = self.loss_supervised_sail(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs
            )
        else:
            loss_supervised_sail = zero

        if self.alpha_supervised_explicit > 0:
            loss_supervised_explicit = self.loss_supervised_explicit(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs
            )
        else:
            loss_supervised_explicit = zero

        if self.alpha_supervised_implicit > 0:
            loss_supervised_implicit = self.loss_supervised_implicit(
                log_plan_pairs=log_plan_pairs
            )
        else:
            loss_supervised_implicit = zero

        if self.alpha_marginal > 0:
            loss_marginal = self.loss_marginal(plan) + self.loss_marginal(plan_pairs)
        else:
            loss_marginal = zero

        if self.alpha_semisupervised_ot > 0:
            loss_semisupervised_ot = self.loss_semisupervised_ot(
                X=X, Y=Y, Sxx=Sxx, Syy=Syy, Sxy=Sxy, log_plan=log_plan
            )
        else:
            loss_semisupervised_ot = zero

        if self.alpha_semisupervised_ot_all > 0:
            loss_semisupervised_ot_all = self.loss_semisupervised_ot(
                X=torch.cat([X_pairs, X], dim=0),
                Y=torch.cat([Y_pairs, Y], dim=0),
                Sxx=Sxx,
                Syy=Syy,
                Sxy=Sxy,
                log_plan=log_plan_all,
            )
        else:
            loss_semisupervised_ot_all = zero

        if self.alpha_semisupervised_clusters > 0:
            loss_semisupervised_clusters = self.loss_semisupervised_clusters(
                X=X, Y=Y, log_plan=log_plan
            )
        else:
            loss_semisupervised_clusters = zero

        if self.alpha_semisupervised_sail > 0:
            loss_semisupervised_sail = self.loss_semisupervised_sail(
                fX_pairs=fX_pairs, fY_pairs=fY_pairs, fX=fX, fY=fY
            )
        else:
            loss_semisupervised_sail = zero

        # DIV: compute graph only if actually used; otherwise skip or compute under no_grad for logging
        if self.alpha_semisupervised_div > 0:
            loss_semisupervised_div = self.loss_semisupervised_div(
                X=X, Y=Y, Sxx=Sxx, Syy=Syy, Sxy=Sxy, fX=fX, fY=fY
            )
        else:
            # If you still want to log it, uncomment the no_grad block:
            # with torch.no_grad():
            #     loss_semisupervised_div = self.loss_semisupervised_div(
            #         X=X, Y=Y, Sxx=Sxx, Syy=Syy, Sxy=Sxy, fX=fX, fY=fY
            #     )
            loss_semisupervised_div = zero

        if self.alpha_unsupervised > 0:
            loss_unsupervised = self.loss_unsupervised(X=X, Y=Y, plan=plan)
        else:
            loss_unsupervised = zero

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
