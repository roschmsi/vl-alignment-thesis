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
        self.mean = torch.zeros(1, hidden_dim, device=device, dtype=dtype)
        self.M2 = torch.zeros(hidden_dim, hidden_dim, device=device, dtype=dtype)

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        batch = batch.to(self.device, non_blocking=True).float()
        batch = F.normalize(batch, p=2, dim=1)

        batch = batch.to(self.mean.dtype)
        nb = batch.size(0)
        if nb == 0:
            return

        mb = batch.mean(dim=0, keepdim=True)
        Xc = batch - mb
        M2b = Xc.T @ Xc

        if self.n == 0:
            self.mean.copy_(mb)
            self.M2.copy_(M2b)
            self.n = nb
            return

        n = self.n
        nt = n + nb
        delta = mb - self.mean

        self.mean += delta * (nb / nt)
        self.M2 += M2b + (delta.T @ delta) * (n * nb / nt)
        self.n = nt

    @torch.no_grad()
    def compute(self):
        denom = max(self.n - 1, 1)
        cov = (self.M2 / denom).to(torch.float32)
        mean = self.mean.to(torch.float32)
        return cov, mean


class MatchingModel(nn.Module):

    def __init__(self, config):
        super(MatchingModel, self).__init__()
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
        # self.anchor_center = config.get("anchor_center", False)
        # self.anchor_whiten = config.get("anchor_whiten", False)
        self.cca_lam_x = config.get("cca_lam_x", None)
        self.cca_lam_y = config.get("cca_lam_y", None)
        self.cca_topk_x = config.get("cca_topk_x", None)
        self.cca_topk_y = config.get("cca_topk_y", None)
        self.eig_eps = config.get("eig_eps", 1e-6)

        # Unbalanced OT
        # TODO implement unbalanced OT
        self.unbalanced = config.get("unbalanced", False)
        self.tau_x = config.get("tau_x", 1.0)
        self.tau_y = config.get("tau_y", 1.0)

        self.kernel_cca = config.get("kernel_cca", False)
        self.procrustes = config.get("procrustes", False)
        self.local_cca = config.get("local_cca", False)
        self.sparse_cca = config.get("sparse_cca", False)

        # Centering and whitening
        self.register_buffer("x_mean", None)
        self.register_buffer("y_mean", None)
        self.register_buffer("Wxx", None)
        self.register_buffer("Wyy", None)
        self.register_buffer("Sxx", None)
        self.register_buffer("Syy", None)
        self.register_buffer("Sxy_w", None)

    def _cca_prep(self, X, mean):
        X = X.float()
        X = F.normalize(X, p=2, dim=1)
        return X - mean

    def project_cca_x(self, X):
        Z = self._cca_prep(X, self.x_mean) @ self.CCA_Wx
        return F.normalize(Z, p=2, dim=1)

    def project_cca_y(self, Y):
        Z = self._cca_prep(Y, self.y_mean) @ self.CCA_Wy
        return F.normalize(Z, p=2, dim=1)

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

        Xp = F.normalize(X_pairs.float(), p=2, dim=1)
        Yp = F.normalize(Y_pairs.float(), p=2, dim=1)

        Xc = Xp - Xp.mean(dim=0, keepdim=True)
        Yc = Yp - Yp.mean(dim=0, keepdim=True)

        denom = max(Xc.size(0) - 1, 1)
        self.Sxy = (Xc.T @ Yc) / denom

    def precompute_cca_projections(self):
        """
        Compute CCA projection matrices.
        X: (n_samples, d_x)
        Y: (n_samples, d_y)
        """
        device = self.Sxx.device

        Sigma_xx = (
            self.Sxx + torch.eye(self.Sxx.size(0), device=device) * self.cca_lam_x
        )
        Sigma_yy = (
            self.Syy + torch.eye(self.Syy.size(0), device=device) * self.cca_lam_y
        )

        # compute A^{-1/2} via eigendecomposition
        def compute_inv_sqrt(Sigma, eps=1e-6, topk=None):
            L, V = torch.linalg.eigh(Sigma)  # ascending
            L = torch.clamp(L, min=eps)

            if topk is not None and topk < L.numel():
                idx = torch.arange(L.numel() - topk, L.numel(), device=L.device)
                L = L[idx]
                V = V[:, idx]

            inv_sqrt = (1.0 / torch.sqrt(L)).clamp_max(1e4)
            return (V * inv_sqrt.unsqueeze(0)) @ V.T

        Sxx_inv_sqrt = compute_inv_sqrt(
            Sigma_xx, eps=self.eig_eps, topk=self.cca_topk_x
        )
        Syy_inv_sqrt = compute_inv_sqrt(
            Sigma_yy, eps=self.eig_eps, topk=self.cca_topk_y
        )

        T = Sxx_inv_sqrt @ self.Sxy @ Syy_inv_sqrt
        U, S, Vt = torch.linalg.svd(T, full_matrices=False)

        self.CCA_Wx = Sxx_inv_sqrt @ U
        self.CCA_Wy = Syy_inv_sqrt @ Vt.T

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

    def init_clusters(
        self, X_pairs: torch.Tensor, Y_pairs: torch.Tensor, n_clusters=128, use_cca=True
    ):
        print(f"Initializing Anchors (k={n_clusters}) | CCA: {use_cca} ...")
        device = X_pairs.device
        dtype = X_pairs.dtype

        # crucial for clustering
        X_norm = F.normalize(X_pairs, p=2, dim=1)
        Y_norm = F.normalize(Y_pairs, p=2, dim=1)

        if use_cca:
            with torch.no_grad():
                X_cent = X_pairs - self.x_mean
                Y_cent = Y_pairs - self.y_mean

                X_proj = X_cent @ self.CCA_Wx
                Y_proj = Y_cent @ self.CCA_Wy

                X_proj = F.normalize(X_proj, p=2, dim=1)
                Y_proj = F.normalize(Y_proj, p=2, dim=1)

                Z_joint = (X_proj + Y_proj) / 2.0
                Z_joint = F.normalize(Z_joint, p=2, dim=1)

            Z_np = Z_joint.float().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(Z_np)

            centroids = torch.tensor(
                kmeans.cluster_centers_, device=device, dtype=dtype
            )
            centroids = F.normalize(centroids, p=2, dim=1)

            self.register_buffer("anchor_X", centroids)
            self.register_buffer("anchor_Y", centroids)
            self.use_cca_anchors = True

        else:
            Z_joint = torch.cat([X_norm, Y_norm], dim=1)

            Z_np = Z_joint.float().cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(Z_np)

            centroids_joint = torch.tensor(
                kmeans.cluster_centers_, device=device, dtype=dtype
            )

            dim_x = X_norm.shape[1]
            cx = centroids_joint[:, :dim_x]
            cy = centroids_joint[:, dim_x:]

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

        dist = 1 - cosine_similarity_matrix(fX, fY)

        res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)

        return res["plan"], res["log_plan"]

    def match_in_anchor_cca(self, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            Xc = self.project_cca_x(X)
            Yc = self.project_cca_y(Y)

            sim = torch.mm(Xc, Yc.T)
            dist = 1.0 - sim

            res = sinkhorn(
                dist,
                epsilon=self.epsilon_sinkhorn_anchor,
                max_iter=self.n_iters_sinkhorn_anchor,
            )

            return dist, res["plan"], res["log_plan"]

    def match_in_anchor_procrustes(self, X: torch.Tensor, Y: torch.Tensor):
        with torch.no_grad():
            X_norm = F.normalize(X, p=2, dim=1)
            Y_norm = F.normalize(Y, p=2, dim=1)

            X_cent = X_norm - self.x_mean
            Y_cent = Y_norm - self.y_mean

            X_white = X_cent @ self.W_zca_x
            Y_white = Y_cent @ self.W_zca_y

            # rotate
            X_aligned = X_white @ self.R_procrustes

            X_final = F.normalize(X_aligned, p=2, dim=1)
            Y_final = F.normalize(Y_white, p=2, dim=1)

            sim = torch.mm(X_final, Y_final.T)
            dist = 1.0 - sim

            res = sinkhorn(
                dist,
                epsilon=self.epsilon_sinkhorn_anchor,
                max_iter=self.n_iters_sinkhorn_anchor,
            )
            return dist, res["plan"], res["log_plan"]

    def fit_kcca(
        self,
        X_pairs: torch.Tensor,
        Y_pairs: torch.Tensor,
        kappa=1e-3,
        sigma=None,
        top_k=128,
    ):
        print(f"Fitting Robust Kernel CCA on {len(X_pairs)} pairs (Float32)...")

        X_pairs = X_pairs.float()
        Y_pairs = Y_pairs.float()

        device = X_pairs.device
        N = X_pairs.shape[0]

        X_pairs = F.normalize(X_pairs, p=2, dim=1)
        Y_pairs = F.normalize(Y_pairs, p=2, dim=1)

        self.register_buffer("kcca_X_anchors", X_pairs.detach().clone())
        self.register_buffer("kcca_Y_anchors", Y_pairs.detach().clone())

        def compute_sq_dist_mat(A):
            norm_sq = (A**2).sum(1).view(-1, 1)
            dist_sq = norm_sq + norm_sq.view(1, -1) - 2.0 * (A @ A.T)
            return dist_sq.clamp(min=0.0)

        dist_X = compute_sq_dist_mat(X_pairs)
        dist_Y = compute_sq_dist_mat(Y_pairs)

        if sigma is None:
            idx = torch.randperm(N)[:4096]
            median_X = torch.median(dist_X[idx][:, idx].view(-1))
            median_Y = torch.median(dist_Y[idx][:, idx].view(-1))
            sigma_x = median_X.sqrt().item()
            sigma_y = median_Y.sqrt().item()
            gamma_x = 1.0 / (2 * sigma_x**2 + 1e-6)
            gamma_y = 1.0 / (2 * sigma_y**2 + 1e-6)
        else:
            gamma_x = gamma_y = 1.0 / (2 * sigma**2 + 1e-6)

        self.kcca_gamma_x = gamma_x
        self.kcca_gamma_y = gamma_y

        Kx = torch.exp(-gamma_x * dist_X)
        Ky = torch.exp(-gamma_y * dist_Y)

        I_n = torch.eye(N, device=device, dtype=torch.float32)
        One_n = torch.ones(N, N, device=device, dtype=torch.float32) / N

        def center_kernel(K):
            return (
                K
                - torch.mm(One_n, K)
                - torch.mm(K, One_n)
                + torch.mm(torch.mm(One_n, K), One_n)
            )

        Kx = center_kernel(Kx)
        Ky = center_kernel(Ky)

        def compute_inv_sqrt(K, reg):
            K_reg = K + reg * I_n
            L, V = torch.linalg.eigh(K_reg)
            L = torch.clamp(L, min=1e-6)
            L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L))

            return torch.mm(torch.mm(V, L_inv_sqrt), V.T)

        Kx_inv_sqrt = compute_inv_sqrt(Kx, kappa)
        Ky_inv_sqrt = compute_inv_sqrt(Ky, kappa)

        C = torch.mm(torch.mm(Kx_inv_sqrt, Ky), Ky_inv_sqrt)

        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        self.kcca_alpha = torch.mm(Kx_inv_sqrt, U[:, :top_k]).contiguous()
        self.kcca_beta = torch.mm(Ky_inv_sqrt, Vh[:top_k, :].T).contiguous()

        print(f"KCCA Fit Complete. Top 5 correlations: {S[:5].cpu().numpy()}")
        self.use_kcca = True

    def match_in_anchor_kcca(self, X, Y):
        """
        Projects new data X and Y into the learned KCCA space.
        Args:
            X: (Batch, dx)
            Y: (Batch, dy)
        """
        with torch.no_grad():
            if not hasattr(self, "use_kcca"):
                raise RuntimeError("Must call fit_kcca before projection.")

            eps, iters = self.epsilon_sinkhorn_anchor, self.n_iters_sinkhorn_anchor

            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            dist_x = torch.cdist(X, self.kcca_X_anchors, p=2).pow(2)
            dist_y = torch.cdist(Y, self.kcca_Y_anchors, p=2).pow(2)

            Kx_new = torch.exp(-self.kcca_gamma_x * dist_x)
            Ky_new = torch.exp(-self.kcca_gamma_y * dist_y)

            X_proj = Kx_new @ self.kcca_alpha
            Y_proj = Ky_new @ self.kcca_beta

            X_proj = F.normalize(X_proj, p=2, dim=1)
            Y_proj = F.normalize(Y_proj, p=2, dim=1)

            sim = torch.mm(X_proj, Y_proj.T)
            dist = 1.0 - sim

            res = sinkhorn(dist, epsilon=eps, max_iter=iters)
            return dist, res["plan"], res["log_plan"]

    def _solve_cca(self, X, Y, kappa, top_k):
        """
        Helper: Solves robust CCA for a specific subset of data.
        Returns: Wx, Wy (Projection matrices)
        """
        device = X.device
        N = X.shape[0]

        X_c = X - X.mean(dim=0, keepdim=True)
        Y_c = Y - Y.mean(dim=0, keepdim=True)

        I_x = torch.eye(X.shape[1], device=device)
        I_y = torch.eye(Y.shape[1], device=device)

        Sxx = (X_c.T @ X_c) / (N - 1) + kappa * I_x
        Syy = (Y_c.T @ Y_c) / (N - 1) + kappa * I_y
        Sxy = (X_c.T @ Y_c) / (N - 1)

        def get_inv_sqrt(Cov):
            L, V = torch.linalg.eigh(Cov)
            L = torch.clamp(L, min=1e-6)
            return V @ torch.diag(1.0 / torch.sqrt(L)) @ V.T

        Sxx_inv_sqrt = get_inv_sqrt(Sxx)
        Syy_inv_sqrt = get_inv_sqrt(Syy)

        T = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        U, S, Vh = torch.linalg.svd(T, full_matrices=False)

        Wx = Sxx_inv_sqrt @ U[:, :top_k]
        Wy = Syy_inv_sqrt @ Vh[:top_k, :].T

        return Wx, Wy

    def fit_local_cca(self, X_pairs, Y_pairs, n_clusters=5, kappa=0.01, top_k=256):
        """
        Fits K distinct CCA models.
        """
        print(f"Fitting Local CCA with {n_clusters} clusters (kappa={kappa})...")
        device = X_pairs.device
        dtype = X_pairs.dtype
        dim_x = X_pairs.shape[1]

        # cluster joint space
        X_norm = F.normalize(X_pairs, p=2, dim=1)
        Y_norm = F.normalize(Y_pairs, p=2, dim=1)
        Joint = torch.cat([X_norm, Y_norm], dim=1).float()

        from sklearn.cluster import KMeans

        joint_np = Joint.detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels_np = kmeans.fit_predict(joint_np)

        centroids_all = torch.tensor(
            kmeans.cluster_centers_, device=device, dtype=dtype
        )
        self.local_centroids_x = centroids_all[:, :dim_x]

        labels = torch.tensor(labels_np, device=device)
        self.local_models = []
        active_clusters = 0

        for k in range(n_clusters):
            mask = labels == k
            X_k = X_pairs[mask].float()
            Y_k = Y_pairs[mask].float()

            # Safety: Skip if cluster is too small to estimate covariance
            if len(X_k) < top_k * 2:
                print(
                    f"  > Warning: Cluster {k} has {len(X_k)} samples. Using Global Fallback (Identity)."
                )
                self.local_models.append(None)
                continue

            # Solve CCA for this specific cluster
            Wx, Wy = self._solve_cca(X_k, Y_k, kappa, top_k)

            self.local_models.append(
                {
                    "mean_x": X_k.mean(dim=0, keepdim=True).to(dtype),
                    "mean_y": Y_k.mean(dim=0, keepdim=True).to(dtype),
                    "Wx": Wx.to(dtype),
                    "Wy": Wy.to(dtype),
                }
            )
            active_clusters += 1

        print(f"Local KCCA Ready. {active_clusters}/{n_clusters} clusters active.")
        self.use_local_cca = True

    def match_in_anchor_local_cca(self, X, Y):
        """
        Computes Transport Plan using Local CCA.
        Strategy:
          1. Assign each IMAGE x_i to its nearest cluster k.
          2. Compare x_i ONLY using projection k.
          3. Project ALL texts Y using projection k to compare against x_i.
        """
        if not hasattr(self, "use_local_cca"):
            raise RuntimeError("Run fit_local_kcca first.")

        X = F.normalize(X.float(), p=2, dim=1)
        Y = F.normalize(Y.float(), p=2, dim=1)

        N_x = X.shape[0]
        N_y = Y.shape[0]

        dist_matrix = torch.zeros(N_x, N_y, device=X.device, dtype=X.dtype)

        X_norm = F.normalize(X, p=2, dim=1)
        C_norm = F.normalize(self.local_centroids_x, p=2, dim=1)

        sims = torch.mm(X_norm, C_norm.T)
        cluster_assignments = torch.argmax(sims, dim=1)

        unique_clusters = torch.unique(cluster_assignments)

        for k in unique_clusters:
            k_idx = k.item()
            model = self.local_models[k_idx]

            mask_x = cluster_assignments == k
            indices_x = torch.nonzero(mask_x).squeeze(1)

            if model is None:
                X_sub = X_norm[indices_x]
                Y_sub = F.normalize(Y, p=2, dim=1)
                sim_block = torch.mm(X_sub, Y_sub.T)
                dist_matrix[indices_x] = 1.0 - sim_block
                continue

            X_sub = X[indices_x]
            X_proj = (X_sub - model["mean_x"]) @ model["Wx"]
            X_proj = F.normalize(X_proj, p=2, dim=1)

            Y_proj = (Y - model["mean_y"]) @ model["Wy"]
            Y_proj = F.normalize(Y_proj, p=2, dim=1)

            sim_block = torch.mm(X_proj, Y_proj.T)
            dist_block = 1.0 - sim_block
            dist_matrix[indices_x] = dist_block

        res = sinkhorn(
            dist_matrix,
            epsilon=self.epsilon_sinkhorn_anchor,
            max_iter=self.n_iters_sinkhorn_anchor,
        )

        return dist_matrix, res["plan"], res["log_plan"]

    def fit_sparse_cca(self, X, Y, penalty_x=0.9, penalty_y=0.9, max_iter=20, tol=1e-4):
        """
        Fits Sparse CCA.

        Args:
            penalty_x, penalty_y (float): 0.1 to 0.9.
                Smaller value = More Sparsity (Select fewer features).
                Larger value (approx 1.0) = Standard CCA (Use all features).
                Recommended start: 0.3 - 0.5.
        """
        print(f"Fitting Sparse CCA (Px={penalty_x}, Py={penalty_y})...")
        device = X.device

        X = X.float()
        Y = Y.float()

        X_cent = X - X.mean(dim=0, keepdim=True)
        Y_cent = Y - Y.mean(dim=0, keepdim=True)

        X_std = X_cent.std(dim=0, keepdim=True) + 1e-6
        Y_std = Y_cent.std(dim=0, keepdim=True) + 1e-6
        X_norm = X_cent / X_std
        Y_norm = Y_cent / Y_std

        K = (X_norm.T @ Y_norm) / (X.shape[0] - 1)

        U, S, V = torch.linalg.svd(K, full_matrices=False)
        u = U[:, 0].unsqueeze(1)  # (Dx, 1)
        v = V[0, :].unsqueeze(1)  # (Dy, 1)

        def soft_threshold(w, penalty):
            curr_norm = torch.norm(w, p=1)
            if curr_norm <= penalty:
                return w

            w_abs = w.abs()
            lo, hi = 0.0, w_abs.max().item()
            for _ in range(10):
                mid = (lo + hi) / 2
                new_norm = torch.sum(torch.relu(w_abs - mid))
                if new_norm > penalty:
                    lo = mid
                else:
                    hi = mid
            delta = lo
            return torch.sign(w) * torch.relu(w_abs - delta)

        for i in range(max_iter):
            u_old = u.clone()

            u = torch.mm(K, v)
            u = soft_threshold(u, penalty_x * math.sqrt(X.shape[1]))
            u = F.normalize(u, p=2, dim=0)

            v = torch.mm(K.T, u)
            v = soft_threshold(v, penalty_y * math.sqrt(Y.shape[1]))
            v = F.normalize(v, p=2, dim=0)

            diff = torch.norm(u - u_old)
            if diff < tol:
                break

        print(f"SCCA Converged in {i+1} iterations.")

        self.SCCA_Wx = u
        self.SCCA_Wy = v
        self.mean_x = X.mean(dim=0, keepdim=True)
        self.mean_y = Y.mean(dim=0, keepdim=True)
        self.use_scca = True

        sparsity_x = (u.abs() < 1e-5).float().mean().item()
        print(
            f"Sparsity achieved: X={sparsity_x:.1%}, Y={(v.abs() < 1e-5).float().mean().item():.1%}"
        )

    def match_in_anchor_sparse_cca(self, X, Y):
        with torch.no_grad():
            if not hasattr(self, "use_scca"):
                raise RuntimeError("Call fit_sparse_cca first.")

            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_c = X - self.mean_x
            Y_c = Y - self.mean_y

            X_p = X_c @ self.SCCA_Wx
            Y_p = Y_c @ self.SCCA_Wy

            X_p = F.normalize(X_p, p=2, dim=1)
            Y_p = F.normalize(Y_p, p=2, dim=1)

            sim = torch.mm(X_p, Y_p.T)
            dist = 1.0 - sim

            res = sinkhorn(
                dist,
                epsilon=self.epsilon_sinkhorn_anchor,
                max_iter=self.n_iters_sinkhorn_anchor,
            )
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
        Computes divergence using cosine or frobenius norm
        """
        with torch.autocast(device_type=X.device.type, enabled=False):

            # cast to float
            X_f32 = X.float()
            Y_f32 = Y.float()
            fX_f32 = fX.float()
            fY_f32 = fY.float()
            Wx_f32 = self.CCA_Wx.float()
            Wy_f32 = self.CCA_Wy.float()
            mean_x_f32 = self.x_mean.float()
            mean_y_f32 = self.y_mean.float()

            X_f32 = F.normalize(X_f32, p=2, dim=1)
            Y_f32 = F.normalize(Y_f32, p=2, dim=1)

            X_cca = (X_f32 - mean_x_f32) @ Wx_f32
            Y_cca = (Y_f32 - mean_y_f32) @ Wy_f32
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            fX_norm = F.normalize(fX_f32, p=2, dim=1)
            fY_norm = F.normalize(fY_f32, p=2, dim=1)

            Cxx = torch.mm(X_cca.T, X_cca)
            Cyy = torch.mm(Y_cca.T, Y_cca)
            norm1 = (Cxx * Cyy).sum()

            Dxx = torch.mm(fX_norm.T, fX_norm)
            Dyy = torch.mm(fY_norm.T, fY_norm)
            norm2 = (Dxx * Dyy).sum()

            Mx = torch.mm(X_cca.T, fX_norm)
            My = torch.mm(Y_cca.T, fY_norm)
            dot = (Mx * My).sum()

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

        """
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature

            prob_x2y = F.softmax(logits_anchor, dim=1)
            prob_y2x = F.softmax(logits_anchor, dim=0)

            target_matrix = torch.sqrt(prob_x2y * prob_y2x)
            target_matrix = target_matrix / target_matrix.sum().clamp(min=1e-8)

        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)

        logits_latent = torch.mm(fX, fY.T) / temperature
        log_prob_latent = F.log_softmax(logits_latent.view(-1), dim=0)

        loss = F.kl_div(log_prob_latent, target_matrix.view(-1), reduction="sum")

        return loss

    def loss_semisupervised_conditional_kl(self, X, Y, fX, fY, temperature=0.1):
        """
        Computes symmetric row-wise KL divergence (Conditional Probability).
        """
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)
            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature

        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)
        logits_latent = torch.mm(fX, fY.T) / temperature

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
        """
        with torch.no_grad():
            X = F.normalize(X, p=2, dim=1)
            Y = F.normalize(Y, p=2, dim=1)

            X_cca = (X - self.x_mean) @ self.CCA_Wx
            Y_cca = (Y - self.y_mean) @ self.CCA_Wy
            X_cca = F.normalize(X_cca, p=2, dim=1)
            Y_cca = F.normalize(Y_cca, p=2, dim=1)

            logits_anchor = torch.mm(X_cca, Y_cca.T) / temperature
            target_prob = F.softmax(logits_anchor.view(-1), dim=0)

        fX = F.normalize(fX, p=2, dim=1)
        fY = F.normalize(fY, p=2, dim=1)

        logits_latent = torch.mm(fX, fY.T) / temperature
        log_pred_prob = F.log_softmax(logits_latent.view(-1), dim=0)

        loss = F.kl_div(log_pred_prob, target_prob, reduction="sum")

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


class OptimizedMatchingModel(MatchingModel):
    def __init__(self, config):
        super(OptimizedMatchingModel, self).__init__(config)

        self.tol_sinkhorn = config["tol_sinkhorn"]
        self.match_all = config[
            "match_all"
        ]  # match supervised and unsupervised samples together

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
                tol=self.tol_sinkhorn,
            )
            plan = res["plan"]
            log_plan = res["log_plan"]
            n_iters = res["n_iters"]

        return dist, plan, log_plan, n_iters

    def match_in_anchor_cca(self, X: torch.Tensor, Y: torch.Tensor):
        eps, iters = self.epsilon_sinkhorn_anchor, self.n_iters_sinkhorn_anchor

        X = F.normalize(X, p=2, dim=1)
        Y = F.normalize(Y, p=2, dim=1)

        Xc = (X - self.x_mean) @ self.CCA_Wx
        Yc = (Y - self.y_mean) @ self.CCA_Wy

        Xc = F.normalize(Xc, p=2, dim=1)
        Yc = F.normalize(Yc, p=2, dim=1)

        sim = torch.mm(Xc, Yc.T)
        dist = 1.0 - sim  # (nx, ny)

        with torch.no_grad():
            res = sinkhorn(
                dist,
                epsilon=eps,
                max_iter=iters,
                tol=self.tol_sinkhorn,
            )
            plan = res["plan"]
            log_plan = res["log_plan"]
            n_iters = res["n_iters"]

        return dist, plan, log_plan, n_iters

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
        target_plan[:n_pairs, :n_pairs] = (
            torch.eye(n_pairs, device=plan_all.device) / n_pairs
        )
        target_plan[n_pairs:, n_pairs:] = plan_anchor

        loss_semisupervised_correct_grad = (target_plan - plan_all).detach() * dist_all
        loss_semisupervised_correct_grad = (
            loss_semisupervised_correct_grad.sum() / self.epsilon_sinkhorn_shared
        )

        loss_semisupervised_no_grad_11 = (
            -log_fct(n_pairs) - log_plan_all.diag()[:n_pairs].mean()
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

        dist_anchor, plan_anchor, log_plan_anchor, n_iters_anchor = (
            self.match_in_anchor_cca(X, Y)
        )
        dist_pairs, plan_pairs, log_plan_pairs, _ = self.match_in_latent(
            fX_pairs, fY_pairs
        )
        dist, plan, log_plan, n_iters = self.match_in_latent(fX, fY)

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

        if self.match_all:
            dist_all, plan_all, log_plan_all, n_iters = self.match_in_latent(
                torch.cat([fX_pairs, fX], dim=0), torch.cat([fY_pairs, fY], dim=0)
            )

        loss_supervised_sail = zero
        loss_semisupervised_sail = zero
        loss_semisupervised_ot = zero

        # TODO discuss supervised OT loss Paul used on synthetic data
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
            if self.match_all:
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

        loss = (
            self.alpha_supervised_sail * loss_supervised_sail
            + self.alpha_semisupervised_sail * loss_semisupervised_sail
            + self.alpha_semisupervised_ot * loss_semisupervised_ot
        )

        log = {
            "loss_supervised_sail": loss_supervised_sail.item(),
            "loss_semisupervised_sail": loss_semisupervised_sail.item(),
            "loss_semisupervised_ot": loss_semisupervised_ot.item(),
            "n_iters_sinkhorn_anchor": n_iters_anchor,
            "n_iters_sinkhorn_shared": n_iters,
            "total_loss": loss.item(),
        }

        return loss, log
