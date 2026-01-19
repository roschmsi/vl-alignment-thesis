from abc import ABC, abstractmethod
import torch
import numpy as np
import torch.nn.functional as F


class Affinity(ABC):

    @abstractmethod
    def fit(self, X_pairs, Y_pairs):
        """
        Fit the the affinity model given pairs of matched samples
        X_pairs: Tensor of shape (n_pairs, d_x)
        Y_pairs: Tensor of shape (n_pairs, d_y)
        """
        pass

    def __call__(self, X, Y):
        """
        Compute the affinity matrix between samples in X and Y
        X: Tensor of shape (n_x, d_x)
        Y: Tensor of shape (n_y, d_y)
        Returns: Tensor of shape (n_x, n_y)
        """
        return NotImplemented

    def top_k_retrieval(self, X_pairs, Y_pairs, k=1):
        """
        Evaluate the affinity model on pairs of matched samples (from a validation set ideally)
        X_pairs: Tensor of shape (n_pairs, d_x)
        Y_pairs: Tensor of shape (n_pairs, d_y)
        Returns: Tensor of shape (n_pairs,)
        """
        K = self(X_pairs, Y_pairs)
        topk = torch.argsort(K, axis=1)[:, -k:]
        targets = torch.arange(len(X_pairs), device=K.device)
        top_acc = np.mean(
            np.array([targets[i] in topk[i] for i in range(len(X_pairs))])
        )
        return top_acc


class CCA(Affinity):
    def __init__(self, n_components, lam_x, lam_y, eps):
        self.n_components = n_components
        self.lam_x = lam_x
        self.lam_y = lam_y
        self.eps = eps
        self.Wx = None
        self.Wy = None

    def fit(self, X_pairs, Y_pairs, x_mean=None, y_mean=None):
        """
        Compute CCA projection matrices.
        X: (n_samples, d_x)
        Y: (n_samples, d_y)
        """
        X_pairs = X_pairs.float()
        Y_pairs = Y_pairs.float()

        X_pairs = F.normalize(X_pairs.float(), p=2, dim=1)
        Y_pairs = F.normalize(Y_pairs.float(), p=2, dim=1)

        self.x_mean = X_pairs.mean(axis=0, keepdims=True) if x_mean is None else x_mean
        self.y_mean = Y_pairs.mean(axis=0, keepdims=True) if y_mean is None else y_mean

        X_pairs = X_pairs - self.x_mean
        Y_pairs = Y_pairs - self.y_mean

        n_pairs = len(X_pairs)
        Sxx = X_pairs.T @ X_pairs / (n_pairs - 1)
        Syy = Y_pairs.T @ Y_pairs / (n_pairs - 1)
        Sxy = X_pairs.T @ Y_pairs / (n_pairs - 1)

        Sigma_xx = Sxx + torch.eye(Sxx.size(0), device=X_pairs.device) * self.lam_x
        Sigma_yy = Syy + torch.eye(Syy.size(0), device=Y_pairs.device) * self.lam_y

        # compute A^{-1/2} via eigendecomposition
        def compute_inv_sqrt(Sigma, eps):
            # eigenvalues (L) and eigenvectors (V)
            L, V = torch.linalg.eigh(Sigma)
            L = torch.clamp(L, min=eps)
            L_inv_sqrt = torch.diag(1.0 / torch.sqrt(L))
            return V @ L_inv_sqrt @ V.T

        Sxx_inv_sqrt = compute_inv_sqrt(Sigma_xx, self.eps)
        Syy_inv_sqrt = compute_inv_sqrt(Sigma_yy, self.eps)

        T = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt
        U, S, Vt = torch.linalg.svd(T, full_matrices=False)

        if self.n_components is not None:
            U = U[:, : self.n_components]
            Vt = Vt[: self.n_components, :]

        self.Wx = Sxx_inv_sqrt @ U
        self.Wy = Syy_inv_sqrt @ Vt.T

    def __call__(self, X, Y):
        X = F.normalize(X.float(), p=2, dim=1)
        Y = F.normalize(Y.float(), p=2, dim=1)

        X = X - self.x_mean
        Y = Y - self.y_mean

        X_proj = X @ self.Wx
        X_proj = X_proj / torch.linalg.norm(X_proj, axis=1, keepdims=True)

        Y_proj = Y @ self.Wy
        Y_proj = Y_proj / torch.linalg.norm(Y_proj, axis=1, keepdims=True)

        K = X_proj @ Y_proj.T

        return K


class Anchors(Affinity):

    def fit(self, X_pairs, Y_pairs, x_mean=None, y_mean=None):
        self.n_pairs = len(X_pairs)
        self.x_mean = X_pairs.mean(axis=0, keepdims=True) if x_mean is None else x_mean
        self.y_mean = Y_pairs.mean(axis=0, keepdims=True) if y_mean is None else y_mean
        self.A = X_pairs - self.x_mean
        self.A = self.A / torch.linalg.norm(self.A, axis=1, keepdims=True)
        self.B = Y_pairs - self.y_mean
        self.B = self.B / torch.linalg.norm(self.B, axis=1, keepdims=True)
        # Precompute for efficiency
        self.Wxx = self.A.T @ self.A
        self.Wyy = self.B.T @ self.B
        self.Wxy = self.A.T @ self.B

    def forward_naive(self, X, Y):
        """
        Explicitly compute the anchor space representations of dimensions (n_anchors)
        """
        X = X - self.x_mean
        Y = Y - self.y_mean

        X_anchors = X @ self.A.T
        X_anchors = X_anchors / torch.linalg.norm(X_anchors, axis=1, keepdims=True)

        Y_anchors = Y @ self.B.T
        Y_anchors = Y_anchors / torch.linalg.norm(Y_anchors, axis=1, keepdims=True)

        K = X_anchors @ Y_anchors.T

        return K

    def forward_optimized(self, X, Y):
        """
        No explicit computation of the anchor space representations
        Use precomputed Gram matrices for efficiency
        """
        X = X - self.x_mean
        Y = Y - self.y_mean

        YWxyT = Y @ self.Wxy.T
        XWxx = X @ self.Wxx
        YWyy = Y @ self.Wyy

        X_norms = torch.sqrt((XWxx * X).sum(-1, keepdims=True))
        Y_norms = torch.sqrt((YWyy * Y).sum(-1, keepdims=True))

        K = (X @ YWxyT.T) / (X_norms * Y_norms.T)

        return K

    def __call__(self, X, Y):
        X = F.normalize(X.float(), p=2, dim=1)
        Y = F.normalize(Y.float(), p=2, dim=1)

        if X.shape[1] < self.n_pairs and Y.shape[1] < self.n_pairs:
            return self.forward_optimized(X, Y)
        else:
            return self.forward_naive(X, Y)

    def sanity_check(self, X, Y):
        K1 = self.forward_naive(X, Y)
        K2 = self.forward_optimized(X, Y)
        diff = torch.abs(K1 - K2).max().item()
        print(f"Max difference between naive and optimized: {diff:.6f}")
