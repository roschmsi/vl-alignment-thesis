# Define model
import torch.nn as nn
import torch
from optimal_transport.ot_simplified import (
    basic_marginal_loss,
    sinkhorn,
    optimized_quad_loss,
)
from optimal_transport.utils import cosine_similarity_matrix
from math import log


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
        self.alpha_semisupervised_div = config["alpha_semisupervised_div"]
        self.alpha_unsupervised = config["alpha_unsupervised"]

        self.epsilon_sinkhorn_shared = config["epsilon_sinkhorn_shared"]
        self.n_iters_sinkhorn_shared = config["n_iters_sinkhorn_shared"]
        self.epsilon_sinkhorn_anchor = config["epsilon_sinkhorn_anchor"]
        self.n_iters_sinkhorn_anchor = config["n_iters_sinkhorn_anchor"]

        self.temperature_sail = config["temperature_sail"]
        self.bias_sail = config["bias_sail"]

        # self.init_mapping()

    # def forward(
    #     self, X: torch.Tensor, Y: torch.Tensor, fX: torch.Tensor, fY: torch.Tensor
    # ):
    #     # fX, fY = self.encode(X, Y)
    #     plan, log_plan = self.match_in_latent(fX, fY)
    #     return fX, fY, plan

    # def init_mapping(self):
    #     """
    #     Initialize the mapping functions (ex: MLPs)
    #     """
    #     self.f_X = nn.Linear(self.dx, self.d)
    #     self.f_Y = nn.Linear(self.dy, self.d)

    def precompute_anchor_covariances(
        self, X_anchor: torch.Tensor, Y_anchor: torch.Tensor
    ):
        """
        Precompute covariance matrices in anchor space.
        Inputs:
            X_anchor: (N_anchor, dx)
            Y_anchor: (N_anchor, dy)
        Outputs:
            Sxx: (dx, dx)
            Syy: (dy, dy)
            Sxy: (dx, dy)
        """
        X_anchor = X_anchor / torch.norm(X_anchor, dim=1, keepdim=True).clamp(min=1e-8)
        Y_anchor = Y_anchor / torch.norm(Y_anchor, dim=1, keepdim=True).clamp(min=1e-8)
        N_anchor = X_anchor.shape[0]
        Sxx = torch.mm(X_anchor.T, X_anchor) / N_anchor
        Syy = torch.mm(Y_anchor.T, Y_anchor) / N_anchor
        Sxy = torch.mm(X_anchor.T, Y_anchor) / N_anchor
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
        res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)
        plan = res["plan"]
        log_plan = res["log_plan"]
        return plan, log_plan

    def match_in_anchor(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        Sxx: torch.Tensor,
        Syy: torch.Tensor,
        Sxy: torch.Tensor,
    ):
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
        with torch.no_grad():

            epsilon_sinkhorn, n_iters_sinkhorn = (
                self.epsilon_sinkhorn_anchor,
                self.n_iters_sinkhorn_anchor,
            )

            # Compute distances in anchor space
            norm_X = (X * (X @ Sxx)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            norm_Y = (Y * (Y @ Syy)).sum(-1, keepdim=True).sqrt().clamp(min=1e-8)
            X_normalized = X / norm_X
            Y_normalized = Y / norm_Y
            dist = -torch.linalg.multi_dot([X_normalized, Sxy, Y_normalized.T])

            # Find OT plan in anchor space
            res = sinkhorn(dist, epsilon=epsilon_sinkhorn, max_iter=n_iters_sinkhorn)
            plan_anchor = res["plan"]
            log_plan_anchor = res["log_plan"]

        return plan_anchor, log_plan_anchor

    # def encode(self, X: torch.Tensor, Y: torch.Tensor):
    #     fX = self.f_X(X)
    #     fY = self.f_Y(Y)
    #     return fX, fY

    def loss_marginal(self, plan: torch.Tensor):
        """
        Encourages the marginals of the plan to be uniform.
        """
        n_x, n_y = plan.shape
        a = torch.ones(n_x, device=plan.device) / n_x
        b = torch.ones(n_y, device=plan.device) / n_y
        marg_loss = basic_marginal_loss(plan, a, b)
        return marg_loss

    def loss_SAIL(self, fX_pairs: torch.Tensor, fY_pairs: torch.Tensor):
        """
        Sigmoid loss from SAIL paper.
        Intuition: Contrastive learning.
        """
        cosine_latent = cosine_similarity_matrix(fX_pairs, fY_pairs)
        target = -1 + 2 * torch.eye(len(fX_pairs), device=fX_pairs.device)
        loss = -torch.mean(
            torch.nn.functional.logsigmoid(
                cosine_latent * self.temperature_sail * target - self.bias_sail
            )
        )
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
        fX: torch.Tensor,
        fY: torch.Tensor,
        fX_pairs: torch.Tensor,
        fY_pairs: torch.Tensor,
        Sxx: torch.Tensor,
        Syy: torch.Tensor,
        Sxy: torch.Tensor,
    ):
        # X_pairs: torch.Tensor,
        # Y_pairs: torch.Tensor,
        """
        Compute the loss between two batches of samples.

        Inputs:
            X1: (N1, dx) samples in left space
            X2: (N2, dy) samples in right space
            X_pairs: (N, dx) paired samples in left space
            Y_pairs: (N, dy) paired samples in right space
            Sxx: (dx, dx) constant to compute similarities in anchor space
            Syy: (dy, dy) idem
            Sxy: (dx, dy) idem
        Outputs:
            loss: scalar
            log: dictionary with any useful information, e.g. loss components
        """
        # Encode embeddings
        # fX, fY = self.encode(X, Y)
        # fX_pairs, fY_pairs = self.encode(X_pairs, Y_pairs)

        # Compute transport plans
        plan, log_plan = self.match_in_latent(fX, fY)
        plan_pairs, log_plan_pairs = self.match_in_latent(fX_pairs, fY_pairs)

        # Compute loss components
        loss_supervised_sail = self.loss_SAIL(fX_pairs=fX_pairs, fY_pairs=fY_pairs)
        loss_supervised_explicit = self.loss_supervised_explicit(
            fX_pairs=fX_pairs, fY_pairs=fY_pairs
        )
        loss_supervised_implicit = self.loss_supervised_implicit(
            log_plan_pairs=log_plan_pairs
        )
        loss_marginal = self.loss_marginal(plan) + self.loss_marginal(plan_pairs)
        loss_semisupervised_ot = self.loss_semisupervised_ot(
            X=X, Y=Y, Sxx=Sxx, Syy=Syy, Sxy=Sxy, log_plan=log_plan
        )
        loss_semisupervised_div = self.loss_semisupervised_div(
            X=X, Y=Y, Sxx=Sxx, Syy=Syy, Sxy=Sxy, fX=fX, fY=fY
        )
        loss_unsupervised = self.loss_unsupervised(X=X, Y=Y, plan=plan)

        loss = (
            self.alpha_marginal * loss_marginal
            + self.alpha_supervised_sail * loss_supervised_sail
            + self.alpha_supervised_explicit * loss_supervised_explicit
            + self.alpha_supervised_implicit * loss_supervised_implicit
            + self.alpha_semisupervised_ot * loss_semisupervised_ot
            + self.alpha_semisupervised_div * loss_semisupervised_div
            + self.alpha_unsupervised * loss_unsupervised
        )

        log = {
            "loss_marginal": loss_marginal.item(),
            "loss_supervised_sail": loss_supervised_sail.item(),
            "loss_supervised_explicit": loss_supervised_explicit.item(),
            "loss_supervised_implicit": loss_supervised_implicit.item(),
            "loss_semisupervised_ot": loss_semisupervised_ot.item(),
            "loss_semisupervised_div": loss_semisupervised_div.item(),
            "loss_unsupervised": loss_unsupervised.item(),
            "total_loss": loss.item(),
        }

        return loss, log
