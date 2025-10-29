# Define model
import torch.nn as nn
import torch
from optimal_transport.ot_simplified import (
    basic_quad_loss,
    basic_lin_loss,
    basic_marginal_loss,
    sinkhorn,
)
from abc import abstractmethod
from optimal_transport.utils import cosine_similarity_matrix, log_softmax_row_col
from math import log
import torch.nn.functional as F


class MatchingModel(nn.Module):

    def __init__(self, config):
        super(MatchingModel, self).__init__()
        self.check_config(config)
        self.config = config
        # self.init_mapping()

    @abstractmethod
    def check_config(self, config: dict):
        required = [
            "alpha_unsupervised",
            "alpha_supervised_explicit",
            "alpha_supervised_implicit",
            "alpha_marginal",
        ]
        for key in required:
            assert key in config, f"config must have attribute {key}"

    # @abstractmethod
    # def init_mapping(self):
    #     """
    #     Initialize the mapping functions (ex: MLPs)
    #     """
    #     self.fc1 = ...
    #     self.fc2 = ...

    @abstractmethod
    def match_in_latent(self, Y1: torch.Tensor, Y2: torch.Tensor):
        """
        Compute the transport plan between the two sets of encoded points in the latent space.
        Return both the plan and its log. Try to use the most stable computation possible (e.g. log-sum-exp trick when available).

        Inputs:
            Y1: (b, d_latent)
            Y2: (b, d_latent)
        Outputs:
            plan: (b, b) transport plan between Y1 and Y2
            log_plan: (b, b) log of the transport plan
        """
        plan = ...
        log_plan = ...
        return plan, log_plan

    # def encode(self, X1: torch.Tensor, X2: torch.Tensor):
    #     """
    #     Encode the inputs (left and right) into the common latent space.

    #     Inputs:
    #         X1: (b, d1)
    #         X2: (b, d2)
    #     Outputs:
    #         Y1: (b, d_latent)
    #         Y2: (b, d_latent)
    #     """
    #     Y1 = self.fc1(X1)
    #     Y2 = self.fc2(X2)
    #     return Y1, Y2

    def loss_supervised_explicit(
        self, Y1: torch.Tensor, Y2: torch.Tensor, is_pair: torch.Tensor
    ):
        """
        Explicitly encourages Y1 = Y2 for paired samples.

        Inputs:
            Y1: (b, d_latent)
            Y2: (b, d_latent)
            is_pair: (b) tensor of booleans indicating if the samples are paired or not
        """
        loss = ((Y1 - Y2) ** 2).sum(dim=1)
        loss = loss[is_pair].mean()
        return loss

    def loss_supervised_implicit(self, log_plan: torch.Tensor, is_pair: torch.Tensor):
        """
        Encourages to recover the pairs i.e. max T_ii for paired samples.
        """
        n = len(log_plan)
        log_plan_diag = torch.diag(log_plan)[is_pair] + log(n)
        loss = -log_plan_diag
        loss = loss.mean()
        return loss

    def loss_marginal(self, plan: torch.Tensor):
        """
        Encourages the marginals of the plan to be uniform.
        """
        n = plan.shape[0]
        a = torch.ones(n, device=plan.device) / n
        b = torch.ones(n, device=plan.device) / n
        marg_loss = basic_marginal_loss(plan, a, b)
        return marg_loss

    def loss_unsupervised(self, X1: torch.Tensor, X2: torch.Tensor, plan: torch.Tensor):
        """
        Unsupervised loss that encourages the plan to preserve the cosine similarity in the original space.
        """
        C1 = cosine_similarity_matrix(X1, X1)
        C2 = cosine_similarity_matrix(X2, X2)
        quad_loss = basic_quad_loss(C1, C2, plan)
        return quad_loss

    def forward(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        Y1: torch.Tensor,
        Y2: torch.Tensor,
        is_pair: torch.Tensor,
    ):
        """
        Compute the loss between two batches of samples.

        Inputs:
            X1: (b, d1)
            X2: (b, d2)
            is_pair: (b) tensor of booleans indicating if the samples are paired or not
        Outputs:
            loss: scalar
            log: dictionary with any useful information, e.g. loss components
        """
        # we perform encoding in the model, here only compute loss
        # Y1, Y2 = self.encode(X1, X2)
        # TODO should we L2 normalize Y1, Y2 here?
        X1 = F.normalize(X1, p=2, dim=-1)
        X2 = F.normalize(X2, p=2, dim=-1)
        Y1 = F.normalize(Y1, p=2, dim=-1)
        Y2 = F.normalize(Y2, p=2, dim=-1)

        plan, log_plan = self.match_in_latent(Y1, Y2)

        loss1 = self.loss_supervised_explicit(Y1, Y2, is_pair)
        loss2 = self.loss_supervised_implicit(log_plan, is_pair)
        loss3 = self.loss_marginal(plan)
        loss4 = self.loss_unsupervised(X1, X2, plan)

        alpha_supervised_explicit = self.config["alpha_supervised_explicit"]
        alpha_supervised_implicit = self.config["alpha_supervised_implicit"]
        alpha_marginal = self.config["alpha_marginal"]
        alpha_unsupervised = self.config["alpha_unsupervised"]

        loss = (
            loss1 * alpha_supervised_explicit
            + loss2 * alpha_supervised_implicit
            + loss3 * alpha_marginal
            + loss4 * alpha_unsupervised
        )

        log = {
            "loss_supervised_explicit": loss1.item(),
            "loss_supervised_implicit": loss2.item(),
            "loss_marginal": loss3.item(),
            "loss_unsupervised": loss4.item(),
            "total_loss": loss.item(),
        }

        return loss, log

    # def forward(self, Y1: torch.Tensor, Y2: torch.Tensor):
    #     # Y1, Y2 = self.encode(X1, X2)
    #     plan, _ = self.match_in_latent(Y1, Y2)
    #     return Y1, Y2, plan


class BasicMatchingModel(MatchingModel):

    def check_config(self, config: dict):
        super().check_config(config)
        required = ["epsilon"]
        for key in required:
            assert key in config, f"config must have attribute {key}"

    # def init_mapping(self):
    #     self.fc1 = nn.Linear(
    #         self.config["n_features_left"], self.config["n_features_latent"]
    #     )
    #     self.fc2 = nn.Linear(
    #         self.config["n_features_right"], self.config["n_features_latent"]
    #     )

    def match_in_latent(self, Y1, Y2):
        """
        The plan is simply an affinity matrix in the latent space.
        The marginals are not enforced to be uniform.
        """
        n = Y1.shape[0]
        # Compute cost matrix in latent space
        affinity = cosine_similarity_matrix(Y1, Y2) / self.config["epsilon"]
        # Compute transport plan (simple softmax)
        log_plan = log_softmax_row_col(affinity) - log(n)
        plan = log_plan.exp()
        return plan, log_plan


class SinkhornMatchingModel(MatchingModel):

    def check_config(self, config: dict):
        super().check_config(config)
        required = ["epsilon", "n_iters_sinkhorn"]
        for key in required:
            assert key in config, f"config must have attribute {key}"

    # def init_mapping(self):
    #     self.fc1 = nn.Linear(
    #         self.config["n_features_left"], self.config["n_features_latent"]
    #     )
    #     self.fc2 = nn.Linear(
    #         self.config["n_features_right"], self.config["n_features_latent"]
    #     )

    def match_in_latent(self, Y1, Y2):
        """
        The plan is simply an affinity matrix in the latent space.
        The marginals are not enforced to be uniform.
        """
        # Compute transport plan (sinkhorn)
        M = -cosine_similarity_matrix(Y1, Y2)  # Cost matrix
        res = sinkhorn(
            M,
            epsilon=self.config["epsilon"],
            a=None,
            b=None,
            max_iter=self.config["n_iters_sinkhorn"],
            tol=1e-5,
            symmetric=True,
        )
        plan = res["plan"]
        log_plan = res["log_plan"]
        return plan, log_plan
