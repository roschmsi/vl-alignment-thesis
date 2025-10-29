# No solvers here, only the most simple functions
import torch


def basic_quad_loss(C1: torch.Tensor, C2: torch.Tensor, T: torch.Tensor):
    """
    Compute the FGW loss:
        loss = < LxT, T > = sum_ijkl (C1_ij - C2_kl)^2 T_kl T_jl
    If T is a permutation matrix, this is equal to:
        loss = || C1 - TC2T^T||_2^2
    """
    a = T.sum(1)  # Sum over columns
    b = T.sum(0)  # Sum over rows
    LxT = (C1**2 @ a)[:, None] + (C2**2 @ b)[None, :] - 2 * (C1 @ T @ C2.T)
    quad_loss = torch.sum(LxT * T)
    return quad_loss


def basic_lin_loss(C: torch.Tensor, T: torch.Tensor):
    """
    Compute the linear loss:
        loss = < C, T > = sum_ij C_ij T_ij
    """
    lin_loss = torch.sum(C * T)
    return lin_loss


def basic_marginal_loss(
    T: torch.Tensor, a: torch.Tensor = None, b: torch.Tensor = None
):
    """
    Compute the marginal loss:
        loss = KL(a || T1) + KL(b || T^T1)
    """
    if a is None:
        a = torch.ones(T.shape[0], device=T.device) / T.shape[0]
    if b is None:
        b = torch.ones(T.shape[1], device=T.device) / T.shape[1]
    Ta = T.sum(1) + 1e-8  # Sum over columns
    Tb = T.sum(0) + 1e-8  # Sum over rows
    loss_a = torch.sum(a * (torch.log(a) - torch.log(Ta)) - a + Ta)
    loss_b = torch.sum(b * (torch.log(b) - torch.log(Tb)) - b + Tb)
    marginal_loss = loss_a + loss_b
    return marginal_loss


def sinkhorn(M, epsilon, a=None, b=None, max_iter=10000, tol=1e-5, symmetric=True):

    K = -M / epsilon
    n, m = K.shape

    if a is None:
        a = torch.ones(n, device=K.device) / n
    if b is None:
        b = torch.ones(m, device=K.device) / m

    u = torch.zeros(
        (n), dtype=K.dtype, device=K.device
    )  # u = torch.log(a) - torch.logsumexp(K, dim=2).squeeze()
    v = torch.zeros(
        (m), dtype=K.dtype, device=K.device
    )  # v = torch.log(b) - torch.logsumexp(K, dim=1).squeeze()

    for n_iters in range(max_iter):
        u = torch.log(a) - torch.logsumexp(K + v[None, :], dim=1).squeeze()
        v = torch.log(b) - torch.logsumexp(K + u[:, None], dim=0).squeeze()

        # Check convergence once every 10 iterations
        if n_iters % 10 == 0:
            T = torch.exp(K + u[:, None] + v[None, :])
            marginal = torch.sum(T, dim=1)
            err = torch.max(torch.abs(marginal - a))
            if err < tol:
                break

    if (
        symmetric
    ):  # Make it more symmetric, no marginals are exactly satisfied, both are approximately satisfied.
        u_extra = torch.log(a) - torch.logsumexp(K + v[None, :], dim=1).squeeze()
        u = 0.5 * (u + u_extra)
    log_T = K + u[:, None] + v[None, :]  # Marginals on the left are correct
    T = torch.exp(log_T)

    return {"plan": T, "log_plan": log_T, "n_iters": n_iters}


def top_k_accuracy(C, k):
    """
    Assume a batch with c pairs (fully supervised)
    C of shape (n,n) the distance matrix in the latent space.
    Return the top-k accuracy, i.e. the proportion of samples
    that have their true match in their k nearest neighbors.
    """
    target = torch.arange(C.shape[0], device=C.device)  # The target is the diagonal
    preds = C.topk(k, dim=1, largest=False).indices  # Shape (n,k)
    correct = (preds == target[:, None]).any(dim=1).float()  # Shape (n,)
    return correct


def accuracy(T):
    """
    Assume a batch with c pairs (fully supervised)
    T of shape (n,n) the transport plan in the latent space.
    Return the accuracy, i.e. the proportion of samples
    that are matched to their true match.
    """
    target = torch.arange(T.shape[0], device=T.device)  # The target is the diagonal
    preds = T.argmax(dim=1)  # Shape (n,)
    correct = (preds == target).float()  # Shape (n,)
    return correct


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8
    C1 = torch.rand((n, n), device=device)
    C2 = torch.rand((n, n), device=device)
    T = torch.rand((n, n), device=device)
    T = T / T.sum()

    loss_true = torch.empty((n, n, n, n), device=device)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    loss_true[i, j, k, l] = (
                        (C1[i, j] - C2[k, l]) ** 2 * T[i, k] * T[j, l]
                    )
    loss_true = loss_true.sum()
    print("True quad loss:", loss_true.item())
    loss = basic_quad_loss(C1, C2, T)
    print("Computed quad loss:", loss.item())

    M = torch.rand((n, n), device=device)
    loss = basic_lin_loss(M, T)
    print("Lin loss:", loss.item())

    a = torch.ones(n, device=device) / n
    b = torch.ones(n, device=device) / n
    loss = basic_marginal_loss(a, b, T)
    print("Marginal loss:", loss.item())
