from scipy.optimize import linear_sum_assignment
import torch


def plan_to_permutation(plan):
    r"""Convert a transport plan to a permutation.

    Parameters
    ----------
    plan : array-like, shape (n, n)
        Transport plan

    Returns
    -------
    permutation : array-like, shape (n)
        Permutation indices for each batch element
    """
    _, permutation = linear_sum_assignment(-plan.T)
    return permutation


def permutation_to_plan(permutation, device="cpu"):
    r"""Convert a permutation to a transport plan.

    Parameters
    ----------
    permutation : array-like, shape (n)
        Permutation indices
    n : int
        Size of the permutation

    Returns
    -------
    plan : array-like, shape (n, n)
        Transport plan corresponding to the permutation
    """
    n = len(permutation)
    plan = torch.eye(n, device=device)[:, permutation]
    return plan


def discretize_plan(plan):
    r"""Discretize a transport plan to a permutation.

    Parameters
    ----------
    plan : array-like, shape (n, n)
        Transport plan

    Returns
    -------
    permutation : array-like, shape (n)
        Permutation indices for each batch element
    """
    permutation = plan_to_permutation(plan)
    return permutation_to_plan(permutation, device=plan.device)


def cosine_similarity_matrix(X, Y):
    r"""Compute the cosine similarity matrix between two sets of vectors.

    Parameters
    ----------
    X : array-like, shape (n, d)
        First set of vectors
    Y : array-like, shape (m, d)
        Second set of vectors

    Returns
    -------
    M : array-like, shape (n, m)
        Cosine similarity matrix
    """
    X_norm = X / torch.max(
        X.norm(dim=1, keepdim=True), torch.tensor(1e-8, device=X.device)
    )
    Y_norm = Y / torch.max(
        Y.norm(dim=1, keepdim=True), torch.tensor(1e-8, device=Y.device)
    )
    return torch.mm(X_norm, Y_norm.T)


def log_softmax_row_col(A):
    r"""
    Return Geometric Average of Row and Column Softmax of A.
    Compared to the geometric average this enables to computes use the log-sum-exp trick.

    output = sqrt(softmax(A, dim=1) * softmax(A, dim=0))
    i.e.
    log_output_ij = = A_ij - 0.5 * (logsumexp(A, dim=1)_i + logsumexp(A, dim=0)_j)
    """
    log_output_ij = A - 0.5 * (
        torch.logsumexp(A, dim=1, keepdim=True)
        + torch.logsumexp(A, dim=0, keepdim=True)
    )
    return log_output_ij


if __name__ == "__main__":
    import torch

    # Example usage
    plan = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    print("Original plan:\n", plan)
    permutation = plan_to_permutation(plan)
    print("Permutation indices:", permutation)
    discrete_plan = permutation_to_plan(permutation, device=plan.device)
    print("Discrete plan:\n", discrete_plan)
