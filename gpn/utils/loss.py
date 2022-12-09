import torch
import numpy as np

import torch
import torch.nn as nn
import scipy.linalg as slinalg


class EigValsH(torch.autograd.Function):
    """ Solving the generalized eigenvalue problem A x = lambda B x

    Gradients of this function is customized.

    Parameters
    ----------
    A: Tensor
        Left-side matrix with shape [D, D]
    B: Tensor
        Right-side matrix with shape [D, D]

    Returns
    -------
    w: Tensor
        Eigenvalues, with shape [D]

    Reference:
    https://github.com/Theano/theano/blob/d395439aec5a6ddde8ef5c266fd976412a5c5695/theano/tensor/slinalg.py#L385-L440

    """
    @staticmethod
    def forward(ctx, *args, **kwargs):
        A, B = args
        device = A.device
        A = A.detach().data.cpu().numpy()
        B = B.detach().data.cpu().numpy()
        w, v = slinalg.eigh(A, B)
        w = torch.from_numpy(w).to(device)
        v = torch.from_numpy(v).to(device)
        ctx.save_for_backward(w, v)
        return w

    @staticmethod
    def backward(ctx, *grad_outputs):
        w, v = ctx.saved_tensors
        dw = grad_outputs[0]
        gA = torch.matmul(v, torch.matmul(torch.diag(dw), torch.transpose(v, 0, 1)))
        gB = -torch.matmul(v, torch.matmul(torch.diag(dw * w), torch.transpose(v, 0, 1)))
        return gA, gB


def eigh(A, B):
    device = A.device
    A = A.detach().data.cpu().numpy()
    B = B.detach().data.cpu().numpy()
    w, v = slinalg.eigh(A, B)
    w = torch.from_numpy(w).to(device)
    v = torch.from_numpy(v).to(device)
    return w, v


def linear_discriminative_eigvals(y, X, lambda_val=1e-3, ret_vecs=False):
    """
    Compute the linear discriminative eigenvalues

    Usage:

    >>> y = [0, 0, 1, 1]
    >>> X = [[1, -2], [-3, 2], [1, 1.4], [-3.5, 1]]
    >>> eigvals = linear_discriminative_eigvals(y, X, 2)
    >>> eigvals.numpy()
    [-0.33328852 -0.17815116]

    Parameters
    ----------
    y: Tensor, np.ndarray
        Ground truth values, with shape [N, 1]
    X: Tensor, np.ndarray
        The predicted values (i.e., features), with shape [N, d].
    lambda_val: float
        Lambda for stablizing the right-side matrix of the generalized eigenvalue problem
    ret_vecs: bool
        Return eigenvectors or not.
        **Notice:** If False, only eigenvalues are returned and this function supports
        backpropagation (used for training); If True, both eigenvalues and eigenvectors
        are returned but the backpropagation is undefined (used for validation).

    Returns
    -------
    eigvals: Tensor
        Linear discriminative eigenvalues, with shape [cls]

    References:
    Dorfer M, Kelz R, Widmer G. Deep linear discriminant analysis[J]. arXiv preprint arXiv:1511.04707, 2015.

    """
    classes = torch.unique(y, sorted=True)

    def compute_cov(i):
        # Hypothesis: equal number of samples (Ni) for each class
        Xg = X[y == i]                                                                  # [None, d]
        Xg_bar = Xg - torch.mean(Xg, dim=0, keepdim=True)                               # [None, d]
        m = float(Xg_bar.shape[0])                                                     # []
        return (1. / (m - 1)) * torch.sum(
            Xg_bar.unsqueeze(dim=1) * Xg_bar.unsqueeze(dim=2), dim=0)                   # [d, d]

    # convariance matrixs for all the classes
    covs = []
    for c in classes:
        covs.append(compute_cov(c))
    # Within-class scatter matrix
    Sw = sum(covs) / len(covs)                                                          # [d, d]

    # Total scatter matrix
    X_bar = X - torch.mean(X, dim=0, keepdim=True)                                      # [N, d]
    m = float(X_bar.shape[0])                                                          # []
    St = (1. / (m - 1)) * torch.sum(
        X_bar.unsqueeze(dim=1) * X_bar.unsqueeze(dim=2), dim=0)                         # [d, d]

    # Between-class scatter matrix
    Sb = St - Sw                                                                        # [d, d]

    # Force Sw_t to be positive-definite (for numerical stability)
    Sw = Sw + torch.eye(Sw.shape[0]).to(Sw.device) * lambda_val  # [d, d]

    # Solve the generalized eigenvalue problem: Sb * W = lambda * Sw * W
    # We use the customed `eigh` function for generalized eigenvalue problem
    if ret_vecs:
        return eigh(Sb, Sw)                                                             # [cls], [d, cls]
    else:
        return EigValsH.apply(Sb, Sw)                                                   # [cls]


def linear_discriminative_loss(y, X, lambda_val=1e-3):
    """
    Compute the linear discriminative loss

    Usage:

    >>> y = torch.from_numpy(np.array([0, 0, 1, 1]))
    >>> X = torch.from_numpy(np.array([[1, -2], [-3, 2], [1, 1.4], [-3.5, 1]]))
    >>> X.requires_grad = True
    >>> loss_obj = LinearDiscriminativeLoss()
    >>> loss = loss_obj(X, y)
    >>> loss.backward()
    >>> print(loss)
    tensor(0.1782, dtype=torch.float64, grad_fn=<NegBackward>)
    >>> print(X.grad)
    tensor([[ 0.0198,  0.0608],
            [ 0.0704,  0.2164],
            [-0.0276, -0.0848],
            [-0.0626, -0.1924]], dtype=torch.float64)

    Parameters
    ----------
    y: Tensor, np.ndarray
        Ground truth values, with shape [N, 1]
    X: Tensor, np.ndarray
        The predicted values (i.e., features), with shape [N, d].
    lambda_val: float
        Lambda for stablizing the right-side matrix of the generalized eigenvalue problem

    Returns
    -------
    costs: Tensor
        Linear discriminative loss value, with shape [bs]

    References:
    Dorfer M, Kelz R, Widmer G. Deep linear discriminant analysis[J]. arXiv preprint arXiv:1511.04707, 2015.

    """
    eigvals = linear_discriminative_eigvals(y, X, lambda_val)                           # [cls]

    # At most cls - 1 non-zero eigenvalues
    classes = torch.unique(y, sorted=True)                                              # [cls]
    cls = classes.shape[0]
    eigvals = eigvals[-cls + 1:]                                                        # [cls - 1]
    thresh = torch.min(eigvals) + 1.0                                                   # []

    # maximize variance between classes
    top_k_eigvals = eigvals[eigvals <= thresh]                                          # [None]
    costs = -top_k_eigvals                                                              # []
    return costs


class LinearDiscriminativeLoss(nn.Module):
    """

    Parameters
    ----------
    num_classes: int
        Number of classes
    lambda_val: float
        Lambda for stablizing the right-side matrix of the generalized eigenvalue problem
    reduction: tf.keras.losses.Reduction
        (Optional) Applied to loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases this defaults to
        `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of built-in
        training loops such as `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial]
        (https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
    name: str
        (Optional) Name for the op. Defaults to 'sparse_categorical_crossentropy'.
    """
    def __init__(self,
                 lambda_val=1e-3,
                 name="linear_discriminative_analysis"):
        super(LinearDiscriminativeLoss, self).__init__()
        self.lambda_value = lambda_val

    def forward(self, input, target):
        return linear_discriminative_loss(target, input, lambda_val=self.lambda_value)

class LDA(object):
    def __init__(self, lambda_val=1e-3, n_components=None, verbose=False, logger=None):
        self.lambda_val = lambda_val
        self.n_components = n_components
        self.verbose = verbose
        self.logger = logger.info if logger is not None else print

    def fit(self, X, y):
        # Assert X is float32, y is int32
        classes = torch.unique(y)

        if self.n_components is None:
            self.n_components = classes.shape[0] - 1

        means = []
        for i in classes:
            Xg = X[y == i]
            means.append(torch.mean(Xg, dim=0))
        self.means = torch.stack(means, dim=0)                                          # [cls, d]

        eigvals, eigvecs = linear_discriminative_eigvals(y, X, self.lambda_val, ret_vecs=True)
        eigvecs = eigvecs.flip(dims=(1,))                                               # [d, cls]
        eigvecs = eigvecs / torch.norm(eigvecs, dim=0, keepdim=True)                    # [d, cls]
        self.scaling = eigvecs.detach().data.cpu().numpy()
        self.coef = torch.matmul(
            torch.matmul(self.means, eigvecs), torch.transpose(eigvecs, 0, 1))          # [cls, d]
        self.intercept = -0.5 * torch.diag(
            torch.matmul(self.means, torch.transpose(self.coef, 0, 1)))                 # [cls]
        self.coef = self.coef.detach().data.cpu().numpy()
        self.intercept = self.intercept.detach().data.cpu().numpy()

        eigvals = eigvals.detach().data.cpu().numpy()
        if self.verbose:
            top_k_evals = eigvals[-self.n_components + 1:]
            self.logger("\nLDA-Eigenvalues:", np.array_str(top_k_evals, precision=2, suppress_small=True))
            self.logger("Eigenvalues Ratio min/max: %.3f, Mean: %.3f" % (
                top_k_evals.min() / top_k_evals.max(), top_k_evals.mean()))

        return self

    def prob(self, X):
        prob = np.dot(X, self.coef.T) + self.intercept                                  # [N, cls]
        # prob_sigmoid = 1. / (np.exp(prob) + 1)                                        # [N, cls]
        # sigmoid = prob_sigmoid / np.sum(prob_sigmoid, axis=1, keepdims=True)          # [N, cls]
        # return sigmoid
        return prob

    def pred(self, Xt):
        return np.argmax(self.prob(Xt), axis=1)                                          # [N]

    def test(self, X, y):
        pred = self.pred(X)
        return np.sum(pred == y) / len(pred)

    def map(self, X):
        X_new = np.dot(X, self.scaling)                                                 # [N, cls]
        return X_new[:, :self.n_components]                                             # [N, cls - 1]

