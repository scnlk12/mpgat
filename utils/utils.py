import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import copy
import scipy.sparse as sp

# log string
def log_string(log, string):
    if log is not None:
        log.write(string + '\n')
        log.flush()
    print(string)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def cal_lape(adj_mx, lape_dim):
    # lape_dim = 8
    # lape_dim = 64
    L, isolated_point_num = calculate_normalized_laplacian(adj_mx)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    laplacian_pe = torch.from_numpy(EigVec[:, isolated_point_num + 1: lape_dim + isolated_point_num + 1]).float()
    laplacian_pe.require_grad = False
    return laplacian_pe, L

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    isolated_point_num = np.sum(np.where(d, 0, 1))
    # self._logger.info(f"Number of isolated points: {isolated_point_num}")
    # print(f"Number of isolated points: {isolated_point_num}")

    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian, isolated_point_num

def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list[np.ndarray], length: K, from T_0 to T_{K-1}

    '''

    L_tilde_dense = L_tilde.toarray()
    N = L_tilde_dense.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde_dense]

    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde_dense * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean