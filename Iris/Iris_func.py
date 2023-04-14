import numpy as np

def sigmoid(x):
    return np.array(1 / (1 + np.exp(-x)))

def grad_W_MSE_func(g_k, t_k, x_k, D):
    A = (g_k - t_k)*g_k*(1-g_k)
    A = A.reshape(3, 1)
    B = x_k
    B = B.reshape(D+1, 1)
    return A @ B.T