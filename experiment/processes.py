# %% Processes
import numpy as np

def ma_process(theta:np.ndarray, n:int, m:int):
    Y = np.zeros((n,m))
    for n_idx in range(n):
        err = np.random.randn(m + len(theta) - 1)
        y = np.convolve(err, theta, mode='valid')
        Y[n_idx,:] = y
    C = np.zeros((m,m))
    for tau in range(len(theta)):
        gamma = 0
        for i in range(len(theta) - tau):
            gamma = gamma + theta[i]*theta[tau+i]
        C = C + np.diag([1]*(m-np.abs(tau)),tau)*gamma
        if tau != 0:
            C = C + np.diag([1]*(m-np.abs(tau)),-tau)*gamma
    return (Y,C)

def tvarma_process(A:np.ndarray, B:np.ndarray, n:int):
    d = B.shape[1]
    w = np.random.randn(d,n)
    Y = np.linalg.inv(np.eye(d) - A)
    X = (Y@B@w).transpose()
    C = Y@B@B.transpose()@Y.transpose()
    return (X,C)