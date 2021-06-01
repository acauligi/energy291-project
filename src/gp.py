import gurobipy
import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import sklearn.gaussian_process as gp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

class GaussianProcess():
    def __init__(self, kernel=None, length_scale=20.0):
        self.kernel = kernel
        if not kernel:
            self.kernel = 1.0 * RBF(length_scale=length_scale)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    def train(self, Ts, data):
        self.gp.fit(Ts, data)

    def predict(self, Ts, ε=1e3):
        self.mean, self.std = self.gp.predict(Ts, return_std=True)
        Σ = np.diag(self.std).dot(np.diag(self.std).T) + np.eye(self.mean.size)
        self.L = np.linalg.cholesky(Σ) + ε*np.eye(self.mean.size)

    def sample(self):
        sample = self.mean.flatten() + self.L.dot(np.random.normal(size=(self.mean.size)))
        return sample
