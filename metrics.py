import numpy as np


def mae(y_hat: np.array, y: np.array):
    return np.mean(np.abs(y_hat - y))


def r2(y_hat: np.array, y: np.array):
    return 1 - np.mean((y - y_hat) ** 2) / np.mean((y - np.mean(y)) ** 2)
