import numpy as np
from scipy.stats import f
from scipy.stats import t
import warnings


class LinearRegression:
    def _check_intercept(self, X: np.ndarray):
        columns_with_all_ones = np.all(X == 1, axis=0)
        all_ones_positions = np.where(columns_with_all_ones)[0]
        if len(all_ones_positions) == 1 and all_ones_positions[0] != 0:
            new_index = np.setdiff1d(range(X.shape[1]), all_ones_positions)
            new_index = np.concatenate([all_ones_positions, new_index])
            X = X[:, new_index]
        elif len(all_ones_positions) == 0:
            X = np.concatenate([np.ones(X.shape[0]).reshape(-1, 1), X], axis=1)
        elif len(all_ones_positions) > 1:
            warnings.warn(
                "Multiple columns with all ones detected. Only the first one will be kept as intercept."
            )
            new_index = np.setdiff1d(range(X.shape[1]), all_ones_positions)
            new_index = np.concatenate([[all_ones_positions[0]], new_index])
            X = X[:, new_index]
        else:
            pass

        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y
        self.X = self._check_intercept(self.X)
        self.dof = self.X.shape[0] - self.X.shape[1]
        self.XtXinv = np.linalg.inv(np.matmul(self.X.T, self.X))
        projection_matrix = np.matmul(self.XtXinv, self.X.T)
        self.beta_hat = np.matmul(projection_matrix, self.y)
        self.dof = self.X.shape[0] - self.X.shape[1]

    def predict(self, X):
        X = self._check_intercept(X)
        return np.matmul(X, self.beta_hat)

    def ftest(self, X: np.ndarray, y: np.ndarray, y_hat: np.ndarray):
        X = self._check_intercept(X)
        dof_full = X.shape[0] - X.shape[1]
        dof_reduced = X.shape[0] - 1
        sse_full = np.sum((y - y_hat) ** 2)
        sse_reduced = np.sum((y - np.mean(y)) ** 2)

        F = ((sse_reduced - sse_full) / (dof_reduced - dof_full)) / (
            sse_full / dof_full
        )
        p_value = 1 - f.cdf(F, dof_reduced - dof_full, dof_full)
        return p_value

    def t_test(self, residuals: np.ndarray, alpha: float = 0.05):
        c = (np.sum(residuals**2) / self.dof) * self.XtXinv
        se = np.sqrt(c.diagonal())
        lower_bound = t.ppf(alpha, self.dof)
        upper_bound = -t.ppf(alpha, self.dof)
        tvalues = self.beta_hat / se
        return ~((tvalues > lower_bound) & (tvalues < upper_bound))
