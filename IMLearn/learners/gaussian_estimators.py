from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        if self.biased_:
            # To keep the estimator biased, use an expected value that does not change with the number of samples,
            # and a variance that does not subtract 1 from the degrees of freedom (we said this is biased)
            self.var_ = np.var(X, ddof=0)

        else:
            # Use the sample mean and sample variance
            self.var_ = np.var(X, ddof=1)

        self.mu_ = np.mean(X)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        # Using the basic density function for a Gaussian distributed random variable.
        return (1 / np.sqrt(2 * np.pi * self.var_)) * np.exp(-np.power(X - self.mu_, 2) / (2 * self.var_))

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # Implementing the calculation in Definition 1.1.11 in the school book as the uni-variate Gaussian pdf
        # A cleaner alternative is using the scipy.stats package, to do:
        #  np.sum(np.log(norm(loc=mu, scale=sigma).pdf(arr)))
        exp_power = -(1 / 2 * sigma) * np.sum(np.power(X - mu, 2))
        exp_divisor = np.power(2 * np.pi * sigma, X.size / 2)
        return np.log(np.exp(exp_power) / exp_divisor)


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        m = X.shape[0]
        centered_X_matrix = X - self.mu_
        self.cov_ = (1 / (m - 1)) * (centered_X_matrix.T @ centered_X_matrix)

        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        d = len(X)
        centered_X = X - self.mu_
        denominator = np.sqrt(
            np.power(2 * np.pi, d) * det(self.cov_)
        )
        # Now we calculate the nominator for each row (entry) by itself, and add them on top of each other as a
        # vector.
        # This means we need to multiply each row in X (1 by d) by the inverse of the covariance matrix (d by d),
        # and then multiply the resulting vector (1 by d) with the column vector of each row in X (d by 1).
        # To achieve the last part, we will need to multiply each row in X @ inv(self.cov_), by the matching row
        # in X as a column vector! This is basically equal to pairwise multiplication, and then summing each row.

        nominator = np.exp(
            (-1 / 2) * np.sum(centered_X @ inv(self.cov_) * centered_X, axis=1)
        )
        return nominator / denominator

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        centered_X = X - mu
        sample_count, feature_count = X.shape

        # Calculate the sum section of the log_likelihood, using the same logic as "self.pdf" method.
        required_likelihood_sum = np.sum(centered_X @ inv(cov) * centered_X)
        return -0.5 * (
                required_likelihood_sum +
                sample_count * feature_count * np.log(2 * np.pi) +
                sample_count * slogdet(cov)[1]
        )


if __name__ == '__main__':
    # print(UnivariateGaussian.log_likelihood(1, 1, np.array([-1, 0, 0, 1])))
    # from scipy.stats import norm
    # arr = np.array([-1, 0, 0, 1], dtype=int)
    # print(arr)
    # print(norm.pdf(arr))
    # print(np.sum(np.log(norm(loc=1, scale=1).pdf(arr))))
    mg = MultivariateGaussian()
    arr = np.array(
        [
            [150, 45],
            [170, 74],
            [184, 79]
        ]
    )
    mg.fit(arr)
    print(mg.pdf(arr))
