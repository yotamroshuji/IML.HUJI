from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        sample_count = len(X)

        # Save the classes in a specific order (just to be civil)
        label_counts = dict(sorted(zip(*np.unique(y, return_counts=True)), key=lambda x: x[0]))
        label_order = {label: idx for idx, label in enumerate(label_counts)}

        self.classes_ = np.array(label_counts.keys())
        self.pi_ = np.array([label_count / sample_count for label_count in label_counts.values()])
        self.mu_ = np.array([X[y == label].mean(axis=0) for label in label_counts])

        # Create a matrix where each row in X has the corresponding mu value in the same row
        X_corr_mu_matrix = np.array([self.mu_[label_order[label]] for label in y])

        self.cov_ = (1 / sample_count) * (np.einsum('mi,mj->ij', X - X_corr_mu_matrix, X - X_corr_mu_matrix))
        self._cov_inv = inv(self.cov_)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Prediction is given using claim 3.5.2 from the book
        a_k__b_k = np.array(
            [
                (
                    (self._cov_inv @ mu_k),
                    np.log(pi_k) - 0.5 * mu_k.T @ self._cov_inv @ mu_k
                )
                for mu_k, pi_k in zip(self.mu_, self.pi_)
            ]
        )
        maximizing_k_index = np.argmax([a_k.T @ X + b_k for a_k, b_k in a_k__b_k])
        return self.classes_[maximizing_k_index]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        features_count = X.shape[1]

        # Calculate the likelihood for each sample under each class
        denominator = np.sqrt((np.pi ** features_count) * det(self.cov_))

        # This creates a matrix, where each row is a sample, and each column is the likelihood of that class
        # (in the order they appear in the class)
        return np.array(
            [
                # Using einsum because it's cool -> multiplies the first two matrices, and does by-element
                # multiplication of the resulting matrix and the last one, then sums all the cols in a row, leaving
                # a vector of size X.shape[0] (number of samples)
                pi_k * np.exp(-0.5 * np.einsum('ij,jk,ik->i', X - pi_k, self._cov_inv, X - pi_k))
                for mu_k, pi_k, in zip(self.mu_, self.pi_)
            ]) / denominator

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
