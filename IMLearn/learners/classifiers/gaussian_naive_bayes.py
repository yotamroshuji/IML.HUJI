from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        sample_count = X.shape[0]

        # Save the classes in a specific order (just to be civil)
        label_counts = dict(sorted(zip(*np.unique(y, return_counts=True)), key=lambda x: x[0]))

        self.classes_ = np.array(label_counts.keys())
        self.pi_ = np.array([count / sample_count for _, count in label_counts.items()])
        self.mu_ = np.array([
            # Take each row in X that matches the correct label and sum the columns (features)
            (1 / label_count) * X[y == label].sum(axis=1)
            for label, label_count in label_counts.items()
        ])  # This way, each row is a label, and each column is a feature

        self.vars_ = np.array([
            (1 / label_count) * np.sum((X[y == label] - self.mu_[label_idx, :]) ** 2, axis=1)
            for label_idx, (label, label_count) in enumerate(label_counts.items())
        ])

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
        # Build a "covariance" matrices - will just be a variance matrix for with all variances on the diagonal
        variance_matrices = [np.identity(len(self.vars_)) @ class_variances for class_variances in self.vars_]

        # Use the normal Bayes Optimal classifier prediction shown in claim 3.5.2
        a_k__b_k = np.array(
            [
                (
                    (variance_matrix @ mu_k),
                    np.log(pi_k) - 0.5 * mu_k.T @ variance_matrix @ mu_k
                )
                for mu_k, pi_k, variance_matrix in zip(self.mu_, self.pi_, variance_matrices)
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
         # TODO: this (got too tired)
        np.array([
            np.product((1 / np.sqrt(2 * np.pi * var_k)) * np.exp(-0.5 * (X - mu_k) ** 2 / var_k), axis=1)
            for mu_k, var_k in zip(self.mu_, self.vars_)
        ])

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
