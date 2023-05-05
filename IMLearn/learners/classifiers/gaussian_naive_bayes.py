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

        self.classes_ = np.array(list(label_counts.keys()))
        self.pi_ = np.array([count / sample_count for _, count in label_counts.items()])
        self.mu_ = np.array([X[y == label].mean(axis=0) for label in self.classes_])
        # This way, each row is a label, and each column is a feature

        self.vars_ = np.array([
            (1 / label_count) * np.sum((X[y == label] - self.mu_[label_idx, :]) ** 2, axis=0)
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
        # Just need to maximize likelihood
        maximizing_k_index = np.argmax(self.likelihood(X), axis=1)
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

        return np.array([
            np.product(
                (1 / np.sqrt(2 * np.pi * var_k)) * np.exp(-0.5 * (X - mu_k) ** 2 / var_k),
                axis=1) * pi_k
            for mu_k, var_k, pi_k in zip(self.mu_, self.vars_, self.pi_)
        ]).T

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
