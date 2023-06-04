import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn, Optional, Iterable, Tuple, List

from ..metrics import misclassification_error


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations

    self.weights_: List[float]
        List of weights for each fitted estimator, fitted along the boosting iterations

    self.D_: List[np.ndarray]
        List of weights for each sample, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_: Optional[List[BaseEstimator]] = None
        self.weights_: Optional[List[float]] = None
        self.D_: Optional[List[float]] = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        num_of_samples = len(X)

        # Reset boosting values
        self.weights_ = []
        self.models_ = []
        # Set first distribution to be (1/num_of_samples)
        self.D_ = [np.ones((num_of_samples,)) / num_of_samples]

        for t in range(self.iterations_):
            # Pick samples by their distributions (with replacing/choosing same sample twice)
            weighted_samples_idxs = np.random.choice(num_of_samples, size=num_of_samples, p=self.D_[-1], replace=True)
            weighted_samples = X[weighted_samples_idxs, :]
            weighted_samples_y = y[weighted_samples_idxs]

            # Fit the model on the selected samples, and predict on the original data
            self.models_.append(self.wl_().fit(weighted_samples, weighted_samples_y))
            y_prediction = self.models_[-1].predict(X)

            # Compute the weight of the learner
            epsilon_t = self.D_[-1][y != y_prediction].sum()  # Total weight of mistakes
            self.weights_.append(0.5 * np.log((1 - epsilon_t) / epsilon_t))

            # Update the distributions
            self.D_.append(
                self.D_[-1] * np.exp(-y * self.weights_[-1] * y_prediction)
            )

            # Normalize last distributions (so each one is between 0 and 1)
            self.D_[-1] /= np.sum(self.D_[-1])

    def _predict(self, X):
        """
        Predict responses for given samples using fitted estimator over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Call partial predict with all the models
        return self.partial_predict(X, len(self.models_))

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under mis-classification loss function over all boosting iterations

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under mis-classification loss function
        """
        # Call partial loss with entire set of models
        return self.partial_loss(X, y, len(self.models_))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        # Create a 2d matrix of size (n_samples, t_model_prediction)
        predictions = np.array([model.predict(X)
                                for model in self.models_[:T]]
                               ).T

        weighted_sum_of_predictions = predictions @ self.weights_[:T]
        return np.sign(weighted_sum_of_predictions)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function using fitted estimators up to T learners

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
