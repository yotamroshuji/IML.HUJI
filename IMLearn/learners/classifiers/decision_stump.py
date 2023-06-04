from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a decision stump to the given data. That is, finds the best feature and threshold by which to split

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        # Set this to 1.1, because 1 is the maximum error possible
        min_error = np.inf

        # For each feature, check the threshold for each sign.
        for feature_idx, sign in product(range(X.shape[1]), (-1, 1)):
            threshold, threshold_error = self._find_threshold(X[:, feature_idx], y, sign)

            # Choose the first feature with the minimal error (> and not >=)
            if min_error > threshold_error:
                min_error = threshold_error
                self.sign_ = sign
                self.j_ = feature_idx
                self.threshold_ = threshold

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sign responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Mis-classification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # After playing around with large datasets, I noticed this function takes forever to run
        # (used cProfile + snakeviz). Which is why I've made the code more complex to reduce time.

        # To avoid calling misclassification_error and comparing each prediction to the labels for each new threshold
        # (takes a lot of time),
        # I order the list of values, and update the threshold by order. This way, only the prediction for a single
        # value will change each time.

        # Sort both values and labels
        sorted_indexes = np.argsort(values)
        values = values[sorted_indexes]
        labels = labels[sorted_indexes]

        # Find number of mislabeled values for the initial threshold (the first value), where all predicted labels
        # are "sign".
        mislabeled = [np.sum(labels == sign)]

        # Go over each updated threshold, update the number of mislabeled values.
        # Example - Start with prediction [sign, sign, sign], say mislabeled = 2
        #           If we update the threshold, we will get [-sign, sign, sign]. If labels[0] == -sign, update
        #           mislabeled = 1, otherwise update mislabeled to be 3.

        for labeled_correctly in labels == -sign:
            mislabeled.append(mislabeled[-1] + (-1 if labeled_correctly else 1))

        # Finally return the least error, normalized between 0 and 1, with the correct threshold that caused it
        min_mislabeled_idx = np.argmin(mislabeled)
        thresholds = np.concatenate([[-np.inf], values[1:], [np.inf]])

        return thresholds[min_mislabeled_idx], mislabeled[min_mislabeled_idx] / len(labels)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under mis-classification loss function

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
        return misclassification_error(y, self.predict(X))
