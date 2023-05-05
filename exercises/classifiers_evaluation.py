import os

import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    from IMLearn.metrics.loss_functions import misclassification_error

    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(os.path.join("../datasets/", f))

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_recorder(perceptron: Perceptron, sample: np.ndarray, response: int):
            # Record loss for all except the last iteration (where sample and response return as null),
            # to avoid double loss collection.
            if sample is not None and response is not None:
                losses.append(perceptron.loss(X, y))

        Perceptron(callback=loss_recorder).fit(X, y)

        # Plot figure of loss as function of fitting iteration
        figure = go.Figure(
            [
                go.Scatter(x=np.array(range(1, len(losses) + 1)), y=np.array(losses))
            ]
        )
        figure.update_layout(xaxis_title='Iteration', yaxis_title='Misclassification Error',
                             title=f"Perceptron Misclassification Error on {n} data")
        figure.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    class_color_palette = np.array(px.colors.DEFAULT_PLOTLY_COLORS)

    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(os.path.join('../datasets', f))

        # Fit models and predict over training set
        lda = LDA().fit(X, y)
        lda_prediction = lda.predict(X)
        naive_bayes = GaussianNaiveBayes().fit(X, y)
        naive_bayes_prediction = naive_bayes.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        figure = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"Gaussian Naive Bayes (accuracy: {accuracy(y, naive_bayes_prediction)})",
                            f"LDA (accuracy: {accuracy(y, lda_prediction)})"])

        # Create covariance matrices for each class variance (will be a diagonal matrix)
        naive_bayes_covs = [np.identity(len(vars_k)) * vars_k
                            for mu_k, vars_k in zip(naive_bayes.mu_, naive_bayes.vars_)]

        # To reduce code duplication, I require each model to supply its prediction, mu (for each class) and cov
        # (also for each class).
        for idx, (prediction, mu, covs) in enumerate([
            (naive_bayes_prediction, naive_bayes.mu_, naive_bayes_covs),
            (lda_prediction, lda.mu_, [lda.cov_] * len(lda.mu_))
        ]):

            # Add traces for data-points setting symbols and colors
            figure.add_trace(
                go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                           marker=go.scatter.Marker(color=class_color_palette[prediction],
                                                    symbol=y)),
                row=1, col=idx + 1
            )

            # Add `X` dots specifying fitted Gaussians' means
            figure.add_trace(
                go.Scatter(x=mu[:, 0], y=mu[:, 1], mode='markers',
                           marker=go.scatter.Marker(color="black", symbol="x", size=18)), row=1, col=idx + 1
            )

            # Add ellipses depicting the covariances of the fitted Gaussians
            for k_idx, cov_k in enumerate(covs):
                figure.add_trace(get_ellipse(mu[k_idx], cov_k), row=1, col=idx + 1)

        figure.update_layout(title=f'Dataset: {f} predictions', showlegend=False)
        figure.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
