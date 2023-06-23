import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    _values = []
    _weights = []

    def recorder(solver=None, weights=None, val=None, grad=None, t=None, eta=None, delta=None):
        _values.append(val)
        _weights.append(weights)

    return recorder, _values, _weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for module in [L1, L2]:
        minimum_loss = None
        minimum_loss_eta = None
        for eta in etas:
            callback, values, weights = get_gd_state_recorder_callback()

            gd_best_weights = GradientDescent(FixedLR(eta), out_type='best', callback=callback).fit(module(init.copy()),
                                                                                                    None,
                                                                                                    None)
            # Update minimum loss
            best_loss_for_eta = module(gd_best_weights).compute_output()
            if minimum_loss is None or best_loss_for_eta < minimum_loss:
                minimum_loss = best_loss_for_eta
                minimum_loss_eta = eta

            # Plot descent path
            weights_array = np.array(weights)
            max_xy = np.abs(weights_array).max() * 1.2
            ranges = (-max_xy, max_xy)
            figure_1 = plot_descent_path(module, weights_array,
                                         f'{module.__name__} with fixed learning rate {eta}',
                                         xrange=ranges, yrange=ranges)
            figure_1.show()

            # Plot convergence rate (norm as function of iteration
            figure_2 = go.Figure(
                go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode='markers+lines')
            )
            figure_2.update_layout(title=f"Convergence of GD: {module.__name__} norm, learning rate: {eta}",
                                   xaxis_title="GD Iteration", yaxis_title="Norm value")
            figure_2.show()

        print(f"{module.__name__} minimum value: {minimum_loss}, under learning rate: {minimum_loss_eta}.")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Plotting ROC curve
    fitted_model = LogisticRegression().fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_test, fitted_model.predict_proba(X_test))
    figure_1 = go.Figure(
        go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                   hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")
    )
    figure_1.update_layout(title="ROC Curve of Logistic Regression trained with GD",
                           xaxis_title="False Positive Rate (FPR)", yaxis_title="True Positive Rate (TPR)")
    figure_1.show()

    # Finding best alpha value and test error
    fitted_model.alpha_ = thresholds[np.argmax(tpr - fpr)]
    print(f"Optimal Alpha value: {fitted_model.alpha_}, "
          f"Loss: {fitted_model.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    # Search for correct lambda
    gd_model = GradientDescent(tol=1e-4, max_iter=20000)
    v_cross_validate = np.vectorize(cross_validate, excluded=['X', 'y'])
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for penalty in ['l1', 'l2']:
        models = [
            LogisticRegression(penalty=penalty, solver=gd_model, lam=lambda_param)
            for lambda_param in lambdas
        ]

        _, validation_errors = v_cross_validate(models, X=X_train, y=y_train,
                                                scoring=misclassification_error)
        best_lambda = lambdas[np.argmin(validation_errors)]
        test_loss = LogisticRegression(penalty=penalty, solver=gd_model, lam=best_lambda) \
            .fit(X_train, y_train) \
            .loss(X_test, y_test)
        print(f"Regularization Penalty: {penalty.upper()}, Best lambda: {best_lambda}, Loss on test: {test_loss}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
