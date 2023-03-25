from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    EXPECTATION = 10
    VARIANCE = 1
    FULL_SAMPLE_SIZE = 1000

    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(EXPECTATION, VARIANCE, FULL_SAMPLE_SIZE)
    univariate_gaussian_q1 = UnivariateGaussian()
    univariate_gaussian_q1.fit(samples)
    print((np.round(univariate_gaussian_q1.mu_, 3), np.round(univariate_gaussian_q1.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent
    univariate_gaussian_q2 = UnivariateGaussian()
    sample_size_expectations = np.array([
        (sample_size, np.abs(EXPECTATION - univariate_gaussian_q2.fit(samples[:sample_size]).mu_))
        for sample_size in range(10, FULL_SAMPLE_SIZE + 1, 10)
    ]).transpose()

    figure_q2 = go.Figure(
        go.Scatter(x=sample_size_expectations[0],
                   y=sample_size_expectations[1],
                   mode='markers')
    )
    figure_q2.update_layout(width=950, title="Delta of Sample Mean Expected Value, Based on Sample Size",
                            xaxis_title='Sample Size',
                            yaxis_title='Sample Mean Expected Value Delta')
    figure_q2.write_image("univariate_sample_size_to_expectation_diff_q2.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    sample_value_to_pdf = np.column_stack([samples, univariate_gaussian_q1.pdf(samples)])

    # Only ordering because it sounded like this would be required in the question.
    # In reality this makes no difference to the plot itself.
    sample_value_to_pdf = sample_value_to_pdf[sample_value_to_pdf[:, 0].argsort()]

    figure_q3 = go.Figure(
        go.Scatter(x=sample_value_to_pdf[:, 0], y=sample_value_to_pdf[:, 1], mode='markers')
    )
    figure_q3.update_layout(
        width=950,
        title=r"$\text{PDFs of N(%s, %s) Distributed Samples}$" % (EXPECTATION, VARIANCE),
        xaxis_title='Sample Value',
        yaxis_title='PDF'
    )
    figure_q3.write_image("univariate_sample_to_pdf_q3.png")


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    SAMPLE_SIZE = 1000
    mean = np.array([0, 0, 4, 0]).transpose()
    cov = np.array([
        [1, 0.2, 0, 0.5],
        [0.2, 2, 0, 0],
        [0, 0, 1, 0],
        [0.5, 0, 0, 1]
    ])
    samples = np.random.multivariate_normal(mean, cov, SAMPLE_SIZE)
    multivariate_gaussian_q1 = MultivariateGaussian()
    multivariate_gaussian_q1.fit(samples)
    print(f"Estimated Expectation: {multivariate_gaussian_q1.mu_}",
          f"Covariance Matrix: \n{multivariate_gaussian_q1.cov_}", sep='\n')

    # Question 5 - Likelihood evaluation
    f1_values = np.linspace(-10, 10, 200)
    f3_values = np.linspace(-10, 10, 200)

    # Create 2d grid of f1 and f3 values
    f1_values_matrix, f3_values_matrix = np.meshgrid(f1_values, f3_values)

    # Create an array where each entry is a mean value to be tested
    mean_values = np.column_stack(
        [f1_values_matrix.ravel(), np.zeros(f1_values_matrix.size),
         f3_values_matrix.ravel(), np.zeros(f3_values_matrix.size)])

    z_values = np.apply_along_axis(MultivariateGaussian.log_likelihood, axis=1, arr=mean_values,
                                   cov=cov, X=samples)

    figure = go.Figure(
        go.Heatmap(x=f3_values_matrix.ravel(), y=f1_values_matrix.ravel(), z=z_values)
    )
    figure.update_layout(
        title=r"$\text{Log Likelihood of Multivariate Gaussian By Expectation: }(f_1, 0, f_3, 0)$",
        xaxis_title="$f_3$", yaxis_title="$f_1$")
    figure.write_image("multivariate_log_likelihood_by_expectation_f1f3_q5.png")

    # Question 6 - Maximum likelihood
    max_z_index = np.argmax(z_values)
    max_indexes = np.unravel_index(max_z_index, f1_values_matrix.shape)
    print(f"f1={np.round(f1_values_matrix[max_indexes], 3)}, "
          f"f3={np.round(f3_values_matrix[max_indexes], 3)}, "
          f"log_likelihood={np.round(z_values[max_z_index], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
