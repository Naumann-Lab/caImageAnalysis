import numpy as np
import scipy.stats as stats


def calculate_ci(data, confidence_level = 0.95):
    """
    Calculate the 95% confidence interval for each dimension of a 2D array.

    Parameters:
    - data: 2D array containing the data
    - confidence_level: Confidence level for the interval (default is 0.95)

    Returns:
    - confidence_intervals: Tuple containing the confidence intervals for each dimension
    """

    # Calculate mean and standard error of the mean (SEM) along each axis
    means = np.mean(data, axis=0)
    sems = stats.sem(data, axis=0)

    # Set the degrees of freedom
    df = data.shape[0] - 1

    # Calculate the margin of error for each dimension
    margin_of_errors = stats.t.ppf((1 + confidence_level) / 2, df) * sems

    # Calculate the confidence interval for each dimension
    lower_bounds = means - margin_of_errors
    upper_bounds = means + margin_of_errors

    return lower_bounds, upper_bounds