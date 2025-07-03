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

def calculate_HWHM(mean_intensities, radii_microns):
    import scipy

    # Step 1: Find peak intensity and half-maximum value
    peak_intensity = np.max(mean_intensities)
    half_max = peak_intensity / 2

    # Step 2: Interpolate to find radii at the half-max level on each side of the peak
    interp_func = scipy.interpolate.interp1d(mean_intensities, radii_microns)
    half_max_radius_left = interp_func(half_max)  # Radius on the left side
    half_max_radius_right = interp_func(half_max)  # Radius on the right side

    # Step 3: Calculate HWHM (half-width is the distance from the peak to one side)
    peak_radius = radii_microns[np.argmax(mean_intensities)]
    hwhm = abs(half_max_radius_right - peak_radius)

    return hwhm

def weighted_avg_and_ci(data, weights=None, ci=0.95):
    """
    Calculate weighted mean and confidence interval (or std) across trials (axis=0).
    data shape: (n_trials, n_timepoints)
    weights shape: (n_trials,)
    """
    from scipy.stats import t
    
    data = np.array(data)
    if weights is None:
        weights = np.ones(data.shape[0])
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # normalize

    # Weighted mean across trials
    mean = np.average(data, axis=0, weights=weights)

    # Effective sample size
    n_eff = (np.sum(weights))**2 / np.sum(weights**2)

    # Weighted sample variance across trials
    variance = np.average((data - mean)**2, axis=0, weights=weights)
    std = np.sqrt(variance)

    # Use either CI or std
    t_score = t.ppf(1 - (1 - ci) / 2, df=n_eff - 1)
    se = std / np.sqrt(n_eff)
    lower = mean - t_score * se
    upper = mean + t_score * se

    return mean, lower, upper