import numpy as np


def pretty(x, n=3):
    """
    runs a little smoothing fxn over the array
    :param x: arr
    :param n: width of smooth
    :return: smoothed arr
    """
    return np.convolve(x, np.ones(n) / n, mode="same")


def tolerant_mean(arrs):
    """
    takes an average of arrays of different lengths
    :param arrs: N * arrs
    :return: mean arr, std arr
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens), len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[: len(l), idx] = l
    return arr.mean(axis=-1), arr.std(axis=-1)


def norm_fdff(cell_array):
    minVals = np.percentile(cell_array, 10, axis=1)
    zerod_arr = np.array([np.subtract(cell_array[n], i) for n, i in enumerate(minVals)])
    normed_arr = np.array([np.divide(arr, arr.max()) for arr in zerod_arr])
    return normed_arr


def subsection_arrays(input_array, offsets=(-10, 10)):
    a = []
    for repeat in range(len(input_array)):
        s = input_array[repeat] + offsets[0]
        e = input_array[repeat] + offsets[1]
        a.append(np.arange(s, e))
    return np.array(a)


def zdiffcell(arr):
    from scipy.stats import zscore

    diffs = np.diff(arr)
    zscores = zscore(diffs)
    prettyz = pretty(zscores, 3)
    return prettyz
