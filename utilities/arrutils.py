import numpy as np

import math


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

def norm_0to1(cell_array):
    if len(cell_array.shape) == 1:
        norm_cell_arr = np.array((cell_array - np.min(cell_array)) / (np.max(cell_array) - np.min(cell_array)))
    else:
        norm_cell_arr = np.array([(c - np.min(c)) / (np.max(c) - np.min(c)) for c in cell_array])
    return norm_cell_arr

def norm_fdff(cell_array):
    minVals = np.percentile(cell_array, 10, axis=1)
    zerod_arr = np.array([np.subtract(cell_array[n], i) for n, i in enumerate(minVals)])
    normed_arr = np.array([np.divide(arr, arr.max()) for arr in zerod_arr])
    return normed_arr

def norm_fdff_new(cell_array, lowPct=15, highPct=95):
    minVals = np.percentile(cell_array, lowPct, axis=1)
    zerod_arr = np.array([np.subtract(cell_array[n], i) for n, i in enumerate(minVals)])
    normed_arr = np.array([np.divide(arr, np.percentile(arr, highPct)) for arr in zerod_arr])
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

def zscoring(data_array):
    from scipy.stats import zscore

    zscores = zscore(data_array)
    conv_zscores = pretty(zscores)
    return conv_zscores

def arrs_to_medians(arrs, off1, off2):
    return np.nanmedian([i[off1 : off1 + off2] for i in arrs], axis=0)

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return array[idx - 1], idx - 1
    else:
        return array[idx], idx
    
# remove values in a list of elements in form [x,y] where the range from element overlaps with range of next element
def remove_nearest_vals(list_of_some_vals):
    new_list = []
    bad_vals = []
    for a, val in enumerate(list_of_some_vals):
        try:
            b = a+1
            next_val = list_of_some_vals[b]
            if val[1] >= next_val[0]: #if the y value is less than the x of the previous value
                new_val = [val[0], next_val[1]] #make a new value that has x of 1, y of 2
                new_list.append(new_val)
                bad_vals.append(val)
                bad_vals.append(next_val)
            else:
                new_list.append(val)
        except:
            new_list.append(val) #adds the last value in the list

    final_list = new_list
    for x in final_list:
        if x in bad_vals:
            final_list.remove(x) # removing the bad values

    return final_list

def convert_frame_to_sec(frame_lst, framerate):
    '''
    converts a list of frames to seconds given the framerate (helpful for plotting)
    :param frame_lst: list of frames
    :param framerate: framerate of the recording
    '''
    return [x / framerate for x in frame_lst]

def filter_list(lst, interval):
    filtered_list = [lst[0]]
    for i in range(1, len(lst)):
        if abs(lst[i-1]-lst[i]) > interval:
            filtered_list.append(lst[i])
    return filtered_list