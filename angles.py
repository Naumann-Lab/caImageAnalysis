import constants

import numpy as np


def add_angles(ang1, ang2):
    return (ang1 + ang2) % 360


def calc_dsi(neuron_dict):
    # use neuron dict per neuron
    """
    (Rpref - Rnull) / Rpref
    looks like:
        {
        'medial_left': 0.461736435336726,
         'left': 0.5925878932078679,
         'converging': 0.029447067041127453,
         'lateral_left': 0.15455876752024603,
         'forward': 1.8847683771025565,
         'forward_left': 1.589355786641439,
         'backward_right': -0.05418827056529977,
         'lateral_right': -0.026855216466910425,
         'diverging': 0.010750960026468545,
         'backward_left': 0.07068517921669852,
         'medial_right': -0.05511859467341785,
         'forward_right': 0.18775133336228983,
         'right': -0.01675205305495877,
         'backward': 0.029301415571743355
         }
    """
    monoc_neuron = {
        k: v for k, v in neuron_dict.items() if k in constants.monocular_dict
    }
    inverse_dict = {v: k for k, v in monoc_neuron.items()}

    max_val = max(monoc_neuron.values())
    max_stim = inverse_dict[max_val]

    inverse_stim = constants.nulldict[max_stim]
    inverse_val = monoc_neuron[inverse_stim]
    return np.clip((max_val - inverse_val) / max_val, a_min=0, a_max=1)


def weighted_mean_angle(degs, weights):

    from cmath import rect, phase
    from math import radians, degrees

    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d] * rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums) / np.sum(weights)))


def color_returner(val, theta, threshold=0.5):

    if theta < 0:
        theta += 360

    if val >= threshold:
        # Forward
        if theta >= 337.5 or theta <= 22.5:
            outputColor = [0, 1, 0]

        # Forward Right
        elif 22.5 < theta <= 67.5:
            outputColor = [0.75, 1, 0]

        # Right
        elif 67.5 < theta <= 112.5:
            outputColor = [1, 0.25, 0]

        # Backward Right
        elif 112.5 < theta <= 157.5:
            outputColor = [1, 0, 0.25]

        # Backward
        elif 157.5 < theta <= 202.5:
            outputColor = [1, 0, 1]

        # Backward Left
        elif 202.5 < theta <= 247.5:
            outputColor = [0.25, 0, 1]

        # Left
        elif 247.5 < theta <= 292.5:
            outputColor = [0, 0.25, 1]

        # Forward Left
        elif 292.5 < theta <= 337.5:
            outputColor = [0, 0.75, 1]

        # if somehow we make it to here just make it gray
        else:
            outputColor = [0.66, 0.66, 0.66]

    else:
        # if not above some minimum lets make it gray
        outputColor = [0.66, 0.66, 0.66]
    return outputColor


def color_returner_continuous(val, theta):
    from colour import Color

    theta = int(theta % 360)  # get in 0-360 range

    clr_array = np.zeros([100, 360, 3])

    clr_array[0:99, :45] = [i.rgb for i in Color(rgb=[0, 1, 0]).range_to(Color(rgb=[0.75, 1, 0]), 45)]
    clr_array[0:99, 45:90] = [i.rgb for i in Color(rgb=[0.75, 1, 0]).range_to(Color(rgb=[1, 0.25, 0]), 45)]

    # the rgb of orange -> red wraps the wrong way so we manually fill it
    bs = np.linspace(0, 0.25, 45)
    gs = bs[::-1]
    reds = []
    for b, g in zip(bs, gs):
        reds.append([1, g, b])
    clr_array[0:99, 90:135] = reds

    clr_array[0:99, 135:180] = [i.rgb for i in Color(rgb=[1, 0, 0.25]).range_to(Color(rgb=[1, 0, 1]), 45)]
    clr_array[0:99, 180:225] = [i.rgb for i in Color(rgb=[1, 0, 1]).range_to(Color(rgb=[0.25, 0, 1]), 45)]
    clr_array[0:99, 225:270] = [i.rgb for i in Color(rgb=[0.25, 0, 1]).range_to(Color(rgb=[0, 0.25, 1]), 45)]
    clr_array[0:99, 270:315] = [i.rgb for i in Color(rgb=[0, 0.25, 1]).range_to(Color(rgb=[0, 0.75, 1]), 45)]
    clr_array[0:99, 315:360] = [i.rgb for i in Color(rgb=[0, 0.75, 1]).range_to(Color(rgb=[0, 1, 0]), 45)]

    for i in range(clr_array.shape[1]):
        rgb = clr_array[0, i]

        for n in [0, 1, 2]:
            clrs = np.linspace(0, rgb[n], 100)
            clr_array[:, i, n] = clrs

    final_clrs = np.zeros([100, 360, 4])
    final_clrs[:, :, :3] = clr_array
    del clr_array

    for i in range(final_clrs.shape[1]):
        final_clrs[:, i, 3] = np.linspace(0, 1, 100)

    val = np.clip(int(val * 100), a_min=0, a_max=99)

    return final_clrs[val, theta]


def make_clr_array():
    from colour import Color

    clr_array = np.zeros([100, 360, 3])

    clr_array[0:99, :45] = [i.rgb for i in Color(rgb=[0, 1, 0]).range_to(Color(rgb=[0.75, 1, 0]), 45)]
    clr_array[0:99, 45:90] = [i.rgb for i in Color(rgb=[0.75, 1, 0]).range_to(Color(rgb=[1, 0.25, 0]), 45)]

    # the rgb of orange -> red wraps the wrong way so we manually fill it
    bs = np.linspace(0, 0.25, 45)
    gs = bs[::-1]
    reds = []
    for b, g in zip(bs, gs):
        reds.append([1, g, b])
    clr_array[0:99, 90:135] = reds

    clr_array[0:99, 135:180] = [i.rgb for i in Color(rgb=[1, 0, 0.25]).range_to(Color(rgb=[1, 0, 1]), 45)]
    clr_array[0:99, 180:225] = [i.rgb for i in Color(rgb=[1, 0, 1]).range_to(Color(rgb=[0.25, 0, 1]), 45)]
    clr_array[0:99, 225:270] = [i.rgb for i in Color(rgb=[0.25, 0, 1]).range_to(Color(rgb=[0, 0.25, 1]), 45)]
    clr_array[0:99, 270:315] = [i.rgb for i in Color(rgb=[0, 0.25, 1]).range_to(Color(rgb=[0, 0.75, 1]), 45)]
    clr_array[0:99, 315:360] = [i.rgb for i in Color(rgb=[0, 0.75, 1]).range_to(Color(rgb=[0, 1, 0]), 45)]

    for i in range(clr_array.shape[1]):
        rgb = clr_array[0, i]

        for n in [0, 1, 2]:
            clrs = np.linspace(0, rgb[n], 100)
            clr_array[:, i, n] = clrs

    final_clrs = np.zeros([100, 360, 4])
    final_clrs[:, :, :3] = clr_array

    for i in range(final_clrs.shape[1]):
        final_clrs[:, i, 3] = np.linspace(0, 1, 100)
    return final_clrs


def continuous_clr_array(val, theta, clr_array):
    theta = int(theta % 360)  # get in 0-360 range

    val = np.clip(int(val * 100), a_min=0, a_max=99)

    return clr_array[val, theta]