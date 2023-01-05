import constants

import numpy as np


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
