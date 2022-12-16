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
