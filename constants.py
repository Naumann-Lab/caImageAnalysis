invStimDict = {
    "medial_right": "medial_left",
    "medial_left": "medial_right",
    "right": "left",
    "left": "right",
    "converging": "converging",
    "diverging": "diverging",
    "lateral_left": "lateral_right",
    "lateral_right": "lateral_left",
    "forward": "backward",
    "backward": "forward",
    "forward_left": "backward_right",
    "backward_left": "forward_right",
    "backward_right": "forward_left",
    "forward_right": "backward_left",
    "x_forward": "backward_x",
    "forward_x": "x_backward",
    "backward_x": "x_forward",
    "x_backward": "forward_x",
    "forward_backward": "forward_backward",
    "backward_forward": "backward_forward",
}

# monocular are switched
bruker_invStimDict = {
    "medial_right": "lateral_right",
    "medial_left": "lateral_left",
    "right": "right",
    "left": "left",
    "converging": "diverging",
    "diverging": "converging",
    "lateral_left": "medial_left",
    "lateral_right": "medial_right",
    "forward": "forward",
    "backward": "backward",
    "forward_left": "backward_right",
    "backward_left": "forward_right",
    "backward_right": "forward_left",
    "forward_right": "backward_left",
    "x_forward": "forward_x",
    "forward_x": "x_forward",
    "backward_x": "x_backward",
    "x_backward": "backward_x",
    "forward_backward": "backward_forward",
    "backward_forward": "forward_backward",
}

monocular_dict = {
    "forward": [0, 1, 0],
    "forward_right": [0.75, 1, 0],
    "right": [1, 0.25, 0],
    "backward_right": [1, 0, 0.25],
    "backward": [1, 0, 1],
    "backward_left": [0.25, 0, 1],
    "left": [0, 0.25, 1],
    "forward_left": [0, 0.75, 1],
}

nulldict = {
    "right": "left",
    "left": "right",
    "forward": "backward",
    "backward": "forward",
    "forward_right": "backward_left",
    "backward_left": "forward_right",
    "forward_left": "backward_right",
    "backward_right": "forward_left",
}

baseBinocs = [
    "medial_left",
    "lateral_left",
    "medial_right",
    "lateral_right",
    "diverging",
    "converging",
]

deg_dict = {
    "forward": 0,
    "forward_left": 315,
    "left": 270,
    "backward_left": 225,
    "backward": 180,
    "backward_right": 135,
    "right": 90,
    "forward_right": 45,
    "medial_left": 999,
    "lateral_left": 999,
    "medial_right": 999,
    "lateral_right": 999,
    "diverging": 999,
    "converging": 999
}
eva_stims = [
    "converging",
    "diverging",
    "left",
    "medial_left",
    "lateral_left",
    "right",
    "medial_right",
    "lateral_right",
    "forward",
    "backward",
]
eva_typesL = {
    "oB": [False, True, True, True, True, False, False, False, True, True],
    "B": [False, False, True, True, True, False, False, False, True, True],
    "iB": [True, False, True, True, True, False, False, False, True, True],
    "ioB": [True, True, True, True, True, False, False, False, True, True],
    "iMm": [True, False, True, True, False, False, False, False, True, True],
    "Mm": [False, False, True, True, False, False, False, False, True, False],
    "oMl": [False, True, True, False, True, False, False, False, True, True],
    "S": [False, False, True, False, False, False, False, False, True, True],
}


velocity_mono_dict = {
    "forward": {0.04: [[0, 1, 0], 0.8], 0.02: [[0, 1, 0], 0.5], 0.01: [[0, 1, 0], 0.2]},
    "right": {
        0.04: [[1, 0.25, 0], 0.8],
        0.02: [[1, 0.25, 0], 0.5],
        0.01: [[1, 0.25, 0], 0.2],
    },
    "left": {
        0.04: [[0, 0.25, 1], 0.8],
        0.02: [[0, 0.25, 1], 0.5],
        0.01: [[0, 0.25, 1], 0.2],
    },
    "backward": {
        0.04: [[1, 0, 1], 0.8],
        0.02: [[1, 0, 1], 0.5],
        0.01: [[1, 0, 1], 0.2],
    },
}


bout_timing_color_dict = {
    "before": [0, 1, 0],
    "after": [1, 0.25, 0],
    "during": [0, 0.25, 1]}

allcolor_dict = {
    "right": [[1, 0.25, 0], [1, 0.25, 0]],
    "left": [[0, 0.25, 1], [0, 0.25, 1]],
    "forward": [[0, 1, 0], [0, 1, 0]],
    "backward": [[1, 0, 1], [1, 0, 1]],
    #"forward_backward": [[0, 1, 0], [1, 0, 1]],
    #"backward_forward": [[1, 0, 1], [0, 1, 0]],
    #"x_backward": [[0, 0, 0], [1, 0, 1]],
    #"x_forward": [[0, 0, 0], [0, 1, 0]],
    #"forward_x": [[0, 1, 0], [0, 0, 0]],
    #"backward_x": [[1, 0, 1], [0, 0, 0]],
    "forward_left": [[0, 0.75, 1], [0, 0.75, 1]],
    "forward_right": [[0.75, 1, 0], [0.75, 1, 0]],
    "backward_left": [[0.25, 0, 1], [0.25, 0, 1]],
    "backward_right": [[1, 0, 0.25], [1, 0, 0.25]],
    'diverging': [[0, 0.25, 1], [1, 0.25, 0]],
    'converging': [[1, 0.25, 0], [0, 0.25, 1]],
    'lateral_left': [[0, 0.25, 1], [0, 0, 0]],
    'medial_left': [[0, 0, 0], [0, 0.25, 1]],
    'lateral_right': [[0, 0, 0], [1, 0.25, 0]],
    'medial_right': [[1, 0.25, 0], [0, 0, 0]]
}

import cmasher as cmr
cmaplist = {'telencephalon': cmr.get_sub_cmap('spring', 0.3, 0.8),
            'PT': cmr.get_sub_cmap('Purples', 0.3, 0.8),
            'OT': cmr.get_sub_cmap('Blues', 0.3, 0.8),
            'tectum': cmr.get_sub_cmap('Blues', 0.3, 0.8),
            'tegmentum': cmr.get_sub_cmap('summer', 0.3, 0.8),
            'HBr': cmr.get_sub_cmap('Oranges', 0.3, 0.8),
            'anteriorHBr': cmr.get_sub_cmap('Wistia', 0.1, 0.5),
            'posteriorHBr': cmr.get_sub_cmap('Oranges', 0.3, 0.8)}

from matplotlib.colors import ListedColormap
import numpy as np
vals = np.ones((360, 4))
vals[:, 0] = np.concatenate((np.linspace(0, 1, 90), np.linspace(1, 1, 90), np.linspace(1, 0, 90), np.linspace(0, 0, 90)))
vals[:, 1] = np.concatenate((np.linspace(1, 0.25, 90), np.linspace(0.25, 0, 90), np.linspace(0, 0.25, 90), np.linspace(0.25, 1, 90)))
vals[:, 2] = np.concatenate((np.linspace(0, 0, 90), np.linspace(0, 1, 90), np.linspace(1, 1, 90), np.linspace(1, 0, 90)))
circmp = ListedColormap(vals)

dir_sort = ['forward', #'x_forward', 'forward_x',
            'forward_right', 'right', 'lateral_right', 'medial_right',
           'backward_right', 'backward', #'x_backward', 'backward_x',
            'backward_left',  'left', 'medial_left', 'lateral_left', 'forward_left',
           'converging', 'diverging'#, 'forward_backward', 'backward_forward'
            ]
