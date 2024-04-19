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
    "right": [1, 0.25, 0],
    "left": [0, 0.25, 1],
    "forward": [0, 1, 0],
    "backward": [1, 0, 1],
    "forward_left": [0, 0.75, 1],
    "forward_right": [0.75, 1, 0],
    "backward_left": [0.25, 0, 1],
    "backward_right": [1, 0, 0.25],
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
    "right": 90,
    "forward_right": 45,
    "forward": 0,
    "forward_left": 315,
    "left": 270,
    "backward_left": 215,
    "backward": 180,
    "backward_right": 135,
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

eva_typesR = {
    "oB": [False, True, False, False, True, True, True, False, True, True],
    "B": [False, False, False, False, False, True, True, True, True, True],
    "iB": [True, False, False, False, False, True, True, True, True, True],
    "ioB": [True, True, False, False, False, True, True, True, True, True],
    "iMm": [True, False, False, False, False, True, True, False, True, True],
    "Mm": [False, False, False, False, False, True, True, False, True, False],
    "oMl": [False, True, False, False, False, True, False, True, True, True],
    "S": [False, False, False, False, False, True, False, False, True, True],
}


eva_types_all = {
    "oB_L": [False, True, True, True, True, False, False, False, True, True],
    "B_L": [False, False, True, True, True, False, False, False, True, True],
    "iB_L": [True, False, True, True, True, False, False, False, True, True],
    "ioB_L": [True, True, True, True, True, False, False, False, True, True],
    "iMm_L": [True, False, True, True, False, False, False, False, True, True],
    "Mm_L": [False, False, True, True, False, False, False, False, True, False],
    "oMl_L": [False, True, True, False, True, False, False, False, True, True],
    "S_L": [False, False, True, False, False, False, False, False, True, True],
    "oB_R": [False, True, False, False, True, True, True, False, True, True],
    "B_R": [False, False, False, False, False, True, True, True, True, True],
    "iB_R": [True, False, False, False, False, True, True, True, True, True],
    "ioB_R": [True, True, False, False, False, True, True, True, True, True],
    "iMm_R": [True, False, False, False, False, True, True, False, True, True],
    "Mm_R": [False, False, False, False, False, True, True, False, True, False],
    "oMl_R": [False, True, False, False, False, True, False, True, True, True],
    "S_R": [False, False, False, False, False, True, False, False, True, True],
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

