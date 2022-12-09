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
