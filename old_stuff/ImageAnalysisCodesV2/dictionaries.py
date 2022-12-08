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

monocular_dict_alpha = {
    "right": [1, 0.25, 0, 1],
    "left": [0, 0.25, 1, 1],
    "forward": [0, 1, 0, 1],
    "backward": [1, 0, 1, 1],
    "forward_left": [0, 0.75, 1, 1],
    "forward_right": [0.75, 1, 0, 1],
    "backward_left": [0.25, 0, 1, 1],
    "backward_right": [1, 0, 0.25, 1],
}
baseBinocs = ["medial_left", "lateral_left", "medial_right", "lateral_right"]
