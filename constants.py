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

# monocular are switched (Nov 2024 - Dec 2024)
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

binocular_dict = {
    "medial_left" : [0, 1, 0],
    "medial_right" : [0, 1, 0],
    "lateral_left" : [1, 0, 1],
    "lateral_right" : [1, 0, 1]
}

combined_binocular_dict = {
    "medial" : [0, 1, 0],
    "lateral" : [1, 0, 1],
    "binocular" : [0, 0, 0],
    "converging" : [0, 1, 1],
    "diverging" : [1, 0, 1],
}

shearing_stims_dict = {
    'forward_backward': [0, 0, 0],
    'forward_x': [0, .5, .5],
    'backward_x': [1, 0.647, 0],
    'backward_forward': [0, 0, 0],
    'x_forward': [0, .5, .5],
    'x_backward': [1, 0.647, 0],
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

eight_directions_ordered = ['forward',
                            'forward_right',
                            'right',
                            'backward_right',
                            'backward',
                            'backward_left',
                            'left',
                            'forward_left']

deg_dict_expanded = {
    "right": 90,
    "forward_right": 45,
    "forward": 0,
    "forward_left": 315,
    "left": 270,
    "backward_left": 215,
    "backward": 180,
    "backward_right": 135,

    # adding in the other directions so that i can make tuning circles as well
    "medial_right": 45,
    "converging": 0,
    "medial_left": 315,
    "lateral_left": 215,
    "diverging": 180,
    "lateral_right": 135,
    "forward_backward": 0,
    "forward_x": 60,
    "x_backward": 300,
    "backward_forward": 180,
    "backward_x": 120,
    "x_forward": 240,
}

direction_types_dict = {'cardinal': ['forward', 'right', 'backward', 'left'],
                        'binocular': ['converging','medial_right', 'right', 'lateral_right',  'diverging', 'lateral_left', 'left', 'medial_left'],
                        'shearing': ['forward_backward', 'forward_x', 'backward_x', 'backward_forward', 'x_forward','x_backward',]}


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

eva_types_colors = {
    "oB": 'tab:blue',
    "B": 'tab:orange',
    "iB": 'tab:pink',
    "ioB": 'tab:brown',
    "iMm": 'tab:green',
    "Mm": 'tab:cyan',
    "oMl": 'tab:purple',
    "S": 'tab:olive'
}

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

eva_types_all_short = {
    "oB_L": [False, True, True, True, True, False, False, False],
    "B_L": [False, False, True, True, True, False, False, False],
    "iB_L": [True, False, True, True, True, False, False, False],
    "ioB_L": [True, True, True, True, True, False, False, False],
    "iMm_L": [True, False, True, True, False, False, False, False],
    "Mm_L": [False, False, True, True, False, False, False, False],
    "oMl_L": [False, True, True, False, True, False, False, False],
    "S_L": [False, False, True, False, False, False, False, False],
    "oB_R": [False, True, False, False, True, True, True, False],
    "B_R": [False, False, False, False, False, True, True, True],
    "iB_R": [True, False, False, False, False, True, True, True],
    "ioB_R": [True, True, False, False, False, True, True, True],
    "iMm_R": [True, False, False, False, False, True, True, False],
    "Mm_R": [False, False, False, False, False, True, True, False],
    "oMl_R": [False, True, False, False, False, True, False, True],
    "S_R": [False, False, False, False, False, True, False, False],
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


bout_inducing_stims = ['forward', 'forward_x', 'x_forward', 'forward_right', 'forward_left','right', 'medial_right', 'left', 'medial_left', 'converging']
bout_reducing_stims = ['backward', 'backward_right', 'backward_left', 'x_backward', 'backward_x', 'lateral_right', 'lateral_left', 'diverging']


# visual motion cues used for photostimulation experiments, matching colors
photostim_motion_cues = ['medial_left', 'lateral_left', 'left', 'medial_right', 'lateral_right', 'right', 
               'converging', 'diverging',  'backward_forward', 'x_forward', 'x_backward', 
               'forward_backward', 'forward_x', 'backward_x','forward', 'backward']

photostim_motion_cues_colors = ['tab:green', 'tab:purple', 'black', 'tab:green', 'tab:purple', 'black', 'c', 'm', 
                                'black', 'teal', 'tab:orange', 'black', 'teal', 'tab:orange', 'tab:blue', 'tab:orange']

shearing_stimuli = ['backward_forward', 'x_forward', 'x_backward', 
               'forward_backward', 'forward_x', 'backward_x','forward', 'backward']

bouting_stims = [
    "converging",
    "diverging",
    "left",
    "medial_left",
    "lateral_left",
    "right",
    "medial_right",
    "lateral_right",
    'forward', 
    'forward_x', 
    'x_forward', 
    'backward', 
    'backward_x', 
    'x_backward', 
    'forward_backward', 
    'backward_forward']

bouting_barcode_types = {
    'iMm_L_F': [True, False, True, True, False, False, False, False, True, False, True, False, False, False, False, True],
    'iMm_R_F' : [True, False, False, False, False, True, True, False, True, True, False, False, False, False, True, False],
    'iMm_L_B' : [True, False, True, True, False, False, False, False, False, False, False, True, False, True, True, False],
    'iMm_R_B' : [True, False, False, False, False, True, True, False, False, False, False, True, True, False, False, True],
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


stim_title_dict = {
    "right" : r'$\Rightarrow$$\Rightarrow$',
    "left" : r'$\Leftarrow$$\Leftarrow$',
    "forward" : r'$\Uparrow$$\Uparrow$',
    "backward" : r'$\Downarrow$$\Downarrow$',
    "forward_right" : r'$\Nearrow$$\Nearrow$',
    "backward_right" : r'$\Searrow$$\Searrow$',
    "backward_left" : r'$\Swarrow$$\Swarrow$',
    "forward_left" : r'$\Nwarrow$$\Nwarrow$',
    "medial_left" : r' ---$\Leftarrow$',
    "medial_right" : r'$\Rightarrow$--- ',
    "lateral_left" : r'$\Leftarrow$--- ',
    "lateral_right" : r' ---$\Rightarrow$',
    "converging" : r'$\Rightarrow$$\Leftarrow$',
    "diverging" : r'$\Leftarrow$$\Rightarrow$',

    "forward_backward" : r'$\Uparrow$$\Downarrow$',
    "backward_forward" : r'$\Downarrow$$\Uparrow$',
    "x_forward" : r' ---$\Uparrow$',
    "forward_x" : r'$\Uparrow$--- ',
    "x_backward" : r' ---$\Downarrow$',
    "backward_x" : r'$\Downarrow$--- ',
}
