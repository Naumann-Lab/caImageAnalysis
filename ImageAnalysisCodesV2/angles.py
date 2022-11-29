from cmath import rect, phase
from math import radians, degrees

import numpy as np


deg_dict = {

    'right' : 90,
    'forward_right' : 45,
    'forward' : 0,
    'forward_left' : 315,
    'left' : 270,
    'backward_left' : 215,
    'backward' : 180,
    'backward_right' : 135
}
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

def weighted_mean_angle(degs, weights):
    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d]*rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums)/np.sum(weights)))