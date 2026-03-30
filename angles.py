import constants
import numpy as np

import photostim_data_pipeline
from utilities import arrutils


def add_angles(ang1, ang2):
    return (ang1 + ang2) % 360


def calc_dsi(neuron_dict):
    # use neuron dict per neuron
    """
    Typical DSI metric
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


def calc_dsi_cardinaldirs(vizstimfishy, base_sec = 4, motion_on_sec = 10, dsi_threshold = 0.45, use_df_f = False):
    '''
    calculate the dsi for all the neurons in a vizstimfishy
    :param vizstimfishy: class instance of a VizStimFish
    :param base_sec: seconds of baseline to use before motion starts for calculating baseline values
    :param motion_on_sec: seconds of motion to use in calculating response during on period
    :param dsi_threshold: threshold for dsi calculation, if under this value than dsi == 0
    :param use_df_f: whether to calculate df/f for each neuron, or just use the normalized trace
    :return: list of dsi values for all the neurons in the vizstim fishy
    '''

    motion_frame_offsets = vizstimfishy.offsets
    base_frames = int(base_sec * vizstimfishy.img_hz)
    base_start = -motion_frame_offsets[0] - base_frames
    motion_on_frames = int(motion_on_sec * vizstimfishy.img_hz)
    motion_end_frames = -motion_frame_offsets[0] + motion_on_frames

    # using 4 cardinal directions to calculate DSI
    directions = ['forward', 'right', 'backward', 'left']
    # Cardinal directions in radians
    directions_radians = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    unit_vecs = np.column_stack((np.cos(directions_radians), np.sin(directions_radians)))

    motion_resp_dict_list = photostim_data_pipeline.gather_visual_motion_responses_for_df(vizstimfishy,
                                                                                          cell_id_array = None,
                                                                                          motion_cues = directions,
                                                                                          get_df_f = use_df_f,
                                                                                          response_type = 'mean')
    dsi_list = []
    for neuron, motion_responses in enumerate(motion_resp_dict_list):
        motion_responses = motion_resp_dict_list[neuron]
        stim_scores = []
        for d in directions:
            resp = arrutils.pretty(motion_responses[d]['mean'])
            base_avg = np.nanmean(resp[base_start:-motion_frame_offsets[0]])
            on_window = resp[-motion_frame_offsets[0]:motion_end_frames]
            on_avg = np.nanmean(on_window)
            on_max = np.nanmax(on_window)
            on_min = np.nanmin(on_window)
            score = on_max - base_avg if on_avg > base_avg else on_min - base_avg
            stim_scores.append(score)

        stim_scores = np.array(stim_scores)
        stim_scores = np.maximum(stim_scores, 0) # rectify, treats negative responses like 0

        # Vector DSI calculation
        total = stim_scores.sum()
        if total < dsi_threshold:
            dsi = 0
        else:
            dsi = np.linalg.norm(stim_scores @ unit_vecs) / total
        dsi_list.append(dsi)

    return dsi_list


def weighted_mean_angle(degs, weights):

    from cmath import rect, phase
    from math import radians, degrees

    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d] * rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums) / np.sum(weights)))


def is_within_short_arc(angle, start, end):
    '''
    Check if an angle is within a short arc defined by start and end angles.
    '''

    def normalize(deg):
        return deg % 360

    a = normalize(angle)
    s = normalize(start)
    e = normalize(end)

    diff = (e - s) % 360

    if diff == 0:
        return a == s  # exact point
    elif diff <= 180:
        # Clockwise arc from start to end
        return (a - s) % 360 <= diff
    else:
        # Counter-clockwise arc (shorter)
        return (s - a) % 360 <= (360 - diff)


def angle_to_rgba(angle, saturation, alpha):
    import colorsys
    #@ChatGPt
    # Normalize angle to be between 0 and 360 degrees
    angle =  360 - angle % 360 +90
    # Convert to HSL
    h = angle / 360  # Normalize angle to [0, 1] range for colorsys
    s = saturation # [0, 1]
    l = 0.5  # Mid lightness
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    rgba = (rgb[0], rgb[1], rgb[2], alpha)
    return rgba

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

def sort_df_by_continous_color(df, target_value, df_column_name='color'):
    '''
    For plotting purposes, I want to have all the grey values at the bottom of actual color dots.
    Here, I sort the DataFrame by the color column, where the target_value is the color of the grey dots.
    Need to convert it into tuples since too many values to deal with in a sort function.
    '''
    # Sort the DataFrame
    df['tuple_values'] = df[df_column_name].apply(convert_to_tuple)
    df_sorted = df.sort_values(by='tuple_values', key=lambda x: x.apply(lambda v: (v != tuple(target_value), v)), ascending = True)

    # Drop the helper column
    df_sorted = df_sorted.drop(columns=['tuple_values'])

    return df_sorted

# Convert lists to tuples for comparison
def convert_to_tuple(arr):
    return tuple(arr)
