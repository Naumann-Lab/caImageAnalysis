import colorsys
import numpy as np
from scipy.signal import butter, lfilter, freqz
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgb
pink = "#FB008C"
green = "#72D100"
pink = to_rgb(pink)
green = to_rgb(green)

#codes for omr related colors
def angle_to_rgba(angle, saturation, alpha):
    #@ChatGPt
    # Normalize angle to be between 0 and 360 degrees
    angle =  360 - angle % 360 +90
    # Convert to HSL
    h = angle / 360  # Normalize angle to [0, 1] range for colorsys
    s = saturation # [0, 1]
    l = .7 # Mid lightness
    # Convert HSL to RGB
    rgb = colorsys.hls_to_rgb(h, l, s)
    rgba = (rgb[0], rgb[1], rgb[2], alpha)
    return rgba

omr_angles = {'forward': 0, 'left': 270, 'right': 90, 'backward': 180,
               'forward_left': 315, 'forward_right': 45, 'backward_left': 225, 'backward_right': 135,
               'medial_left': 270, 'lateral_left': 270, 'medial_right': 90, 'lateral_right': 90,
               'converging': 0, 'diverging': 0, 'forward_backward': 0, 'backward_forward': 0,
                'x_forward': 0, 'x_backward': 180, 'forward_x': 0, 'backward_x': 180,
               'cw': 90, 'ccw': 270}
angles_omr = {0: 'forward', 270: 'left', 90: 'right', 180: 'backward'}
omr_colors = {key: angle_to_rgba(angle, 1, 1) for key, angle in omr_angles.items()}

omr_colors['stationary'] = (0.8, 0.8, 0.8, 1)
omr_cbars = {'forward': 'Greens', 'right': 'Reds', 'backward': 'Purples', 'left': 'Blues'}

#codes for dot angles
dot_angles = {'l': 270, 'r': 90}
dot_dists = {'far': 0.5, 'cls': 1}
dot_sizes = {'l': 1, 'm': 0.8, 's': 0.6, '3': 0.2, '5': 0.4, '10': 0.6, '20': 0.8, '40': 1}

dot_colors = {'dot_l': (0.415686, 0.352941, 0.803922, 1.0), 'dot_r': (0.78, 0.08, 0.52, 1.0)}#{'dot_' + dir + '_' + size: angle_to_rgba(angle, .5, alpha)
              #for dir, angle in dot_angles.items()
              #for size, alpha in dot_sizes.items()}
# dot_colors = {'dot_' + dir + '_' + dist + '_' + size: angle_to_rgba(angle, saturation, alpha)
#               for dir, angle in dot_angles.items()
#               for dist, saturation, in dot_dists.items()
#               for size, alpha in dot_sizes.items()}
simple_dot_colors = {'dot_' + dir : angle_to_rgba(angle, .5, 1)
              for dir, angle in dot_angles.items()}
dot_colors['stationary'] = (0.8, 0.8, 0.8, 1)
dot_colors['pause'] = (0.8, 0.8, 0.8, 1)

stim_colors = omr_colors|dot_colors#|simple_dot_colors

def make_cmap(stim_color, name = "costumcmap"):
    """Make a color map that goes from color (rgba), to white, to the anti color"""
    complement = (1 - stim_color[0], 1 - stim_color[1], 1 - stim_color[2], stim_color[3])
    colorlist = [(0.0, complement), (0.4, (1, 1, 1, 1)), (0.6, (1, 1, 1, 1)), (1.0, stim_color)]
    cmap = LinearSegmentedColormap.from_list(name, colorlist)
    return cmap


def weighted_mean_angle(degs, weights):
    """
    @Matt's pandastim
    :param degs:
    :param weights:
    :return: deg
    """
    from cmath import rect, phase
    from math import radians, degrees

    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d] * rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums) / np.sum(weights)))

def is_within_short_arc(angle, start, end):
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

def find_nearest_frameindex(frametime, target_time_s):
    """
    Return the index of in the frametime that is closest to the target_time_s
    :param frametime: the real time s for each frame
    :param target_time_s: the target time in s
    """
    return np.abs(np.subtract(frametime, target_time_s)).argmin()

def px_to_deg(y_distance_px, d_px):
    """
    Convert d from pixel distance to angular distance according to the distance y
    :param y_distance_px: the vertical/y distandce from the fish
    :param d: the distance in pixels to be converted
    :return: angle: the angle in deg for d
    """
    angle = np.arctan(d_px / y_distance_px)
    return np.rad2deg(angle)

def get_dot_pos(stim_df, canvas_size = 1024, timescale = None):
    """
    Get the fish egocentric dot angular trajectory across the trial

    :param stim_df: the one row dataframe for the stimulus, must be a stimulus containing dot
    :param canvas_size: the size of the canvas in pixel
    :param timescale: the timescale of the stimulus position to calculate, if None, generate something

    Return
        stim_x_distance_deg: where the stimulus starts from the egocentric view of the fish in deg (during stationary)
        stim_loc_timescale: the corresponding time point for stim_loc_scale
        stim_loc_scale: where the dot is for the egocentric view of the fish in deg
        center_time: the timepoint when the dot crosses the center
    """
    if 'dot' not in str(stim_df.stim_name):
        stim_x_distance_deg = np.nan
        stim_loc_timescale = np.full_like(timescale, np.nan)
        stim_loc_scale = np.full_like(timescale, np.nan)
        center_time = np.nan
    else:
        if '[' not in str(stim_df.stim_name):#pure dot stimulus
            stim_velocity_px = stim_df['velocity'] * canvas_size
            try:
                stim_y_distance_px = np.abs(stim_df['circle_center'][0][1])
                stim_x_distance_px = stim_df['circle_center'][0][0]
                print('plotting 2p setup dot position')
            except:
                stim_y_distance_px = np.abs(stim_df['circle_center'][1])
                stim_x_distance_px = stim_df['circle_center'][0]
                print('plotting behavior setup dot position')
            if stim_df['angle'] == -90:  # right stim, need to flip the x direction
                stim_velocity_px = -stim_velocity_px
                stim_x_distance_px = -stim_x_distance_px
            stationary_time = stim_df['stationary_time']
        else:#overlapping stimulus
            stim_velocity_px = stim_df['velocity'][0] * canvas_size
            stim_y_distance_px = np.abs(stim_df['circle_center'][0][1])
            stim_x_distance_px = stim_df['circle_center'][0][0]
            if stim_df['angle'][0] == -90:  # right stim, need to flip the x direction
                stim_velocity_px = -stim_velocity_px
                stim_x_distance_px = -stim_x_distance_px
            stationary_time = np.max(stim_df['stationary_time'])
        #actually calculating thing
        stim_x_distance_deg = px_to_deg(stim_y_distance_px, stim_x_distance_px)
        max_time = stim_df['duration']
        # plot stimulus angular location from the fish
        stim_loc_timescale = np.arange(0, max_time, 0.01)
        if type(timescale) == np.ndarray or type(timescale) == list:
            stim_loc_timescale = [t for t in timescale]
        stim_loc_scale = [
            px_to_deg(stim_y_distance_px, stim_x_distance_px + stim_velocity_px * (t - stationary_time))
            if t > stationary_time else stim_x_distance_deg
            for t in stim_loc_timescale]

        center_time = stim_loc_timescale[np.argmin(np.abs(stim_loc_scale))]

    return stim_x_distance_deg, stim_loc_timescale, stim_loc_scale, center_time

def cells_in_roi(candidate_pos_df, roi_array, midline = False):
    """
    Return cell index that are in the roi, of a plane
    :param candidate_pos_df: the position of all candidate cells, with a column of xpos and ypos
    :param roi_array: numpy array with the roi outline (x and y positions each occupy a row)
    :return: list of cell index in the roi
    """
    from shapely.geometry import Point, Polygon, LineString
    from shapely.ops import split
    roi = Polygon(roi_array)
    if type(midline) != bool:
        midline = np.array(midline)
        dx, dy = midline[1, 0] - midline[0, 0], midline[1, 1] - midline[0, 1]
        minx, miny, maxx, maxy = roi.bounds
        big = max(maxx - minx, maxy - miny) * 10
        # just extend the line in both directions
        midline = LineString([(midline[0, 0] - dx * big, midline[0, 1] - dy * big),
                               (midline[0, 0] + dx * big, midline[0, 1] + dy * big)])
        rois = list(split(roi, midline).geoms)
        rois.sort(key=lambda r: r.centroid.x)
        cells_in_roi_index_l = candidate_pos_df[
            candidate_pos_df.apply(lambda row: rois[0].contains(Point(row['xpos'], row['ypos'])), axis=1)].index
        cells_in_roi_index_r = candidate_pos_df[
            candidate_pos_df.apply(lambda row: rois[1].contains(Point(row['xpos'], row['ypos'])), axis=1)].index

        return cells_in_roi_index_l, cells_in_roi_index_r
    else:
        cells_in_roi_index = candidate_pos_df[
            candidate_pos_df.apply(lambda row: roi.contains(Point(row['xpos'], row['ypos'])), axis=1)].index
        return cells_in_roi_index

def butter_lowpass(cutoff, fs, order=5):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    :param data:
    :param cutoff: frequency above is done
    :param fs: sampling rate (frame/s)
    :param order:
    :return:
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def sigmoid_flexible(x, L=1, k=1, x0=0, b=0):
    """
    Flexible sigmoid function with adjustable range.
        @chatgpt
    Args:
      x: Input value(s).
      L: Scales the output range.
      k: Controls the steepness of the curve.
      x0: Shifts the curve horizontally.
      b: Adds a vertical offset to the output.

    Returns:
      Sigmoid transformed output with the specified range.
    """
    return list(L / (1 + np.exp(-k * (x - x0))) + b)


def parse_int_tuple(x):
    try:
        return (int(x.split('(')[1].split(',')[0]),
                int(x.split(',')[1].split(')')[0]))
    except (ValueError, IndexError):
        return (int(x.split('(')[2].split(')')[0]), int(x.split('(')[3].split(')')[0]))


def parse_float_tuple(x):
    try:
        return (float(x.split('(')[1].split(',')[0]),
                float(x.split(',')[1].split(')')[0]))
    except (ValueError, IndexError):
        return (float(x.split('(')[2].split(')')[0]), float(x.split('(')[3].split(')')[0]))