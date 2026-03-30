import pandas as pd
import numpy as np
from numpy import nan #important for stimuli fxns
from datetime import datetime


def pandastim_to_df(pstimpath, minimode=False, addvelocity=True):
    '''
    Puts the original text file with pandastim information into a dataframe
    :param pstimpath:
    :param minimode:
    :param addvelocity:
    :return:
    '''
    with open(pstimpath) as file:
        contents = file.read()

    lines = contents.split("\n")

    motionOns = [i for i in lines if "motionOn" in i.split("_&_")[-1]]
    times = [i.split("_&_")[0] for i in motionOns]
    stims = [eval(i[i.find("{") :]) for i in motionOns]
    stimulus_only = [i["stimulus"] for i in stims]

    stimulus_df = pd.DataFrame(stimulus_only)
    stimulus_df.loc[:, "datetime"] = times
    stimulus_df.datetime = pd.to_datetime(stimulus_df.datetime)
    stimulus_df.loc[:, "time"] = [
        pd.Timestamp(i).time() for i in stimulus_df.datetime.values
    ]

    mini_stim = stimulus_df[["stim_name", "time"]]
    mini_stim.stim_name = pd.Series(mini_stim.stim_name, dtype="category")

    mini_stim_vel = stimulus_df[["stim_name", "velocity", "time"]]
    mini_stim_vel.stim_name = pd.Series(mini_stim.stim_name, dtype="category")
    if minimode:
        return mini_stim
    elif addvelocity:
        return mini_stim_vel
    else:
        return stimulus_df


def add_repetitions_to_stimulus_df(stimulus_df, keep_all_reps = True):
    '''

    :param stimulus_df: pstim dataframe
    :param keep_all_reps: if you want to keep all the reps no matter what, else you will only put down reps for complete rounds
    :return: dataframe with a new column 'rep'
    '''
    stimulus_df = stimulus_df.copy()
    stimulus_df['rep'] = 0

    if keep_all_reps:
        # repetition index per stimulus (order-safe)
        stimulus_df['rep'] = stimulus_df.groupby('stim_name').cumcount()
    else:
        # default rep = -1 (invalid / incomplete)
        stimulus_df['rep'] = -1

        # temporary rep index per stimulus (order-safe)
        tmp_rep = stimulus_df.groupby('stim_name').cumcount()

        stimulus_df['_tmp_rep'] = tmp_rep

        n_stims = stimulus_df['stim_name'].nunique()

        # find which rep indices are complete
        rep_counts = stimulus_df.groupby('_tmp_rep')['stim_name'].nunique()
        valid_reps = rep_counts[rep_counts == n_stims].index

        # assign rep only for complete reps
        stimulus_df.loc[
            stimulus_df['_tmp_rep'].isin(valid_reps),
            'rep'
        ] = stimulus_df.loc[
            stimulus_df['_tmp_rep'].isin(valid_reps),
            '_tmp_rep'
        ]

        # clean up
        stimulus_df.drop(columns='_tmp_rep', inplace=True)

    return stimulus_df.reset_index(drop=True)

def get_common_reps(fishy, frames_motion_on):
    """
    Returns indices of reps that are valid for ALL stimuli. Complete reps for all the stim in the experiment
    """
    o_t = fishy.neur_resps_each_stim_rep
    n_reps = o_t.shape[1]
    n_stim = len(fishy.stim_order)

    length_subset = np.diff(fishy.offsets)[0]
    before_stim = -fishy.offsets[0]

    valid_reps = []

    for k in range(n_reps):
        rep_ok = True

        for j in range(n_stim):
            win0 = length_subset*j + before_stim
            win1 = win0 + frames_motion_on

            # check ANY neuron — NaNs pattern is same across neurons
            test_trace = o_t[0, k, win0:win1]

            if np.all(np.isnan(test_trace)):
                rep_ok = False
                break

        if rep_ok:
            valid_reps.append(k)

    return np.array(valid_reps, dtype=int)

def csv_to_df(csvpath, minimode=False, addvelocity=False):
    stimulus_df = pd.read_csv(csvpath)
    stimulus_df.drop(columns=['Unnamed: 0'], inplace=True)

    strings = stimulus_df.time.values
    datetime_full = [datetime.strptime(string, '%H:%M:%S.%f') for string in strings]
    datetime_times = [dt.time() for dt in datetime_full]
    stimulus_df['time'] = datetime_times # convert times to actual datetime objects

    potential_cols = ['stim_name', 'time', 'frame', 'rep'] # adjust to get these columns if they are there
    keep_cols = [s for s in potential_cols if s in stimulus_df.columns]
    mini_stim = stimulus_df[keep_cols]
    mini_stim.stim_name = pd.Series(mini_stim.stim_name, dtype="category")

    if minimode:
        return mini_stim
    elif addvelocity:
        mini_stim_vel = stimulus_df[["stim_name", "velocity", "time"]]
        mini_stim_vel.stim_name = pd.Series(mini_stim.stim_name, dtype="category")
        return mini_stim_vel
    else:
        return stimulus_df

def legacy_struct_pandastim_to_df(folderPath, stim_key, *args, **kwargs):
    import os

    with os.scandir(folderPath.parents[0]) as entries:
        for entry in entries:
            if stim_key in entry.name:
                stimPath = entry.path

    if stimPath:
        df = pandastim_to_df(stimPath, *args, **kwargs)
        return df

def stim_shader(some_fish_class):
    """
    Shades a plot with the stim overlays using class info

    :param some_fish_class:
    :return:
    """
    import constants
    import matplotlib.pyplot as plt

    frames = some_fish_class.stimulus_df["frame"].values
    stimmies = some_fish_class.stimulus_df["stim_name"].values

    for s, stimmy in zip(frames, stimmies):

        begin = s
        end = s + some_fish_class.stim_offset + 2
        midpt = begin + (end - begin) // 2

        if stimmy in constants.monocular_dict.keys():
            plt.axvspan(
                begin,
                end,
                color=constants.monocular_dict[stimmy],
                alpha=0.4,
            )

        elif stimmy in constants.baseBinocs:

            if stimmy == "lateral_left":
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                plt.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
            if stimmy == "medial_left":
                plt.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stimmy == "lateral_right":
                plt.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )
            if stimmy == "medial_right":
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                plt.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")

            if stimmy == "converging":
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stimmy == "diverging":
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )
        else:

            if stimmy == "x_forward":
                plt.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=0.4
                )

            if stimmy == "forward_x":
                plt.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=0.4
                )

            if stimmy == "x_backward":
                plt.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "backward_x":
                plt.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "backward_forward":
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=0.4
                )
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "forward_backward":
                plt.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=0.4
                )
                plt.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=0.4
                )

def stimulus_start_frames_for_plots(baseline_offset = 7, length_of_total_frame_arr = 21, number_of_stims_in_set = 8):
    '''
    baseline_offset -- the relative frames before the stimulus starts (-fishy.offset[0])
    length_of_total_frame_arr -- the total number of frames that is taken from the neural trace before and after the stimulus is on
    (typically diff between offsets, i.e. 21)
    number_of_stims_in_set -- the number of stimuli in the experiment (8 for the 8 barcoded stimuli)

    returns a list of the starting frames for each stimulus in the set, this is what starts the shading in the plots
    '''

    list = np.linspace(0, length_of_total_frame_arr*(number_of_stims_in_set-1),
                       number_of_stims_in_set) + baseline_offset
    list = [int(i) for i in list]

    return list

def flexible_stim_shader(frames, stimmies, frames_motion_on, fs = 14, subplot = None, label = True,
                         ylabel_pos = None, label_offset_x = -6, alpha = 0.3):
    import constants
    import matplotlib.pyplot as plt


    if subplot == None:
        ax_n = plt
        y_top = round(max(plt.gca().get_ylim()))
    else:
        ax_n = subplot
        y_top = round(max(ax_n.get_ylim()))

    if ylabel_pos == None:
        ylabel_pos = y_top + 0.02
    else:
        ylabel_pos = ylabel_pos

    for s, stimmy in zip(frames, stimmies):
        if label:
            ax_n.text(s + label_offset_x, ylabel_pos, constants.stim_title_dict[stimmy], fontsize=fs)

        begin = s
        end = s + frames_motion_on
        midpt = begin + (end - begin) // 2

        if stimmy in constants.monocular_dict.keys():
            ax_n.axvspan(begin, end, color=constants.monocular_dict[stimmy], alpha=alpha,)

        elif stimmy in constants.baseBinocs:

            if stimmy == "lateral_left":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=alpha
                )
                ax_n.axvspan(midpt, end, color="gray", alpha=alpha, hatch=r"\\\\")
            if stimmy == "medial_left":
                ax_n.axvspan(begin, midpt, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=alpha
                )

            if stimmy == "lateral_right":
                ax_n.axvspan(begin, midpt, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=alpha
                )
            if stimmy == "medial_right":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=alpha
                )
                ax_n.axvspan(midpt, end, color="gray", alpha=alpha, hatch=r"\\\\")

            if stimmy == "converging":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=alpha
                )
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=alpha
                )

            if stimmy == "diverging":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=alpha
                )
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=alpha
                )
        else:

            if stimmy == "x_forward":
                ax_n.axvspan(begin, midpt, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=alpha
                )

            if stimmy == "forward_x":
                ax_n.axvspan(midpt, end, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=alpha
                )

            if stimmy == "x_backward":
                ax_n.axvspan(begin, midpt, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=alpha
                )

            if stimmy == "backward_x":
                ax_n.axvspan(midpt, end, color="gray", alpha=alpha, hatch=r"\\\\")
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=alpha
                )

            if stimmy == "backward_forward":
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=alpha
                )
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=alpha
                )

            if stimmy == "forward_backward":
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=alpha
                )
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=alpha
                )


def flexible_stim_bar(
        frames,
        stimmies,
        frames_motion_on,
        fs=14,
        subplot=None,
        label=True,
        ylabel_pos=None,
        label_offset_x=-6,
        alpha=0.8,
        bar_height_frac=0.05,
        bar_offset_frac=0.02,
    ):
        """
        Draw stimulus indicators as a thin bar along the top of the axis
        using matplotlib Rectangle patches (data coordinates).
        """

        import constants
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # ---- axis handling ----
        if subplot is None:
            ax = plt.gca()
        else:
            ax = subplot

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

        bar_height = bar_height_frac * y_range
        bar_bottom = y_max + bar_offset_frac * y_range

        if ylabel_pos is None:
            ylabel_pos = y_max + 0.01 * y_range

        # ---- helper to draw a bar segment ----
        def draw_bar(x0, x1, color, hatch=None):
            rect = Rectangle(
                (x0, bar_bottom),
                x1 - x0,
                bar_height,
                facecolor=color,
                edgecolor='none',
                alpha=alpha,
                hatch=hatch,
                clip_on=False
            )
            ax.add_patch(rect)

        # ---- main loop ----
        for s, stimmy in zip(frames, stimmies):

            begin = s
            end = s + frames_motion_on
            midpt = begin + (end - begin) // 2

            # ---- label ----
            if label:
                ax.text(
                    s + label_offset_x,
                    ylabel_pos,
                    constants.stim_title_dict.get(stimmy, stimmy),
                    fontsize=fs,
                    ha='left',
                    va='bottom'
                )

            # ---- monocular stimuli ----
            if stimmy in constants.monocular_dict:

                draw_bar(begin, end, constants.monocular_dict[stimmy])

            # ---- binocular stimuli ----
            elif stimmy in constants.baseBinocs:

                if stimmy == "lateral_left":
                    draw_bar(begin, midpt, constants.monocular_dict["left"])
                    draw_bar(midpt, end, "gray", hatch=r"\\\\")

                elif stimmy == "medial_left":
                    draw_bar(begin, midpt, "gray", hatch=r"\\\\")
                    draw_bar(midpt, end, constants.monocular_dict["left"])

                elif stimmy == "lateral_right":
                    draw_bar(begin, midpt, "gray", hatch=r"\\\\")
                    draw_bar(midpt, end, constants.monocular_dict["right"])

                elif stimmy == "medial_right":
                    draw_bar(begin, midpt, constants.monocular_dict["right"])
                    draw_bar(midpt, end, "gray", hatch=r"\\\\")

                elif stimmy == "converging":
                    draw_bar(begin, midpt, constants.monocular_dict["right"])
                    draw_bar(midpt, end, constants.monocular_dict["left"])

                elif stimmy == "diverging":
                    draw_bar(begin, midpt, constants.monocular_dict["left"])
                    draw_bar(midpt, end, constants.monocular_dict["right"])

            # ---- mixed forward/backward stimuli ----
            else:

                if stimmy == "x_forward":
                    draw_bar(begin, midpt, "gray", hatch=r"\\\\")
                    draw_bar(midpt, end, constants.monocular_dict["forward"])

                elif stimmy == "forward_x":
                    draw_bar(begin, midpt, constants.monocular_dict["forward"])
                    draw_bar(midpt, end, "gray", hatch=r"\\\\")

                elif stimmy == "x_backward":
                    draw_bar(begin, midpt, "gray", hatch=r"\\\\")
                    draw_bar(midpt, end, constants.monocular_dict["backward"])

                elif stimmy == "backward_x":
                    draw_bar(begin, midpt, constants.monocular_dict["backward"])
                    draw_bar(midpt, end, "gray", hatch=r"\\\\")

                elif stimmy == "backward_forward":
                    draw_bar(begin, midpt, constants.monocular_dict["backward"])
                    draw_bar(midpt, end, constants.monocular_dict["forward"])

                elif stimmy == "forward_backward":
                    draw_bar(begin, midpt, constants.monocular_dict["forward"])
                    draw_bar(midpt, end, constants.monocular_dict["backward"])


def numToStim(dict):
    reverse_dict = {value: key for key, value in dict.items()}
    return reverse_dict

def combine_binocular_stims_for_tail_df(df, column_name = 'tail_angle'):
    '''
    Combine all the binocular stimuli together
    Requires flipping all the left tail angles
    '''
    df_copy = df.copy()

    # Define rename mapping
    stim_map = {
        'medial_right': 'medial',
        'lateral_right': 'lateral',
        'medial_left': 'medial',
        'lateral_left': 'lateral',
        'left': 'binocular',
        'right': 'binocular'
    }

    # Flip sign for the left-side versions
    flip_stimuli = {'medial_left', 'lateral_left', 'left'}
    if column_name is not None: # only if you need to flip the sign of the data
        df_copy.loc[df_copy['tail_stimuli'].isin(flip_stimuli), column_name] *= -1

    # Apply renaming for all cases
    df_copy['tail_stimuli'] = df_copy['tail_stimuli'].replace(stim_map)

    return df_copy

def combine_shearing_stims_for_tail_df(df, column_name = 'tail_angle'):
    '''
    Combine all the binocular stimuli together
    Requires flipping all the left tail angles
    '''
    df_copy = df.copy()

    # Define rename mapping
    stim_map = {
        'forward_x': 'forward_x',
        'x_forward': 'forward_x',
        'backward_x': 'x_backward',
        'x_backward': 'x_backward',
        'forward_backward': 'forward_backward',
        'backward_forward':'forward_backward'
    }

    # Flip sign for the counter-clockwise versions
    flip_stimuli = {'x_forward', 'backward_x', 'backward_forward'}
    if column_name is not None: # only if you need to flip the sign of the data
        df_copy.loc[df_copy['tail_stimuli'].isin(flip_stimuli), column_name] *= -1

    # Apply renaming for all cases
    df_copy['tail_stimuli'] = df_copy['tail_stimuli'].replace(stim_map)

    return df_copy


def kaitlyn_pandastim_to_df(
    pstim_path,
):  # bc KF had a messed up stimulus file in 20220819 expt
    with open(pstim_path) as file:
        contents = file.read()

    lines = contents.split("\n")

    motionOns = [i for i in lines if "motionOn" in i.split("_&_")[-1]]
    times = [i.split("_&_")[0] for i in motionOns]
    stims = [eval(i[i.find("{") :]) for i in motionOns]
    stimulus_only = [i["stimulus"] for i in stims]

    stimulus_df = pd.DataFrame(stimulus_only)
    stimulus_df.loc[:, "datetime"] = times
    stimulus_df.datetime = pd.to_datetime(stimulus_df.datetime)
    stimulus_df.loc[:, "time"] = [
        pd.Timestamp(i).time() for i in stimulus_df.datetime.values
    ]

    mini_stim = stimulus_df[["stim_name", "angle", "time"]]
    mini_stim.stim_name = pd.Series(mini_stim.stim_name, dtype="category")

    angles = {
        "forward": 0,
        "right": 90,
        "backward": 180,
        "left": 270,
        "forward_left": 320,
        "backward_left": 220,
        "backward_right": 150,
        "forward_right": 40,
    }

    mini_stim["name"] = mini_stim.angle.map(numToStim(angles))

    mini_stim.rename(
        columns={"stim_name": "stimulus", "name": "stim_name"}, inplace=True
    )
    mini_stim = mini_stim[["stim_name", "time"]]

    return mini_stim

def validate_stims(stim_df, f_cells):
    stim_frames = stim_df.frame.values
    img_len = f_cells.shape[1]

    if img_len < stim_frames[-1]:

        frame_len = stim_frames[stim_frames < img_len]
        stim_df = stim_df.loc[: stim_df.loc[stim_df["frame"] == frame_len[-2]].index[0]]
        stim_df = stim_df.iloc[:-1]
    else:
        pass

    return stim_df

def label_stim_ax(plot, stim_keys, fs=12, offset=10, stim_offset=6):

    import constants

    y_top = round(max(plot.get_ylim()))
    ylabel_pos = y_top + y_top * 0.05

    x_top = round(max(plot.get_xlim()))
    xlabel_pos = x_top + x_top * 0.05
    xlabel_pos_pct = xlabel_pos / x_top

    for stim, start_pos in stim_keys.items():
        plot.text(start_pos + 2, ylabel_pos, constants.stim_title_dict[stim], fontsize=fs)

        begin = start_pos + offset - 2
        end = start_pos + offset + stim_offset//2
        midpt = begin + (end - begin) // 2


        if stim in constants.monocular_dict.keys():
            plot.axvspan(begin, end, alpha=0.4, color=constants.monocular_dict[stim])


        else:
            if stim == "lateral_left":
                plot.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                plot.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
            if stim == "medial_left":
                plot.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plot.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stim == "lateral_right":
                plot.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                plot.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )
            if stim == "medial_right":
                plot.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                plot.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")

            if stim == "converging":
                plot.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                plot.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stim == "diverging":
                plot.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                plot.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )

