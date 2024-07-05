import pandas as pd
import numpy as np
from numpy import nan #important for stimuli fxns
from datetime import datetime


def pandastim_to_df(pstimpath, minimode=False, addvelocity=True):
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

def csv_to_df(csvpath, minimode=False, addvelocity=True):
    stimulus_df = pd.read_csv(csvpath)
    stimulus_df.drop(columns=['Unnamed: 0'], inplace=True)

    strings = stimulus_df.time.values
    datetime_full = [datetime.strptime(string, '%H:%M:%S.%f') for string in strings]
    datetime_times = [dt.time() for dt in datetime_full]
    stimulus_df['time'] = datetime_times # convert times to actual datetime objects
    
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

def stimulus_start_frames_for_plots(frames_motion_on = 7, length_of_total_frame_arr = 21, number_of_stims_in_set = 8):
    '''
    frames_motion_on -- the first frame that motion is on in the series of frames you want to shade
    length_of_total_frame_arr -- the total number of frames that is taken from the neural trace before and after the stimulus is on (typically diff between offsets, i.e. 21)
    number_of_stims_in_set -- the number of stimuli in the experiment (8 for the 8 barcoded stimuli)

    returns a list of the starting frames for each stimulus in the set, this is what starts the shading in the plots
    '''
    stim_start_frames = []
    start_value = frames_motion_on - 1
    for _ in range(number_of_stims_in_set):
        stim_start_frames.append(start_value)
        start_value += length_of_total_frame_arr

    return stim_start_frames

def flexible_stim_shader(frames, stimmies, frames_motion_on, fs = 14, subplot = None, label = True, ylim = (-3, 3), label_offset_x = -6):
    import constants
    import matplotlib.pyplot as plt

    if subplot == None:
        ax_n = plt
        y_top = round(max(plt.gca().get_ylim()))
        x_top = round(max(plt.gca().get_xlim()))
    else:
        ax_n = subplot
        y_top = round(max(ax_n.get_ylim()))
        x_top = round(max(ax_n.get_xlim()))

    y_top = ylim[1]
    ylabel_pos = y_top + y_top * 0.05

    for s, stimmy in zip(frames, stimmies):
        if label:
            ax_n.text(s + label_offset_x, ylabel_pos, constants.stim_title_dict[stimmy], fontsize=fs)

        begin = s
        end = s + frames_motion_on
        midpt = begin + (end - begin) // 2

        if stimmy in constants.monocular_dict.keys():
            ax_n.axvspan(begin, end, color=constants.monocular_dict[stimmy], alpha=0.4,)

        elif stimmy in constants.baseBinocs:

            if stimmy == "lateral_left":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                ax_n.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
            if stimmy == "medial_left":
                ax_n.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stimmy == "lateral_right":
                ax_n.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )
            if stimmy == "medial_right":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                ax_n.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")

            if stimmy == "converging":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["right"], alpha=0.4
                )
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["left"], alpha=0.4
                )

            if stimmy == "diverging":
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["left"], alpha=0.4
                )
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["right"], alpha=0.4
                )
        else:

            if stimmy == "x_forward":
                ax_n.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=0.4
                )

            if stimmy == "forward_x":
                ax_n.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=0.4
                )

            if stimmy == "x_backward":
                ax_n.axvspan(begin, midpt, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "backward_x":
                ax_n.axvspan(midpt, end, color="gray", alpha=0.4, hatch=r"\\\\")
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "backward_forward":
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["forward"], alpha=0.4
                )
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["backward"], alpha=0.4
                )

            if stimmy == "forward_backward":
                ax_n.axvspan(
                    midpt, end, color=constants.monocular_dict["backward"], alpha=0.4
                )
                ax_n.axvspan(
                    begin, midpt, color=constants.monocular_dict["forward"], alpha=0.4
                )

def numToStim(dict):
    reverse_dict = {value: key for key, value in dict.items()}
    return reverse_dict

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

