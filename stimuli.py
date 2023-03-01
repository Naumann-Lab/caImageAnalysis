import pandas as pd
from numpy import nan
import numpy as np


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


def legacy_struct_pandastim_to_df(folderPath, stimkey, *args, **kwargs):
    import os

    with os.scandir(folderPath.parents[0]) as entries:
        for entry in entries:
            if stimkey in entry.name:
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
        stim_df = stim_df.loc[: stim_df.loc[stim_df['frame'] == frame_len[-1]].index[0]]
        stim_df = stim_df.iloc[:-1]
    else:
        pass

    return stim_df
