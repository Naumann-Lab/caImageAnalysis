import pandas as pd
from numpy import nan
import numpy as np


def pandastim_to_df(pstimpath, minimode=True):
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
    if minimode:
        return mini_stim
    else:
        return stimulus_df


def numToStim (dict):
    reverse_dict = {value: key for key, value in dict.items()}
    return reverse_dict
def kaitlyn_pandastim_to_df(pstim_path): #bc KF had a messed up stimulus file in 20220819 expt
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

    angles = {"forward": 0,
              "right": 90,
              "backward": 180,
              "left": 270,
              "forward_left": 320,
              "backward_left": 220,
              "backward_right": 150,
              "forward_right": 40}

    mini_stim['name'] = mini_stim.angle.map(numToStim(angles))

    mini_stim.rename(columns={'stim_name': 'stimulus', 'name': 'stim_name'}, inplace=True)
    mini_stim = mini_stim[["stim_name", "time"]]

    return mini_stim