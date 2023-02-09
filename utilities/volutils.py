import os

import pandas as pd
import numpy as np

from tifffile import imread, imsave
from tqdm.auto import tqdm
from pathlib import Path
from datetime import datetime as dt


def legacy_volumesplit(unsplit_img_path, frametimes_df, len_thresh=150, crop=False):

    image = imread(unsplit_img_path)

    if crop:
        image = image[:, :, image.shape[1] // 10 :]

    for n, s in enumerate(tqdm(frametimes_df.step.unique())):
        imgInds = frametimes_df[frametimes_df.step == s].index

        sub_img = image[imgInds]
        if len(sub_img) >= len_thresh:
            subStackPath = Path(unsplit_img_path).parents[0].joinpath(f"img_stack_{n}")
            if not os.path.exists(subStackPath):
                os.mkdir(subStackPath)

            subStackFtPath = subStackPath.joinpath("frametimes.h5")
            if os.path.exists(subStackFtPath):
                os.remove(subStackFtPath)

            if 0 in frametimes_df.columns:
                frametimes_df.rename(columns={0: "time"}, inplace=True)

            frametimes_df.loc[imgInds].to_hdf(subStackFtPath, key="frametimes")
            subStackImgPath = subStackPath.joinpath("image.tif")
            imsave(subStackImgPath, sub_img)
    del image


def legacy_raw_text_frametimes_to_df(time_path):

    """
    Parameters
    ----------
    time_path : TYPE path
        DESCRIPTION. path to the frame times (txt) collected by the imaging software

    Returns
    -------
    TYPE dataframe
        DESCRIPTION. raw data frame times will be listed in datetime format
    """
    with open(time_path) as file:
        contents = file.read()
    parsed = contents.split("\n")

    times = []
    for line in range(len(parsed) - 1):
        times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
    return pd.DataFrame(times)


def legacy_raw_text_logfile_to_df(log_path):
    """
    Parameters
    ----------
    log_path : TYPE path
        DESCRIPTION. path to the log txt file from imaging software, contains steps

    Returns
    -------
    log_steps : TYPE dataframe
        DESCRIPTION. raw data log txt is filtered, only have the times and steps when the piezo moved
    """
    with open(log_path) as file:
        contents = file.read()
    split = contents.split("\n")

    movesteps = []
    times = []
    for line in range(len(split)):
        if (
            "piezo" in split[line]
            and "connected" not in split[line]
            and "stopped" not in split[line]
        ):
            t = split[line].split(" ")[0][:-1]
            z = split[line].split(" ")[6]
            try:
                if isinstance(eval(z), float):
                    times.append(dt.strptime(t, "%H:%M:%S.%f").time())
                    movesteps.append(z)
            except NameError:
                continue
    else:
        # last line is blank and likes to error out
        pass
    log_steps = pd.DataFrame({"times": times, "steps": movesteps})
    return log_steps


def legacy_alignmentFramesSteps(
    frametimes, logtimes, intermediate_return=False, time_offset=0.1
):
    """
    Parameters
    ----------
    frametimes : TYPE dataframe
        DESCRIPTION. times that frames were taken, converted from the raw_text_frametimes_to_df function
    logtimes : TYPE dataframe
        DESCRIPTION. times and steps (um) when the piezo moved through the image collection, converted in the raw_text_logfile_to_df function

    Returns
    -------
    frametimes : TYPE dataframe, modified from the raw_text_logfile_to_df frametimes
        DESCRIPTION. contains the aligned steps from the log file to the times that the frames were collected
    """

    ## milliseconds off between the log/step information and frametimes time stamp
    logtimes_mod = []  ## modified logtimes list
    missed_steps = []

    for t in range(len(frametimes)):
        listed_time = str(frametimes.values[t][0]).split(":")
        time_val = float(listed_time[-1])

        seconds_min = time_val - time_offset
        seconds_max = time_val + time_offset
        # clip function to make sure the min is 0, no negative times
        seconds_min = np.clip(seconds_min, a_min=0, a_max=60)

        min_listed_time = listed_time.copy()
        min_listed_time[-1] = str(np.float32(seconds_min))

        if seconds_max >= 60:
            m = int(listed_time[1])

            if m + 1 >= 60:
                h = int(listed_time[0])

                h += 1

                fin_time = listed_time.copy()
                fin_time[0] = str(int(h))
                fin_time[1] = str(int(m))
                fin_time[2] = str(np.float32(0.00))

            else:
                fin_time = listed_time.copy()
                fin_time[1] = str(int(m))
                fin_time[2] = str(np.float32(0.00))
        else:
            fin_time = listed_time.copy()
            fin_time[2] = str(np.float32(seconds_max))

        max_listed_time = fin_time

        mintime = dt.strptime(":".join(min_listed_time), "%H:%M:%S.%f").time()
        maxtime = dt.strptime(":".join(max_listed_time), "%H:%M:%S.%f").time()

        temp = logtimes[(logtimes.times >= mintime) & (logtimes.times <= maxtime)]

        ## sometimes there are missed steps (no frame with the next step in the stack) so we need to take that out
        if len(temp) != 0:
            logtimes_mod.append(temp)
        else:
            missed_steps.append(t)
    ## this is a check here, so if intermediate_return is true, then it will stop here and return the frametimes and logtimes_mod dataframes
    if intermediate_return:
        return frametimes, logtimes_mod

    frametimes_with_steps = []
    for df_row in logtimes_mod:
        frametimes_with_steps.append(df_row.steps.values[0])

    frametimes.drop(missed_steps, inplace=True)
    frametimes.loc[:, "step"] = frametimes_with_steps
    frametimes.loc[:, "step"] = frametimes.step.astype(np.float32)
    return frametimes


def parseFolder(folderpath, key=".txt"):
    pathDict = {}
    with os.scandir(folderpath) as entries:
        for entry in entries:
            if "log" in entry.name:
                pathDict["log"] = entry.path
            elif ".tif" in entry.name:
                pathDict["image"] = entry.path
            elif key in entry.name:
                pathDict["frametimes"] = entry.path

    return pathDict
