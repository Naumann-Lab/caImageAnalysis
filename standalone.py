import os

import numpy as np
import pandas as pd

from datetime import datetime as dt
from pathlib import Path


def norm_fdff(f_cells):
    minVals = np.percentile(f_cells, 10, axis=1)
    zerod_arr = np.array([np.subtract(f_cells[n], i) for n, i in enumerate(minVals)])
    normed_arr = np.array([np.divide(arr, arr.max()) for arr in zerod_arr])
    return normed_arr


def load_suite2p(folderPath):

    datapaths = {}
    s2pPath = Path(folderPath).joinpath("suite2p")

    datapaths["iscell"] = s2pPath.joinpath("plane0/iscell.npy")
    datapaths["stats"] = s2pPath.joinpath("plane0/stat.npy")
    datapaths["ops"] = s2pPath.joinpath("plane0/ops.npy")
    datapaths["f_cells"] = s2pPath.joinpath("plane0/F.npy")

    ops = np.load(datapaths["ops"], allow_pickle=True).item()
    iscell = np.load(datapaths["iscell"], allow_pickle=True)[:, 0].astype(bool)
    stats = np.load(datapaths["stats"], allow_pickle=True)
    f_cells = np.load(datapaths["f_cells"])
    return ops, iscell, stats, f_cells


def pandastim_to_df(pstimpath):
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
    return stimulus_df, mini_stim


def raw_text_frametimes_to_df(time_path):
    with open(time_path) as file:
        contents = file.read()
    parsed = contents.split("\n")

    times = []
    for line in range(len(parsed) - 1):
        times.append(dt.strptime(parsed[line], "%H:%M:%S.%f").time())
    times_df = pd.DataFrame(times)
    times_df.rename({0: "time"}, axis=1, inplace=True)
    return times_df


def return_seconds(timeObject):

    return (
        timeObject.hour * 60 * 60
        + timeObject.second
        + timeObject.minute * 60
        + timeObject.microsecond / 1000000
    )


def chunkSplitter(df, splitTime=60):
    chunks = []

    secondsList = [return_seconds(i) for i in df.time]
    secondsListDiff = abs(np.diff(secondsList))
    secondsBrkPoints = np.where(abs(secondsListDiff) > splitTime)[0]

    myChunk = df.loc[0 : secondsBrkPoints[0]]
    chunks.append(myChunk)

    for n, ind in enumerate(secondsBrkPoints):
        if n == len(secondsBrkPoints) - 1:
            myChunk = df.loc[ind + 1 :]
        else:
            myChunk = df.loc[ind + 1 : secondsBrkPoints[n + 1]]
        chunks.append(myChunk)

    return chunks


def cardinal_pixelwise(image, stimuli, offsets=(5, 20), brighterFactor=1.5, brighter=5):
    cardinals = {
        "forward": [0, 1, 0],
        "forward_left": [0, 0.75, 1],
        "left": [0, 0.25, 1],
        "backward_left": [0.25, 0, 1],
        "backward": [1, 0, 1],
        "backward_right": [1, 0, 0.25],
        "right": [1, 0.25, 0],
        "forward_right": [0.75, 1, 0],
    }

    diff_imgs = {}
    for stimulus_name in cardinals.keys():
        _stims = stimuli[stimuli.stim_name == stimulus_name]
        _img = []
        for ind in _stims.frame.values:
            img = np.nanmean(image[ind + offsets[0] : ind + offsets[1]], axis=0)
            bg = np.nanmean(image[ind - offsets[1] // 2 : ind + offsets[0]], axis=0)
            _img.append(img - bg)

        diff_img = np.mean(_img, axis=0)

        diff_imgs[stimulus_name] = diff_img

    maxVal = np.max([np.max(i) for i in diff_imgs.values()])

    imgs = []
    for name, image in diff_imgs.items():
        image[image < 0] = 0

        r = image * cardinals[name][0]
        g = image * cardinals[name][1]
        b = image * cardinals[name][2]

        r /= maxVal
        g /= maxVal
        b /= maxVal

        r -= r.min()
        g -= g.min()
        b -= b.min()
        imgs.append(
            np.dstack((r**brighterFactor, g**brighterFactor, b**brighterFactor))
        )

    somenewmaxval = np.max(imgs)

    _all_img = []
    for img in imgs:
        _all_img.append(img / somenewmaxval)

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)
    return fin_img * brighter


def movement_correction(img_path, keep_mmaps=False, inputParams=None):
    import caiman as cm

    defaultParams = {
        "max_shifts": (3, 3),
        "strides": (25, 25),
        "overlaps": (15, 15),
        "num_frames_split": 150,
        "max_deviation_rigid": 3,
        "pw_rigid": False,
        "shifts_opencv": True,
        "border_nan": "copy",
        "downsample_ratio": 0.2,
    }
    if inputParams:
        for key, val in inputParams.items():
            defaultParams[key] = val
    try:
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend="local", n_processes=12, single_thread=False
        )
        mc = cm.motion_correction.MotionCorrect(
            [img_path.as_posix()],
            dview=dview,
            max_shifts=defaultParams["max_shifts"],
            strides=defaultParams["strides"],
            overlaps=defaultParams["overlaps"],
            max_deviation_rigid=defaultParams["max_deviation_rigid"],
            shifts_opencv=defaultParams["shifts_opencv"],
            nonneg_movie=True,
            border_nan=defaultParams["border_nan"],
            is3D=False,
        )

        mc.motion_correct(save_movie=True)
        # m_rig = cm.load(mc.mmap_file)
        bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
        mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
        mc.template = (
            mc.mmap_file
        )  # use the template obtained before to save in computation (optional)
        mc.motion_correct(save_movie=True, template=mc.total_template_rig)
        m_els = cm.load(mc.fname_tot_els)

        output = m_els[
            :,
            2 * bord_px_rig : -2 * bord_px_rig,
            2 * bord_px_rig : -2 * bord_px_rig,
        ]

        # imagePathFolder = Path(imagePath).parents[0]
        if not keep_mmaps:
            with os.scandir(img_path.parents[0]) as entries:
                for entry in entries:
                    if entry.is_file():
                        if entry.name.endswith(".mmap"):
                            os.remove(entry)
        dview.terminate()
        cm.stop_server()
        return output

    except Exception as e:
        print(e)
        try:
            dview.terminate()
        except:
            pass
        cm.stop_server()


def hzReturner(frametimes):
    increment = 15
    test0 = 0
    test1 = increment
    while True:
        testerBool = (
            frametimes.loc[:, "time"].values[test0].minute
            == frametimes.loc[:, "time"].values[test1].minute
        )
        if testerBool:
            break
        else:
            test0 += increment
            test1 += increment

        if test0 >= len(frametimes):
            increment = increment // 2
            test0 = 0
            test1 = increment

    times = [
        float(str(f.second) + "." + str(f.microsecond))
        for f in frametimes.loc[:, "time"].values[test0:test1]
    ]
    return 1 / np.mean(np.diff(times))


def run_suite2p(imagepath, input_tau=1.5, image_hz=2):
    from suite2p.run_s2p import run_s2p, default_ops

    s2p_ops = {
        "data_path": [imagepath.parents[0].as_posix()],
        "save_path0": imagepath.parents[0].as_posix(),
        "tau": input_tau,
        "preclassify": 0.15,
        "allow_overlap": True,
        "block_size": [50, 50],
        "fs": image_hz,
    }

    ops = default_ops()
    db = {}
    for item in s2p_ops:
        ops[item] = s2p_ops[item]

    output_ops = run_s2p(ops=ops, db=db)
    return
