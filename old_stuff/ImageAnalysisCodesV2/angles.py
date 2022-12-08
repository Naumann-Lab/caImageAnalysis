from cmath import rect, phase
from math import radians, degrees

import numpy as np


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

monocStims = [
    "right",
    "left",
    "forward",
    "backward",
    "forward_left",
    "forward_right",
    "backward_left",
    "backward_right",
]

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
        _sums.append(weights[d] * rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums) / np.sum(weights)))


def polarVals(fishy, used_offsets, stim_offset):

    all_thetas = {}
    all_thetavals = {}
    for g, vol in enumerate(fishy.dataPaths["volumes"].keys()):
        nrnDict = {}

        for nrn in fishy.bool_dfs[int(vol)].index.values:
            if nrn not in nrnDict.keys():
                nrnDict[nrn] = {}

            for stim in monocStims:
                val = np.nanmedian(
                    fishy.stimdicts[vol]["meanArr"][stim][nrn][
                        -used_offsets[0] : -used_offsets[0] + stim_offset
                    ]
                )
                nrnDict[nrn][stim] = val

        thetas = []
        thetavals = []
        for n in nrnDict.keys():
            degKeys = [deg_dict[i] for i in nrnDict[n].keys()]
            degs = np.clip(list(nrnDict[n].values()), a_min=0, a_max=56)

            theta = weighted_mean_angle(degKeys, degs)
            thetaval = np.nanmean(degs)

            thetas.append(theta)
            thetavals.append(thetaval)
        all_thetas[g] = thetas
        all_thetavals[g] = thetavals
    return all_thetas, all_thetavals


def calcDSI(fishy, used_offsets, stim_offset):

    fishy.zdiff_stimdicts()
    fishy.zdiff_booldf()
    new_dfs = [
        fishy.bool_dfs[i][fishy.bool_dfs[i][monocStims].sum(axis=1) > 0]
        for i in range(len(fishy.bool_dfs))
    ]

    dsis = []
    for _, vol in enumerate(fishy.dataPaths["volumes"].keys()):
        for nrn in new_dfs[_].index.values:
            nrnDict = {}
            if nrn not in nrnDict.keys():
                nrnDict[nrn] = {}
            for stim in fishy.stimdicts[str(vol)]["meanArr"].keys():
                if stim in monocStims:
                    val = np.nanmedian(
                        fishy.stimdicts[str(vol)]["meanArr"][stim][nrn][
                            -used_offsets[0] : -used_offsets[0] + stim_offset
                        ]
                    )
                    nrnDict[nrn][stim] = val

            peakstim = max(nrnDict[nrn])
            nullstim = nulldict[peakstim]

            peakval = max(nrnDict[nrn].values())
            bdict = {v: k for k, v in nrnDict[nrn].items()}

            nullval = nrnDict[nrn][nulldict[bdict[peakval]]]
            # nullval = np.clip(nullval, a_min=0, a_max=5)
            # peakval = np.clip(peakval, a_min=0, a_max=5)

            dsi = (peakval - nullval) / (peakval + nullval)
            dsis.append(dsi)
    return dsis
