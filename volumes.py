import pandas as pd
import numpy as np

import constants


def packVols(a_fish):
    """

    :param a_fish: some Fish Class describing a single experiment
    :return: dict of volumes : s2p stuff, dict of volumes : dataframes with stim chunks for each neuron
    """
    vols = a_fish.dataPaths["volumes"]
    s2p_dict = {vol: a_fish.load_suite2p(vols[vol]["suite2p"]) for vol in vols.keys()}
    stimArrs = {
        vol: pd.DataFrame(a_fish.stimdicts[str(vol)]["meanArr"])[
            constants.monocular_dict.keys()
        ]
        for vol in vols
    }
    return s2p_dict, stimArrs


def volumeValidity(a_fish, stimChoice):
    """
    :param a_fish: some Fish Class describing a single experiment
    :param stimChoice: stimulus of choice
    :return: dict of volumes : valid indices
    """
    return {
        vol: a_fish.bool_dfs[int(vol)][
            a_fish.bool_dfs[int(vol)][stimChoice] == True
        ].index
        for vol in a_fish.dataPaths["volumes"].keys()
    }


def grabPeakNeurons(a_fish, stimChoice, n=25):
    """

    :param a_fish: some Fish Class describing a single experiment
    :param stimChoice: stimulus of choice
    :param n: how many neurons
    :return: normalized traces for top n neurons
    """
    vols2p, volstims = packVols(a_fish)
    for vol, df in volstims.items():
        df.loc[:, "vol"] = vol
    df = pd.concat(volstims.values())

    df_inds = df.index.values
    df_vols = df["vol"].values

    max_dict = {}
    for c in df.iloc[:, :-1].columns:
        vals = [np.nanmax(i) for i in df[c].values]
        max_dict[c] = vals
    max_df = pd.DataFrame(max_dict)
    peakstims = max_df[constants.monocular_dict.keys()].idxmax(axis=1)
    max_inds = (
        max_df[(peakstims == stimChoice)]
        .sort_values(by=stimChoice, ascending=False)
        .index
    )

    valid_inds = volumeValidity(a_fish, stimChoice)

    # validate inds founds
    validated = []
    for m, i in enumerate(max_inds):
        used_vol = df_vols[i]
        vol_ind = df_inds[i]
        if vol_ind in valid_inds[used_vol]:
            validated.append(m)

    max_inds = max_inds[validated][:n]

    cell_arrs = []
    for i in range(n):
        used_vol = df_vols[max_inds[i]]
        vol_ind = df_inds[max_inds[i]]
        ops, iscell, stats, f_cells = vols2p[used_vol]
        norm_cells = a_fish.norm_fdff(f_cells)
        cell_arrs.append(norm_cells[vol_ind])
    return cell_arrs
