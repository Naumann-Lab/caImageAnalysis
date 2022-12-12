import pandas as pd


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
