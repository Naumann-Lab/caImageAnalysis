from nptdms import TdmsFile
import numpy as np
from datetime import datetime as dt


def dateToMillisec(datetime):
    return (
        datetime.microsecond / 1000
        + datetime.second * 1000
        + datetime.minute * 60 * 1000
        + datetime.hour * 60 * 60 * 1000
    )


def tail_reader(tail_path, daylight = True):
    # reads in the tail data into a df
    tail_data = TdmsFile(tail_path)
    tail_df = tail_data.as_dataframe()
    tail_df = tail_df[tail_df["/'TailLoc'/'Time'"].notna()]
    tail_df.loc[:, "t"] = (
        tail_df["/'TailLoc'/'Time'"].values - tail_df["/'TailLoc'/'Time'"].values[0]
    )

    t_arr = []
    for t in range(len(tail_df.t.values)):
        t_arr.append(np.timedelta64(tail_df.t.values[t], "ms").astype(int))
    tail_df["t"] = t_arr
    tail_df["/'TailLoc'/'Time'"] = tail_df["/'TailLoc'/'Time'"].dt.tz_localize(
        "US/Eastern"
    )

    # add extra column at the end with the converted time
    tail_ts = []
    for i in range(len(tail_df)):
        try:
            val = dt.strptime(
                str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
                "%H:%M:%S.%f",
            ).time()
        except:
            val = dt.strptime(
                str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
                "%H:%M:%S",
            ).time()
        tail_ts.append(val)
    tail_df.loc[:, "conv_t"] = tail_ts

    tail_t = []
    tailTimes = tail_df["conv_t"].values
    # tail times
    # might need to change this tailTimes hour depending on when you do this expt
    if daylight = False:
        for i in range(len(tailTimes)):
            tailTimes[i] = tailTimes[i].replace(
                hour=tailTimes[i].hour - 5,
                minute=tailTimes[i].minute,
                second=tailTimes[i].second,
                microsecond=tailTimes[i].microsecond,
            )
    else:
        for i in range(len(tailTimes)):
            tailTimes[i] = tailTimes[i].replace(
                hour=tailTimes[i].hour - 4, # need to change this number if not in daylight savings
                minute=tailTimes[i].minute,
                second=tailTimes[i].second,
                microsecond=tailTimes[i].microsecond,
            )
        tail_t.append(dateToMillisec(tailTimes[i]))
    new_tail_t = np.asarray(tail_t)

    tail_df = tail_df.iloc[1:]

    return tail_df
