# from nptdms import TdmsFile
import numpy as np
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import pathlib as Path



def bhvr_log_to_df(bhvr_log_path, metadata_log_path):
    '''
    turning stytra behavior log into a pandas dataframe with datetime stamps
    bhvr_log_path: path to the behavior log
    metadata_log_path: path to the metadata log
    '''
    
    bhvr_df = pd.read_csv(bhvr_log_path, sep=";", dtype=np.float32)
    bhvr_df.drop(bhvr_df.columns[0], axis=1, inplace = True)

    with open(metadata_log_path, 'r') as json_file:
        metadata = json.load(json_file)
        start_dt_str = metadata['general']['t_protocol_start'].split('T')[1]

    start_dt = datetime.strptime(start_dt_str, "%H:%M:%S.%f")
    tail_dts = []
    for seconds in bhvr_df.t.values:
        microseconds = int(seconds * 1e6)
        new_datetime = start_dt + timedelta(microseconds=microseconds)
        tail_dts.append(new_datetime.time())
    bhvr_df['t_dt'] = tail_dts
    
    return bhvr_df

def tail_df_creator(bhvr_data_folder, saving = True):
    '''
    Creating the tail data dataframe from either single or multiple experiments in the same folder
    bhvr_data_folder: path to the folder with the behavior data
    saving: whether to save the dataframe or not, will be saved in the parents folder path
    '''
    from utilities import pathutils

    behavior_log_paths = pathutils.pathcrawler(bhvr_data_folder, set(), [], mykey = 'behavior_log')
    metadata_paths = pathutils.pathcrawler(bhvr_data_folder, set(), [], mykey = 'metadata')
            
    bhvr_df_lst = []
    for r in range(len(behavior_log_paths)):
        bhvr_df = bhvr_log_to_df(behavior_log_paths[r], metadata_paths[r])
        bhvr_df_lst.append(bhvr_df)
    
    if len(bhvr_df_lst) > 1:
        all_tail_data = pd.concat(bhvr_df_lst)
        all_tail_data.reset_index(drop = True, inplace = True)

        # resetting the t values
        transition_idx = []
        for i in range(1, len(all_tail_data.t.values)):
            if all_tail_data.t.values[i - 1] > all_tail_data.t.values[i]: # compare current value to previous value
                transition_idx.append(i)

        new_t_arr = all_tail_data.t.values[:]
        for e, idx in enumerate(transition_idx):
            previous_value = all_tail_data.t.values[idx - 1] # seconds
            pause_time = dateToMillisec(all_tail_data.t_dt.values[idx+1]) - dateToMillisec(all_tail_data.t_dt.values[idx])  # getting the pause between experiments (in milliseconds)
            if idx == transition_idx[-1]:
                next_idx = len(all_tail_data.t.values)
            else:
                next_idx = transition_idx[e+1]
            for i in range(idx, len(all_tail_data.t.values[idx:next_idx])):
                new_t_arr[i] += previous_value + (pause_time/1000) # pause_time in milliseconds, but adding to a seconds value

        all_tail_data.drop('t', axis = 1, inplace = True) 
        all_tail_data['t'] = new_t_arr
    else:
        all_tail_data = bhvr_df
    
    if saving:
        all_tail_data.to_hdf(bhvr_data_folder.joinpath('tail_df.h5'), key='tail')
        print('saved tail dataframe')

def dateToMillisec(datetime):
    '''
    Changes datetime object into milliseconds
    datetime: datetime object
    returns milliseconds
    '''
    return (
        datetime.microsecond / 1000
        + datetime.second * 1000
        + datetime.minute * 60 * 1000
        + datetime.hour * 60 * 60 * 1000
    )




# reads in the tail data into a df --> custom 2p way

# def tail_reader(tail_path):
#     
#     tail_data = TdmsFile(tail_path)
#     tail_df = tail_data.as_dataframe()
#     tail_df = tail_df[tail_df["/'TailLoc'/'Time'"].notna()]
#     tail_df.loc[:, "t"] = (
#         tail_df["/'TailLoc'/'Time'"].values - tail_df["/'TailLoc'/'Time'"].values[0]
#     )

#     t_arr = []
#     for t in range(len(tail_df.t.values)):
#         t_arr.append(np.timedelta64(tail_df.t.values[t], "ms").astype(int))
#     tail_df["t"] = t_arr
#     tail_df["/'TailLoc'/'Time'"] = tail_df["/'TailLoc'/'Time'"].dt.tz_localize(
#         "US/Eastern"
#     )

#     # add extra column at the end with the converted time
#     tail_ts = []
#     for i in range(len(tail_df)):
#         try:
#             val = dt.strptime(
#                 str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
#                 "%H:%M:%S.%f",
#             ).time()
#         except:
#             val = dt.strptime(
#                 str(tail_df["/'TailLoc'/'Time'"].iloc[i]).split(" ")[1].split("-")[0],
#                 "%H:%M:%S",
#             ).time()
#         tail_ts.append(val)
#     tail_df.loc[:, "conv_t"] = tail_ts

#     converted_tail_times = []
#     tail_times = tail_df["conv_t"].values

#     # converted time needs to be changed by this hour value given by lab view data
#     add_hour = (
#         str(tail_df["/'TailLoc'/'Time'"].iloc[0])
#         .split(" ")[1]
#         .split("-")[1]
#         .split(":")[0]
#     )

#     for i in range(len(tail_times)):
#         tail_times[i] = tail_times[i].replace(
#             hour=tail_times[i].hour - int(add_hour),
#             minute=tail_times[i].minute,
#             second=tail_times[i].second,
#             microsecond=tail_times[i].microsecond,
#         )
#     converted_tail_times.append(
#         dateToMillisec(tail_times[i])
#     )  # convert to milliseconds

#     new_tail_t = np.asarray(converted_tail_times)
#     tail_df = tail_df.iloc[1:]

#     return tail_df
