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

def find_tail_sum_std(tail_df):
    tail_hz = 1/np.mean(np.diff(tail_df.iloc[:100]['t']))
    smooth_tailframe = int(0.05 * tail_hz)
    std = [np.std(tail_df.tail_sum[i - smooth_tailframe:i + smooth_tailframe]) 
            for i in range(smooth_tailframe, len(tail_df.tail_sum) - smooth_tailframe)]
    tail_df['std'] =[0] * smooth_tailframe + std + [0] * smooth_tailframe
    return tail_df

def process_bouts_from_tail_df(tail_df, stimulus_df, imaging_hz = 30, strength_boundary = 0.1, cont_cutoff_s = 0.05, min_on_s = 0.05, stimulus_s = 10):
    '''
    tail_df = tail dataframe with imaging frames, t, and tail_sum columns
    strength_boundary = the standard deviation threshold for tail movement to be considered as a significant movement (default = 0.1)
    cont_cutoff_s = cutoff in seconds for the tail bouts to be considered as separate bouts (default = 0.05)
    min_on_s = minimum duration in seconds for a tail bout to be considered (default = 0.05)
    stimulus_df = visual motion stimulus dataframe (from a VizStimFish)
    stimulus_s = time in seconds that stimulus is on
    imaging_hz = imaging speed of the collected imaging data to pair with tail data
    '''

    tail_hz = 1/np.mean(np.diff(tail_df.iloc[:100]['t']))
    
    #calibrate to mean
    baseline = np.nanmean(tail_df.tail_sum)
    tail_df.tail_sum = np.subtract(tail_df.tail_sum, baseline)
    tail_df.tail_sum = tail_df.tail_sum.ffill()

    #collect positive and negative tail movement
    pos = np.where(tail_df.tail_sum > 0,tail_df.tail_sum, 0)
    neg = np.where(tail_df.tail_sum < 0,tail_df.tail_sum, 0)

    # group/smooth by running window  of ~100ms
    smooth_tailframe = int(0.05 * tail_hz)
    std = [np.std(tail_df.tail_sum[i - smooth_tailframe:i + smooth_tailframe]) 
        for i in range(smooth_tailframe, len(tail_df.tail_sum) - smooth_tailframe)]
    tail_df['std'] =[0] * smooth_tailframe + std + [0] * smooth_tailframe
    bout_on = tail_df['std']> strength_boundary
    bout_on = [int(x) for x in bout_on]
    on_index = np.where(np.diff(bout_on) == 1)[0]
    on_index = [i + smooth_tailframe for i in on_index]
    off_index = np.where(np.diff(bout_on) == -1)[0]
    off_index = [i + smooth_tailframe for i in off_index]
    if len(on_index) != 0 and len(off_index) != 0:
        if on_index[0] > off_index[0]:
            on_index = np.concatenate([[0], on_index])
        if on_index[-1] > off_index[-1]:
            off_index = np.concatenate([off_index, [len(tail_df) - 1]])
        on_tuples = [(on, off) for on, off in zip(on_index, off_index) if off - on < len(tail_df) and off > on]

    #connect bouts that are too close together
    cont_on_index = []
    cont_off_index = []
    if len(on_tuples) > 0:
        cont_on_index = [on_tuples[0][0]]
        if len(on_tuples) > 1:
            big_interval = np.array([on_tuples[i][0] - on_tuples[i - 1][1] for i in range(1, len(on_tuples))]) \
                        > (cont_cutoff_s * tail_hz)
            for i in range(0, len(big_interval)):
                if big_interval[i]:
                    cont_on_index = cont_on_index + [on_tuples[i + 1][0]]
                    cont_off_index = cont_off_index + [on_tuples[i][1]]
        cont_off_index = cont_off_index + [on_tuples[-1][1]]
    cont_tuples = [(on, off) for on, off in zip(cont_on_index, cont_off_index) if off - on > (min_on_s * tail_hz)]

    #calculate the actual frame(approx.) and image onset index/frames
    cont_tuples_imageframe = [(tail_df.iloc[tu[0]].frame, tail_df.iloc[tu[1]].frame) for tu in cont_tuples]

    #calculate actual tail parameters
    mean_tail_angle = np.full(len(cont_tuples), np.nan)
    cumul_tail_angle = np.full(len(cont_tuples), np.nan)
    max_tail_angle = np.full(len(cont_tuples), np.nan)
    tail_duration_s = np.full(len(cont_tuples), np.nan)
    tail_stimuli = ['spontaneous'] *len(cont_tuples)
    for i in range(len(cont_tuples)):
        mean_tail_angle[i] = np.nanmean(tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        cumul_tail_angle[i] = np.sum(tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        max_tail_angle[i] = np.nanmax(tail_df.tail_sum[cont_tuples[i][0]:cont_tuples[i][1]])
        tail_duration_s[i] = np.divide((cont_tuples[i][1] - cont_tuples[i][0]), tail_hz)
        if not stimulus_df.empty:
            if cont_tuples_imageframe[i][0] > stimulus_df.iloc[0]['frame'] :
                stimulus_responding = stimulus_df[stimulus_df['frame'] <= cont_tuples_imageframe[i][0]].iloc[-1]#find the nearest stimuli before and see if tail happens within the stimulus
                if stimulus_responding['frame'] + stimulus_s * imaging_hz >= cont_tuples_imageframe[i][0]:#if tail starts before the stimulus ends
                    tail_stimuli[i] = stimulus_responding['stim_name']
    
    return cont_tuples, cont_tuples_imageframe, mean_tail_angle, max_tail_angle, cumul_tail_angle, tail_duration_s, tail_stimuli

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
