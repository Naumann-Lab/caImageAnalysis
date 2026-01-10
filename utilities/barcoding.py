import numpy as np
import pandas as pd
from pathlib import Path
import constants
from utilities import arrutils, clustering
import stimuli, angles
from fishy import WorkingFish

barcoding_8stim_order = [
    "converging","diverging",
    "left","medial_left","lateral_left",
    "right","medial_right","lateral_right"]

def get_stim_on_frames(somefishy, stim_set = barcoding_8stim_order, motion_on_frames = 7):
    '''
    somefishy -- has to be a VizStimFish with a stimulus_df
    stim_set -- a list of the stimuli you want to get the frames for
    motion_on_frames -- the number of frames the motion is on for (7 frames typicallY)

    getting all the frames for motion on into a dictionary, necessary for Whit's barcoding
    '''
    stim_frame_dict = {}
    for q in stim_set:
        start_frames = somefishy.stimulus_df[somefishy.stimulus_df['stim_name'] == q].frame.values
        stim_on_frame_list = []
        for k in start_frames:
            stim_on_frame_list.extend(list(range(k, k + motion_on_frames)))
        stim_frame_dict[q] = stim_on_frame_list
    
    return stim_frame_dict

def barcode_with_ideal_trace(vizstimfish, barcode_dict = constants.eva_typesL, n_reps = 3, stim_order = barcoding_8stim_order, 
                             frames_motion_on = 7, baseline_len = 8, length_of_total_frame_arr = None, std_thresh = 1.8,
                             response_type = 'median',trace_type = 'norm'):
    '''
    Identifying barcoded neurons with correlations to the 'ideal' trace
    vizstimfish -- a VizStimFish class object
    barcode_dict -- a dictionary with the barcode labels as keys and their binary codes as items (i.e. list of False or True in the order of the stimuli)
    n_reps -- the number of repetitions of the experiment
    stim_order -- the order of the stimuli in the average responses and then average array traces
    frames_motion_on -- the number of frames the motion is on for/how long the calcium response will be (7 frames typically), 
                        this will change depending on gcamp, fish, etc
    std_thresh -- the threshold for the standard deviation of the baseline activity when determining the responsitivity to the code

    '''
    # getting the responses of each cell to each repetition of every stimulus
    stim_resp_each_cell_arr = vizstimfish.neur_resps_each_stim_rep
    if stim_resp_each_cell_arr is None:
        stim_resp_each_cell_arr = WorkingFish.neuron_each_stim_rep_arrays(vizstimfish, vizstimfish.stim_order)
    
    if length_of_total_frame_arr is None:
        length_of_total_frame_arr = -vizstimfish.offsets[0] + vizstimfish.offsets[1]

    # define ideal fake traces for each barcode:
    ideal_barcorde_dict = {key: None for key in barcode_dict.keys()}
    for typ, binary_code in barcode_dict.items():
        ideal_arr = np.zeros(length_of_total_frame_arr*len(stim_order))
        for h, i in enumerate(binary_code):
            if i:
                ideal_arr[h*length_of_total_frame_arr + -vizstimfish.offsets[0] + 1: 
                          h*length_of_total_frame_arr + -vizstimfish.offsets[0] + frames_motion_on] = 1
        ideal_barcorde_dict[typ] = arrutils.pretty(ideal_arr, 3)

    corr_dict = {} # dictionary of all correlation values for each neuron to each barcode
    type_dict = {} # type of barcode for each neuron
    binary_codes_dict = {} # collecting all binary codes for each neuron to determine its forward responses
    cell_rois = vizstimfish.return_cell_rois(range(len(stim_resp_each_cell_arr)))
    x_midline = WorkingFish.return_x_midline(vizstimfish)
    for n, neuron_arr in enumerate(stim_resp_each_cell_arr):
        if n not in type_dict.keys():
            type_dict[n] = np.nan
            corr_dict[n] = np.nan
            binary_codes_dict[n] = {}

        # determining the binary code for this neuron
        # using just normalized values (not df/f)
        if trace_type == 'norm':
            neuron_binary_code = barcode_binary_score(vizstimfish, neuron_arr, base_length = baseline_len, frames_motion_on = frames_motion_on,
                                    std_thresh = std_thresh, num_responding_trials = int(n_reps*0.8), evoked_resp = response_type)
        elif trace_type == 'df/f':
            neuron_binary_code = barcode_binary_score_df_f(vizstimfish, neuron_arr, frames_motion_on = frames_motion_on,
                                        std_thresh = std_thresh, num_responding_trials = int(n_reps*0.8), evoked_resp = response_type)
        else: # default is norm
            neuron_binary_code = barcode_binary_score(vizstimfish, neuron_arr, base_length=baseline_len,
                                                      frames_motion_on=frames_motion_on,
                                                      std_thresh=std_thresh, num_responding_trials=int(n_reps * 0.8),
                                                      evoked_resp=response_type)
        binary_codes_dict[n] = neuron_binary_code
        for typ, l in barcode_dict.items():
            bool_val = True
            if (l == neuron_binary_code) & ('oMl' in typ):  # if the neuron's barcode matches one of those in the barcode dictionary
                bool_val = check_oMl_location(cell_rois[n], x_midline, typ) 
            if (l == neuron_binary_code) & ('Mm' in typ):  # if the neuron's barcode matches one of those in the barcode dictionary
                bool_val = check_barcoded_neur_location(cell_rois[n], x_midline, typ)
            if (l == neuron_binary_code) & (bool_val== True):
                type_dict[n] = typ
                mean_neuron_arr = np.nanmean(neuron_arr, axis = 0)[:len(ideal_barcorde_dict[typ])]
                # correlate ideal fake traces to the barcode that this neuron matches
                corr = np.corrcoef(arrutils.zscoring(ideal_barcorde_dict[typ]), arrutils.zscoring(mean_neuron_arr))[0, 1] 
                corr_dict[n] = corr
            
    return type_dict, corr_dict, binary_codes_dict

def barcode_binary_score(vizstimfish, one_neuron_arr, stims = None, stim_start_frames = None, base_length = 4, frames_motion_on = None, 
                         std_thresh = 1.8, num_responding_trials = 3, evoked_resp = 'median'):
    '''
    Create a binary code for each neuron
    vizstimfish -- a VizStimFish class object
    one_neuron_arr -- the normalized f activity of one neuron
    base_length -- the length of the baseline activity
    frames_motion_on -- the number of frames the motion is on for/how long the calcium response will be (7 frames typically), 
                        this will change depending on gcamp, fish, etc
    num_responding_trials -- the number of responding trials that the neuron needs to consistently respond to the motion
    std_thresh -- the threshold for the standard deviation of the baseline activity
    '''
    if frames_motion_on is None:
        frames_motion_on = int(vizstimfish.img_hz*vizstimfish.seconds_motion_is_on)
    if stims is None:
        stims = vizstimfish.stim_order
    if stim_start_frames is None:
        stim_start_frames = vizstimfish.stim_start_frames

    bool_dict_per_neuron = {key: 0 for key in stims}
    for e, l in enumerate(stim_start_frames):
        key = stims[e]
        base_arr = np.nanmedian(one_neuron_arr[:, (l-base_length):l - 1], axis = 1)
        base_std = np.nanstd(one_neuron_arr[:, (l-base_length):l - 1], axis = 1)
        if evoked_resp == 'median':
            evoked_arr = np.nanmedian(one_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        elif evoked_resp == 'max':
            evoked_arr = np.nanmax(one_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        elif evoked_resp == 'mean':
            evoked_arr = np.nanmean(one_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        count = 0
        for d in range(len(evoked_arr)):
            if evoked_arr[d] > (base_arr[d] + std_thresh*base_std[d]):
                count += 1
        if count >= num_responding_trials:
            bool_dict_per_neuron[key] = 1
    binary_code = [bool(v) for v in bool_dict_per_neuron.values()]

    return binary_code

def barcode_binary_score_df_f(vizstimfish, one_neuron_arr, stims = None, stim_start_frames = None, frames_motion_on = None, 
                         std_thresh = 1.8, num_responding_trials = 3, evoked_resp = 'mean'):
    '''
    Create a binary code for each neuron, using df/f instead of just raw fluorescence
    vizstimfish -- a VizStimFish class object
    one_neuron_arr -- the raw fluorescence array for one neuron
    frames_motion_on -- the number of frames the motion is on for/how long the calcium response will be (7 frames typically), 
                        this will change depending on gcamp, fish, etc
    num_responding_trials -- the number of responding trials that the neuron needs to consistently respond to the motion
    std_thresh -- the threshold for the standard deviation of the baseline activity

    Utilizing df/f from a baseline of the offsets provided for the fish class (i.e. -vizstimfish.offsets[0])
    '''
    if frames_motion_on is None:
        frames_motion_on = int(vizstimfish.img_hz*vizstimfish.seconds_motion_is_on)
    if stims is None:
        stims = vizstimfish.stim_order
    if stim_start_frames is None:
        stim_start_frames = vizstimfish.stim_start_frames

    bool_dict_per_neuron = {key: 0 for key in stims}
    for e, l in enumerate(stim_start_frames):
        key = stims[e]
        base_arr = np.nanmean(one_neuron_arr[:, l+vizstimfish.offsets[0]:l - 1], axis = 1)
        base_std = np.nanstd(one_neuron_arr[:, l+vizstimfish.offsets[0]:l - 1], axis = 1)
        df_f_neuron_arr = np.array([(arr - base_arr[i]) / base_arr[i] for i, arr in enumerate(one_neuron_arr)])
        if evoked_resp == 'median':
            evoked_arr = np.nanmedian(df_f_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        elif evoked_resp == 'max':
            evoked_arr = np.nanmax(df_f_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        elif evoked_resp == 'mean':
            evoked_arr = np.nanmean(df_f_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        count = 0
        for d in range(len(evoked_arr)):
            if evoked_arr[d] > (base_arr[d] + std_thresh*base_std[d]):
                count += 1
        if count >= num_responding_trials:
            bool_dict_per_neuron[key] = 1
    binary_code = [bool(v) for v in bool_dict_per_neuron.values()]

    return binary_code

def find_forward_responders(vizstimfish, stim_order = ['forward', 'backward'], 
                            frames_motion_on = None, std_thresh = 1.8, n_reps = 3, evoked_resp_type = 'mean'):
    '''
    stim_order needs to include forward and backward!
    '''
    frw_back_stim_resps = vizstimfish.neuron_each_stim_rep_arrays(stim_order)
    stim_start_frames = stimuli.stimulus_start_frames_for_plots(frames_motion_on = frames_motion_on, 
                                                            length_of_total_frame_arr = np.diff(vizstimfish.offsets)[0],
                                                            number_of_stims_in_set = len(stim_order))

    forward_responders = []
    backward_responders = []
    for n, neuron_arr in enumerate(frw_back_stim_resps):
        neuron_binary_code = barcode_binary_score_df_f(vizstimfish, neuron_arr, stims = stim_order, stim_start_frames = stim_start_frames,
                                                   frames_motion_on = frames_motion_on, 
                                                  std_thresh = std_thresh, num_responding_trials = int(n_reps*0.8), evoked_resp = evoked_resp_type)
        if neuron_binary_code[0] == True:
            forward_responders.append(n)
        if neuron_binary_code[1] == True:
            backward_responders.append(n)
    # reset the neuron response arrays back to the original order
    vizstimfish.neur_resps_each_stim_rep = vizstimfish.neuron_each_stim_rep_arrays(vizstimfish.stim_order)

    return forward_responders, backward_responders

def suppression_barcode_score(vizstimfish, cell_num, stims = constants.eva_stims[:8], frames_motion_on = None,
                              suppression_std_thresh = 1.8):
    '''
    Finding the suppression barcode score for a single neuron (with normalized values, it should not matter)
    :param vizstimfish: data fish
    :param cell_num: cell id that you want to calculate this for
    :param stims: stim order of choice
    :param frames_motion_on: the number of frames motion is on for, default is the seconds motion is on * img hz
    :param suppression_std_thresh: standard deviation of the baseline calculated for true or false suppression
    :return: list of binary 1, 0 where 1 is suppressed and 0 is not in order of the stimuli in the 'stims' parameter
    '''
    extended_resp = pd.DataFrame(vizstimfish.extended_responses_normf).iloc[cell_num][stims]
    if frames_motion_on is None:
        frames_motion_on = int(vizstimfish.img_hz * vizstimfish.seconds_motion_is_on)
    num_trials = max(vizstimfish.stimulus_df.rep) + 1

    suppression_barcode = []
    for n, each_stim_resp in enumerate(extended_resp.values):
        each_stim_resp = np.array(each_stim_resp)
        baseline_arr = each_stim_resp[:, :-vizstimfish.offsets[0]]
        baseline_mean = np.nanmean(baseline_arr, axis=1)
        baseline_std = np.std(baseline_arr, axis=1)
        # determining suppression barcode
        count = 0
        for e, each_trial in enumerate(each_stim_resp):
            evoked_arr = each_trial[-vizstimfish.offsets[0]:-vizstimfish.offsets[0] + frames_motion_on]
            if np.nanmean(evoked_arr) < (baseline_mean[e] - (suppression_std_thresh * baseline_std[e])):
                count += 1
        score = 1 if count >= int(0.8 * num_trials) else 0  # have to add 1 to the num trials since 0 indexing
        suppression_barcode.append(score)

    return suppression_barcode

def check_oMl_location(cell_roi, x_midline, oMl_type):
    '''
    Check if the oMl neuron is on the predicted side of the brain based on the cell's location
    '''
    if cell_roi[0] < x_midline: # cell on left hemisphere
        if oMl_type == 'oMl_R':
            return True
        else:
            return False
    elif cell_roi[0] > x_midline: # cell on right hemisphere
        if oMl_type == 'oMl_L':
            return True
        else:
            return False
        
def check_barcoded_neur_location(cell_roi, x_midline, barcode_type):
    '''
    Check if the barcoded neuron is on the 'correct' side of the brain based on the cell's location
    '''
    if cell_roi[0] < x_midline: # cell on left hemisphere
        if 'L' in barcode_type:
            return True
        else:
            return False
    elif cell_roi[0] > x_midline: # cell on right hemisphere
        if 'R' in barcode_type:
            return True
        else:
            return False


def make_Pt_R_and_L_side_barcoded_df(volume_barcoding_df):
    # make a right and left specific barcoded neuron dataframes for photostim experiments
    # ideally getting the Mm neurons first and then addding in forward responders
    pt_barcoded_df = volume_barcoding_df[(volume_barcoding_df.Pt == True)]
    R_choose_df = pd.concat([pt_barcoded_df[(pt_barcoded_df.barcoding.str.contains('Mm_R'))],
                             pt_barcoded_df[(pt_barcoded_df.forw_resp == True)
                                            & (pt_barcoded_df.back_resp == False) # really good forward responders
                                             & (~pt_barcoded_df.barcoding.str.contains('Mm'))
                                             & (pt_barcoded_df.barcoding.str.contains('R'))
                                             & (pt_barcoded_df.side == 'R')]])
    L_choose_df = pd.concat([pt_barcoded_df[(pt_barcoded_df.barcoding.str.contains('Mm_L'))],
                             pt_barcoded_df[(pt_barcoded_df.forw_resp == True)
                                            & (pt_barcoded_df.back_resp == False) # really good forward responders
                                             & (~pt_barcoded_df.barcoding.str.contains('Mm'))
                                             & (pt_barcoded_df.barcoding.str.contains('L'))
                                             & (pt_barcoded_df.side == 'L')]])
    R_choose_df.reset_index(drop = True, inplace = True)
    L_choose_df.reset_index(drop = True, inplace = True)

    R_custom_order = ['iMm_R', 'Mm_R', 'S_R', 'B_R', 'iB_R', 'ioB_R', 'oB_R', 'oMl_R']
    R_barcoding_type = pd.CategoricalDtype(categories=R_custom_order, ordered=True)
    R_choose_df['barcoding'] = R_choose_df['barcoding'].astype(R_barcoding_type)

    L_custom_order = ['iMm_L', 'Mm_L', 'S_L', 'B_L', 'iB_L', 'ioB_L', 'oB_L','oMl_L' ]
    L_barcoding_type = pd.CategoricalDtype(categories=L_custom_order, ordered=True)
    L_choose_df['barcoding'] = L_choose_df['barcoding'].astype(L_barcoding_type)

    sorted_R_choose_df = R_choose_df.sort_values(by=['barcoding', 'forw_resp'],
                                                 ascending=[True, False])
    sorted_L_choose_df = L_choose_df.sort_values(by=['barcoding', 'forw_resp'],
                                                 ascending=[True, False,])
    return sorted_R_choose_df, sorted_L_choose_df

def make_Pt_R_and_L_side_df_opposite_tuned(volume_barcoding_df):
    # make a right and left specific barcoded neuron dataframes for photostim experiments
    # grab all the most backward responsive, not forward responsive neurons
    # first Pt neurons, no Mm neurons
    pt_barcoded_df = volume_barcoding_df[(volume_barcoding_df.Pt == True) & (~volume_barcoding_df.barcoding.str.contains('Mm'))]
    R_choose_df = pt_barcoded_df[(pt_barcoded_df.forw_resp == False)
                                & (pt_barcoded_df.back_resp == True) # really good backward responders
                                    & (pt_barcoded_df.barcoding.str.contains('R'))
                                    & (pt_barcoded_df.side == 'R')]
    L_choose_df = pt_barcoded_df[(pt_barcoded_df.forw_resp == False)
                                & (pt_barcoded_df.back_resp == True) # really good backward responders
                                    & (pt_barcoded_df.barcoding.str.contains('L'))
                                    & (pt_barcoded_df.side == 'L')]
    R_choose_df.reset_index(drop = True, inplace = True)
    L_choose_df.reset_index(drop = True, inplace = True)

    R_custom_order = ['S_R', 'B_R', 'iB_R', 'ioB_R', 'oB_R', 'oMl_R'] # definitely don't want any Mm neurons here..
    R_barcoding_type = pd.CategoricalDtype(categories=R_custom_order, ordered=True)
    R_choose_df['barcoding'] = R_choose_df['barcoding'].astype(R_barcoding_type)

    L_custom_order = ['S_L', 'B_L', 'iB_L', 'ioB_L', 'oB_L','oMl_L' ]
    L_barcoding_type = pd.CategoricalDtype(categories=L_custom_order, ordered=True)
    L_choose_df['barcoding'] = L_choose_df['barcoding'].astype(L_barcoding_type)

    sorted_R_choose_df = R_choose_df.sort_values(by=['barcoding', 'back_resp'],
                                                 ascending=[True, False])
    sorted_L_choose_df = L_choose_df.sort_values(by=['barcoding', 'back_resp'],
                                                 ascending=[True, False,])
    return sorted_R_choose_df, sorted_L_choose_df


# finding forward responders in nMLF for stim experiments

def find_forward_responsive_nMLF_cells_by_tuning(fishyvol, frames_motion_on = None, within_deg = 10, save = False):
    '''
    Finding forward responsive nMLF cells based on tuning
    :param fishyvol: fish volume
    :param frames_motion_on: number of frames that motion is on, default (None) is the imaging hz * seconds motion on
    :param within_deg: the degree range from 0 deg that would be considered a forward responder
    :param save: if you want to save the output dataframe
    :return:
    '''
    if frames_motion_on is None:
        frames_motion_on = int(fishyvol[0].img_hz * fishyvol[0].seconds_motion_is_on)

    stimuli_lst = list(fishyvol[0].stimulus_df.stim_name.values.unique())
    stim_angles = [constants.deg_dict[stim] for stim in stimuli_lst]
    nmlf_df_lst = []
    for f, fish in enumerate(fishyvol):
        fish.load_saved_rois()
        nmlf_cells = fish.return_cells_by_saved_roi('nMLF')
        nmlf_rois = fish.return_cell_rois(nmlf_cells)
        x_midline = fish.return_x_midline()
        for i, n in enumerate(nmlf_cells):
            sideness = 'left' if nmlf_rois[i][0] <= x_midline else 'right'
            stim_responses = []
            for stim in stimuli_lst:
                trials = np.array(fish.extended_responses_normf[stim][n])
                all_trial_responses = np.zeros(shape=(len(trials), 1))
                for t, v in enumerate(trials):
                    trial_baseline_mean = v[:-fish.offsets[0]].mean()
                    trial_baseline_std = v[:-fish.offsets[0]].std()
                    trial_response_mean = v[-fish.offsets[0]:-fish.offsets[0] + frames_motion_on].mean()
                    trial_response_tuning_value = (trial_response_mean - trial_baseline_mean) / trial_baseline_std
                    all_trial_responses[t] = trial_response_tuning_value
                response_overall = all_trial_responses.mean(axis=0)[0]
                stim_responses.append(response_overall)
            weights = np.nanmax(stim_responses)
            weighted_angle = angles.weighted_mean_angle(stim_angles, stim_responses)

            bool_resp = True if (weighted_angle >= -within_deg) & (weighted_angle <= within_deg) else False
            nmlf_df_lst.append({
                "plane": f,
                "cell_id": n,
                "region": 'nMLF',
                "location": nmlf_rois[i],
                "side": sideness,
                'forward_resp': bool_resp,
                'weighted_angle': weighted_angle,
                'weights': weights})

        nmlf_df = pd.DataFrame(nmlf_df_lst)
        if save:
            nmlf_df.to_hdf(Path(fish.folder_path.parents[1]).joinpath('nmlf_df.h5'),
                           key='nmlf')  # save this nmlf dataframe for future reference in case
    return nmlf_df


def find_forward_responsive_nMLF_cells(fishyvol, threshold = 1.8, frames_motion_on = None,
                                       num_baseline_frames = 10, save = False):
    '''
    Finding forward responsive nMLF cells based on thresholding using same ideas as above (stricter)
    I found that this will miss some responsive forward cells, so made a new function
    :param fishyvol:
    :param threshold:
    :param frames_motion_on:
    :param num_baseline_frames:
    :param save:
    :return:
    '''

    if frames_motion_on is None:
        frames_motion_on = int(fishyvol[0].img_hz * fishyvol[0].seconds_motion_is_on)
    num_of_stim_reps = list(fishyvol[0].stimulus_df.stim_name.values).count('forward')

    nmlf_df_lst = []
    for f, fish in enumerate(fishyvol):
        fish.load_saved_rois()
        nmlf_cells = fish.return_cells_by_saved_roi('nMLF')
        nmlf_rois = fish.return_cell_rois(nmlf_cells)
        x_midline = fish.return_x_midline()
        for i, n in enumerate(nmlf_cells):
            forward_responses = np.array(fish.extended_responses_normf['forward'][n])
            baseline_means = np.nanmean(forward_responses[:, -fish.offsets[0]-num_baseline_frames:-fish.offsets[0]], axis=1)
            baseline_stds = np.nanstd(forward_responses[:, -fish.offsets[0]-num_baseline_frames:-fish.offsets[0]], axis=1)
            response_means = np.nanmean(
                forward_responses[:, -fish.offsets[0]:-fish.offsets[0] + frames_motion_on],
                axis=1)

            responsive_reps = 0
            for r in range(len(response_means)):
                if response_means[r] >= (threshold * baseline_stds[r]) + baseline_means[r]:
                    responsive_reps += 1
            bool_resp = True if responsive_reps == int(0.8*num_of_stim_reps) else False # we want very forward responsive nmlf neurons

            sideness = 'left' if nmlf_rois[i][0] <= x_midline else 'right'
            nmlf_df_lst.append({
                "plane": f,
                "cell_id": n,
                "region": 'nMLF',
                "location": nmlf_rois[i],
                "side": sideness,
                'forward_resp': bool_resp,
                # "motor_corr": fish.motor_pearson_corrs[n],
                # "motor_corr_pval": fish.motor_pearson_pvals[n],
            })
    nmlf_df = pd.DataFrame(nmlf_df_lst)
    if save:
        nmlf_df.to_hdf(Path(fish.folder_path.parents[1]).joinpath('nmlf_df.h5'),
                   key='nmlf')  # save this nmlf dataframe for future reference in case

    return nmlf_df

### whit's version of barcoding ###
def barcode_score_per_stim(stim_on_frame_list, motion_sensitive_pt_cal_act, n_rep = 3, r_thresh = 0.65):
    '''
    this will find the barcode id per stimulus for each neuron, ends up in a 0 if it does not respond or 1 if it does respond to that stimulus
    stim_on_frame_list -- a list of the frames for a stimulus
    motion_sensitive_pt_cal_act -- the normalized f activity of all the neurons
    n_rep -- the number of repetitions of the experiment
    '''

    test_regressor = [0,0.15,0.3,0.45,0.6,0.75,0.9]

    counter = 0
    rep_num = 0
    base_dur = 4

    on_act_test = np.zeros((len(motion_sensitive_pt_cal_act),n_rep,int(len(stim_on_frame_list)/n_rep)))
    base_act_test = np.zeros((len(motion_sensitive_pt_cal_act),n_rep,int(len(stim_on_frame_list)/n_rep)))

    base_act_avg = np.zeros((len(motion_sensitive_pt_cal_act),n_rep)) # average baseline activity
    base_act_std = np.zeros((len(motion_sensitive_pt_cal_act),n_rep)) # standard deviation of baseline activity
    on_act_max = np.zeros((len(motion_sensitive_pt_cal_act),n_rep)) # maximum activity during motion on
    thresh_score = np.zeros((len(motion_sensitive_pt_cal_act),n_rep)) # threshold score
    
    r_scores = np.zeros((len(motion_sensitive_pt_cal_act),n_rep)) # correlation coefficient 
    pt_m_score = np.zeros((len(motion_sensitive_pt_cal_act)))  # Pt motion score
    
    for i in np.arange(len(motion_sensitive_pt_cal_act)): # for each neuron
        for j in stim_on_frame_list: # for each time the stimulus was on
        
            on_act_test[i][rep_num][counter] = motion_sensitive_pt_cal_act[i][j]
            base_act_test[i][rep_num][counter] = motion_sensitive_pt_cal_act[i][j-base_dur]
            counter = counter + 1
            if counter > 6:
                counter = 0
                rep_num = rep_num + 1
            if rep_num > 2:
                rep_num = 0

        for k in np.arange(n_rep):
            r_scores[i][k] = np.corrcoef(on_act_test[i][k],test_regressor)[0,1]
            base_act_avg[i][k] = np.mean(base_act_test[i][k][-3:-1])
            base_act_std[i][k] = np.std(base_act_test[i][k][-3:-1])
            on_act_max[i][k] = np.max(on_act_test[i][k])
            if (on_act_max[i][k] - base_act_avg[i][k]) > 1.8*base_act_std[i][k]:
                thresh_score[i][k] = 1
        r_scores_avg = np.average(r_scores,axis=1)
        if r_scores_avg[i] > r_thresh and sum(thresh_score[i]) == n_rep:
            pt_m_score[i] = 1 # returns a list of 1s and 0s for each neuron -- 1 if the neuron is sensitive to that motion, 0 if not

    return pt_m_score, r_scores_avg 

def barcoding(pt_scores, num_stims = 6):
    '''
    pt_score -- a dictionary of the scores for each stimulus
    '''

    pt_inward_score = pt_scores['converging']
    pt_owd_score = pt_scores['diverging']
    pt_left_score = pt_scores['left']
    pt_m_left_score = pt_scores['medial_left']
    pt_l_left_score = pt_scores['lateral_left']
    pt_right_score = pt_scores['right']
    pt_m_right_score = pt_scores['medial_right']
    pt_l_right_score = pt_scores['lateral_right']

    len_pt_array = len(pt_left_score)

    # setting up the arrays for each barcode
    pt_bar_stim_count = np.zeros(num_stims + 1) # total amount of stimuli plus another row for one more value

    pt_bi_left_score = np.zeros(len_pt_array)
    pt_med_left_score = np.zeros(len_pt_array)
    pt_lat_left_score = np.zeros(len_pt_array)
    pt_bi_right_score = np.zeros(len_pt_array)
    pt_med_right_score = np.zeros(len_pt_array)
    pt_lat_right_score = np.zeros(len_pt_array)

    # pt_inward_score = np.zeros(len(pt_cal_act))
    # pt_outward_score = np.zeros(len(pt_cal_act))
    # pt_forward_score = np.zeros(len(pt_cal_act))
    # pt_backward_score = np.zeros(len(pt_cal_act))
    # pt_coherent_score = np.zeros(len(pt_cal_act))

    # finding barcoded neurons
    for i in np.arange(len_pt_array):
        if pt_left_score[i] == 1 and pt_m_left_score[i] == 1 and pt_l_left_score[i] == 1 and pt_right_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
            pt_bi_left_score[i] = 1
    
        if pt_left_score[i] == 1 and pt_m_left_score[i] == 1 and pt_l_left_score[i] == 0 and pt_right_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
            pt_med_left_score[i] = 1
        
        if pt_left_score[i] == 1 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 1 and pt_right_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
            pt_lat_left_score[i] = 1

        if pt_right_score[i] == 1 and pt_m_right_score[i] == 1 and pt_l_right_score[i] == 1 and pt_left_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0:
            pt_bi_right_score[i] = 1
        
        if pt_right_score[i] == 1 and pt_m_right_score[i] == 1 and pt_l_right_score[i] == 0 and pt_left_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0:
            pt_med_right_score[i] = 1

        if pt_right_score[i] == 1 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 1 and pt_left_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0:
            pt_lat_right_score[i] = 1
    
    pt_bar_stim_count[0] = sum(pt_bi_left_score)
    pt_bar_stim_count[1] = sum(pt_med_left_score)
    pt_bar_stim_count[2] = sum(pt_lat_left_score)
    pt_bar_stim_count[3] = sum(pt_bi_right_score)
    pt_bar_stim_count[4] = sum(pt_med_right_score)
    pt_bar_stim_count[5] = sum(pt_lat_right_score)
 
    
    # for i in np.arange(len(pt_cal_act)):
    #     if pt_bwd_score[i] == 0 and pt_left_score[i] == 0 and pt_iwd_score[i] == 1 and pt_owd_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0 and pt_right_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
    #         pt_inward_score[i] = 1
    # pt_bar_stim_count[6] = sum(pt_inward_score)
    # for i in np.arange(len(pt_cal_act)):
    #     if pt_fwd_score[i] == 0 and pt_owd_score[i] == 1 and pt_right_score[i] == 0 and pt_inward_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0 and pt_left_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
    #         pt_outward_score[i] = 1
    # pt_bar_stim_count[7] = sum(pt_outward_score)
    # for i in np.arange(len(pt_cal_act)):
    #     if pt_fwd_score[i] == 1 and pt_owd_score[i] == 0 and pt_bwd_score[i] == 0 and pt_right_score[i] == 0 and pt_inward_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0 and pt_left_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
    #         pt_forward_score[i] = 1
    # pt_bar_stim_count[8] = sum(pt_forward_score)
    # for i in np.arange(len(pt_cal_act)):
    #     if pt_fwd_score[i] == 0 and pt_owd_score[i] == 0 and pt_bwd_score[i] == 1 and pt_right_score[i] == 0 and pt_inward_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0 and pt_left_score[i] == 0 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
    #         pt_backward_score[i] = 1
    # pt_bar_stim_count[9] = sum(pt_backward_score)
    # for i in np.arange(len(pt_cal_act)):
    #     if pt_fwd_score[i] == 1 and pt_owd_score[i] == 0 and pt_bwd_score[i] == 1 and pt_right_score[i] == 1 and pt_inward_score[i] == 0 and pt_m_left_score[i] == 0 and pt_l_left_score[i] == 0 and pt_left_score[i] == 1 and pt_m_right_score[i] == 0 and pt_l_right_score[i] == 0:
    #         pt_coherent_score[i] = 1
    # pt_bar_stim_count[10] = sum(pt_coherent_score)
    
    pt_misc_score = np.ones(len_pt_array)
    pt_bar_all_arrays = np.vstack((pt_bi_left_score,pt_med_left_score,pt_lat_left_score,pt_bi_right_score,pt_med_right_score,pt_lat_right_score,
                                #    pt_inward_score,pt_outward_score,pt_forward_score,pt_backward_score,pt_coherent_score)
                                   ))
    pt_bar_all = pt_bar_all_arrays.sum(axis=0)
    pt_bar_all_ind = np.nonzero(pt_bar_all)
    for i in pt_bar_all_ind:
        pt_misc_score[i] = 0
    
    pt_bar_stim_count[num_stims] = sum(pt_misc_score)
    pt_bar_stim_count[0] = sum(pt_bi_left_score)
    pt_bar_stim_count[1] = sum(pt_med_left_score)
    pt_bar_stim_count[2] = sum(pt_lat_left_score)
    pt_bar_stim_count[3] = sum(pt_bi_right_score)
    pt_bar_stim_count[4] = sum(pt_med_right_score)
    pt_bar_stim_count[5] = sum(pt_lat_right_score)
    # pt_bar_stim_count[6] = sum(pt_inward_score)
    # pt_bar_stim_count[7] = sum(pt_outward_score)
    # pt_bar_stim_count[8] = sum(pt_forward_score)
    # pt_bar_stim_count[9] = sum(pt_backward_score)
    # pt_bar_stim_count[10] = sum(pt_coherent_score)
    
    return pt_bi_left_score, pt_med_left_score, pt_lat_left_score, pt_bi_right_score, pt_med_right_score, pt_lat_right_score, pt_misc_score, pt_bar_stim_count
    # return pt_bi_left_score, pt_med_left_score, pt_lat_left_score, pt_bi_right_score, pt_med_right_score, pt_lat_right_score, pt_inward_score, pt_outward_score, pt_forward_score, pt_backward_score, pt_coherent_score, pt_misc_score, pt_bar_stim_count



'''
# Example of how to use barcoding analysis:

all_scores = []
plot_individual = False
n_reps = 3
my_stim_order = clustering.whit_custom_16stim_order
n_stims = len(my_stim_order)

motion_on = 7
baseline = motion_on
length_of_array = motion_on*3

corr_threshold = 0.5 

num_top_neurons = 10

df_lst = []

for plane_no, f in fishvolume.volumes.items():
    
    normcells = arrutils.norm_0to1(f.f_cells)
    f.offsets = (-motion_on, motion_on*2)

    f_neur_resps = clustering.neuron_stim_rep_array(f, n_reps, stim_order = my_stim_order)
    # print(f_neur_resps.shape)

    base_arr, base_std_arr, on_avg_arr, on_max_arr, diff_mean_arr = clustering.various_arrays(f_neur_resps, n_stims, n_reps, 
                                                                                                        len_extendedarr = length_of_array, 
                                                                                                        len_pre = baseline, len_on = motion_on)

    general_resp_neurons = clustering.general_motion_resp_neurons(f_neur_resps, n_stims, n_reps, 
                                                                  len_extendedarr = length_of_array,
                                                                  len_pre = baseline, len_on = motion_on,
                                                                  r_val = corr_threshold)
    
    #only Pt neurons
    general_resp_neur_coords = BaseFish.return_cell_rois(f, general_resp_neurons)
    pt_neurs = [] # neuron ids in terms of the general responsive neurons 
    pt_neurs_coords = [] # neurons coodinates 
    for b, c in enumerate(general_resp_neur_coords):
        if (pt_roi['top_left_x'] < c[0] <= (pt_roi['top_left_x'] + pt_roi['width'])):
            if (pt_roi['top_left_y'] < c[1] <= (pt_roi['top_left_y'] + pt_roi['height'])):
                pt_neurs.append(b)
                pt_neurs_coords.append(c)
    # print(len(pt_neurs))

    pt_resp_neurons = [general_resp_neurons[p] for p in pt_neurs] # getting correct index of Pt neurons from og neuron list
    
    stim_frame_dict = barcoding.get_stim_on_frames(f)
    score_dict = {}
    corr_scores_dict = {}
    for stim_name, stim_frames_lst in stim_frame_dict.items():
        score, corr_scores = barcoding.barcode_score_per_stim(stim_frames_lst, normcells[pt_resp_neurons], r_thresh = corr_threshold)
        score_dict[stim_name] = score
        corr_scores_dict[stim_name] = corr_scores

    pt_bi_left_score, pt_med_left_score, pt_lat_left_score, pt_bi_right_score, pt_med_right_score, pt_lat_right_score, pt_misc_score, pt_bar_stim_count = barcoding.barcoding(score_dict)
    all_scores.append(pt_bar_stim_count)

    barcoding_labels = [] # labeling barcoding clusters for dataframe
    corr_values = [] # correlation values for each neuron in their respective barcoding cluster
    for i in range(len(pt_bi_left_score)):
        if pt_bi_left_score[i] == 1:
            barcoding_labels.append('left')
            corr_values.append(corr_scores_dict['left'][i])
        if pt_med_left_score[i] == 1:
            barcoding_labels.append('medial_left')
            corr_values.append(corr_scores_dict['medial_left'][i])
        if pt_lat_left_score[i] == 1:    
            barcoding_labels.append('lateral_left')
            corr_values.append(corr_scores_dict['lateral_left'][i])
        if pt_bi_right_score[i] == 1:
            barcoding_labels.append('right')
            corr_values.append(corr_scores_dict['right'][i])
        if pt_med_right_score[i] == 1:
            barcoding_labels.append('medial_right')
            corr_values.append(corr_scores_dict['medial_right'][i])
        if pt_lat_right_score[i] == 1:
            barcoding_labels.append('lateral_right')
            corr_values.append(corr_scores_dict['lateral_right'][i])
        if pt_misc_score[i] == 1:
            barcoding_labels.append('misc')
            corr_values.append(np.nan)

    # making the dataframe for one plane
    barcoding_df = pd.DataFrame(columns = ['plane','neur_ids', 'neur_coords', 'barcoding', 'barcode_corr', 'photostimulated'])
    barcoding_df['plane'] = [f.data_paths['suite2p'].parents[1].name] * len(pt_resp_neurons) 
    barcoding_df['neur_ids'] = pt_resp_neurons
    barcoding_df['neur_coords'] = pt_neurs_coords
    barcoding_df['barcoding'] = barcoding_labels
    barcoding_df['barcode_corr'] = corr_values
    df_lst.append(barcoding_df)

    if plot_individual:
        labels = ['left', 'med_left', 'lat_left', 'right', 'med_right', 'lat_right', 'misc']
        num_bars = pt_bar_stim_count.shape[0]
        for x, y in zip(range(num_bars), pt_bar_stim_count):
            plt.bar(x, height = y)
        plt.xticks(ticks = range(num_bars), labels = labels)
        plt.title(f'Plane {plane_no}')
        plt.ylabel('Count')
        plt.show()

volume_barcoding_df = pd.concat(df_lst)
volume_barcoding_df.reset_index(inplace = True, drop = True)

num_bars = pt_bar_stim_count.shape[0]
all_scores_arr = np.array(all_scores)
sum_all_scores = np.sum(all_scores_arr, axis=0)
labels = ['left', 'med_left', 'lat_left', 'right', 'med_right', 'lat_right', 'misc']
for x, y in zip(range(num_bars), sum_all_scores):
    plt.bar(x, height = y)
plt.xticks(ticks = range(num_bars), labels = labels)
plt.ylabel('Count')
plt.title('Full volume counts')
plt.show()
print(sum_all_scores)

'''