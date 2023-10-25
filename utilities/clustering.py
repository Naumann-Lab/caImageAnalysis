import scipy.cluster.hierarchy as sch
from bcdict import BCDict

from utilities import arrutils

import numpy as np

custom_16stim_order = [
    "converging","diverging",
    "left","medial_left","lateral_left",
    "right","medial_right","lateral_right",
    "forward","backward","forward_backward", "forward_x", "x_backward", "backward_forward", "x_forward", "backward_x",
    ]

def groupby(a, b):
    # Get argsort indices, to be used to sort a and b in the next steps
    sidx = b.argsort(kind='mergesort')
    a_sorted = a[sidx]
    b_sorted = b[sidx]
    # Get the group limit indices (start, stop of groups)
    cut_idx = np.flatnonzero(np.r_[True,b_sorted[1:] != b_sorted[:-1],True])
    # Split input array with those start, stop ones
    out = [a_sorted[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])]
    return out


def neuron_stim_rep_array(somefishclass, stim_order = custom_16stim_order):
    '''
    somefishclass -- has to be a VizStimFish
    output -- array of shape: # of neurons, each repetition, and each stim (in the order of the stim_df) 
            array of activity (length of offsets * num of stims) 
    '''
    normcells = arrutils.norm_fdff(somefishclass.f_cells)

    somefishclass.stimulus_df['rep'] = 0

    # choosing a generic stim here for counting reps
    n_stims = somefishclass.stimulus_df.stim_name.nunique()
    for i in range(len(somefishclass.stimulus_df[somefishclass.stimulus_df.stim_name == 'right'])):
        somefishclass.stimulus_df.iloc[(n_stims*i):(n_stims*i+n_stims)]['rep'] = i

    # do not want incomplete reps
    last_rep = int(somefishclass.stimulus_df.rep.iloc[-1])
    if len(somefishclass.stimulus_df[somefishclass.stimulus_df['rep'] == last_rep]) < n_stims:
        drop_rows = somefishclass.stimulus_df[somefishclass.stimulus_df['rep'] == last_rep].index
        somefishclass.stimulus_df.drop(drop_rows, axis=0, inplace = True)


    neur_resps = np.zeros(shape=(
        len(normcells),
        somefishclass.stimulus_df.rep.nunique(),
        np.diff(somefishclass.offsets)[0] * somefishclass.stimulus_df.stim_name.nunique()
        ))

    for r in somefishclass.stimulus_df.rep.unique():
        one_rep = somefishclass.stimulus_df[somefishclass.stimulus_df.rep == r]
        all_arrs = np.zeros(shape=(one_rep.stim_name.nunique(),np.diff(somefishclass.offsets)[0]))

        for st, stim in enumerate(stim_order):
            # finding the frames for each stimuli between the offsets 
            arrs = arrutils.subsection_arrays(one_rep[one_rep.stim_name == stim].frame.values,somefishclass.offsets)
            all_arrs[st] = arrs[0]

        # transforming data types of the arr for indexing into neur list
        _all_arrs = [item for sublist in all_arrs for item in sublist]
        _all_arrs = [int(i) for i in _all_arrs]

        # for each neuron in those specific frame arrays    
        for n, nrn in enumerate(normcells): 
            resp_arr = nrn[_all_arrs]
            neur_resps[n][r] = resp_arr
    
    return neur_resps



def various_arrays(o_t, n_stim, n_reps, len_extendedarr = 21, len_pre = 7, len_on = 7):
    '''    
    o_t = original traces in shape of [# of neurons, # of repetitions, # of stimuli * length of offsets before/after stimulus]
    n_stim = number of stimuli in experiment
    n_reps = number of repetitions or trials of experiments

    len_extendedarr = # total number of frames that is taken from the neural trace (i.e. somefishclass.offsets difference)

    len_pre = length of array before stimulus on
    len_on = length of array when stimulus is on
    '''

    base_start = 4 
    o_t_base = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_base_std = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_on_max = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_on_min = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_on_avg = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_diff = np.zeros(shape=(len(o_t),n_stim,n_reps))
    o_t_diff_mean = np.zeros((len(o_t),n_stim))

    for i in np.arange(len(o_t)):
        for j in np.arange(n_stim):
            for k in np.arange(n_reps):
                o_t_base[i,j,k] = np.mean(o_t[i][k][len_extendedarr*j+base_start: len_extendedarr*j+len_pre]) # average baseline values
                o_t_base_std[i,j,k] = np.std(o_t[i][k][len_extendedarr*j+base_start: len_extendedarr*j+len_pre]) # std of baseline values

                o_t_on_max[i,j,k] = np.max(o_t[i][k][len_extendedarr*j+len_pre: len_extendedarr*j+len_pre+len_on]) # peak during motion on

                o_t_on_min[i,j,k] = np.min(o_t[i][k][len_extendedarr*j+len_pre: len_extendedarr*j+len_pre+len_on]) # minimum during motion on

                o_t_on_avg[i,j,k] = np.mean(o_t[i][k][len_extendedarr*j+len_pre: len_extendedarr*j+len_pre+len_on]) # average during motion on

                # stimulus diff index (made by Whit), assign a score to describe if cell was 'responsive' to motion
                if o_t_on_avg[i,j,k] > o_t_base[i,j,k]:
                    o_t_diff[i,j,k] = o_t_on_max[i][j][k] - o_t_base[i][j][k]
                if o_t_on_avg[i,j,k] <= o_t_base[i,j,k]:
                    o_t_diff[i,j,k] = o_t_on_min[i][j][k] - o_t_base[i][j][k]
                
            o_t_diff_mean[i,j] = np.mean(o_t_diff[i][j]) 

    return o_t_base, o_t_base_std, o_t_on_avg, o_t_on_max, o_t_diff_mean




def general_motion_resp_neurons(o_t, n_stim, n_reps, len_extendedarr = 21, len_pre = 7, len_on = 7, r_val = 0.65):
    '''
    o_t = original traces in shape of [# of neurons, # of repetitions, # of stimuli * length of offsets before/after stimulus]

    '''
    
    o_t_base, o_t_base_std, o_t_on_avg, o_t_on_max, o_t_diff_mean = various_arrays(o_t, n_stim, n_reps, len_extendedarr, len_pre, len_on)

    resp_dict = BCDict()
    coor_dict = BCDict()
    motion_responsive_neurons = []

    for i in np.arange(len(o_t)):
        if i not in resp_dict.keys():
                resp_dict[i] = BCDict()
                coor_dict[i] = BCDict()
        for j in np.arange(n_stim):
            corr_lst = []
            resp_lst = []
            
            for k in np.arange(n_reps):
                # 1 - stim on period is corr with neuron's activity
                cell_arr = o_t[i][k][len_extendedarr*j+len_pre: len_extendedarr*j+len_pre+len_on]
                stim_arr = np.linspace(0, 1, len_on)
                corr_val = round(np.corrcoef(stim_arr, cell_arr)[0][1], 3)
                corr_lst.append(corr_val)

                # 2 - peak vs base response
                if o_t_on_max[i][j][k] >= (o_t_base[i][j][k] + (1.8* o_t_base_std[i][j][k])):
                    resp_lst.append(True)
                else:
                    resp_lst.append(False)

            if np.nanmean(corr_lst) >= r_val:
                coor_dict[i][j] = True
            else:
                coor_dict[i][j] = False
            
            if sum(resp_lst) == len(resp_lst):
                resp_dict[i][j] = True
            else:
                resp_dict[i][j] = False

        if (coor_dict[i][j] == True) & (resp_dict[i][j] == True):
            motion_responsive_neurons.append(i)
    
    return motion_responsive_neurons
