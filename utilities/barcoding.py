import numpy as np
from utilities.clustering import barcoding_8stim_order

def get_stim_on_frames(somefishy, stim_set = barcoding_8stim_order, motion_on_frames = 7):
    '''
    somefishy -- has to be a VizStimFish with a stimulus_df
    stim_set -- a list of the stimuli you want to get the frames for
    motion_on_frames -- the number of frames the motion is on for (7 frames typicallY)
    getting all the frames for motion on into a dictionary
    '''
    stim_frame_dict = {}
    for q in stim_set:
        start_frames = somefishy.stimulus_df[somefishy.stimulus_df['stim_name'] == q].frame.values
        stim_on_frame_list = []
        for k in start_frames:
            stim_on_frame_list.extend(list(range(k, k + motion_on_frames)))
        stim_frame_dict[q] = stim_on_frame_list
    
    return stim_frame_dict

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