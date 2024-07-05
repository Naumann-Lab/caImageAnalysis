import numpy as np
import constants
from utilities import arrutils, clustering

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
                             frames_motion_on = 7, length_of_total_frame_arr = None, std_thresh = 1.6):
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
        stim_resp_each_cell_arr = clustering.neuron_each_stim_rep_arrays(vizstimfish, n_reps, stim_order)
    
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

    # correlate ideal fake traces for each barcode:
    corr_dict = {}
    bool_dict = {}
    for n, neuron_arr in enumerate(stim_resp_each_cell_arr):

        # determining the binary code for this neuron
        neuron_binary_code = barcode_binary_score(vizstimfish, neuron_arr, base_length = 4, frames_motion_on = frames_motion_on, 
                                    std_thresh = std_thresh, num_responding_trials = int(n_reps*0.75))
        
        for typ, binary_code in barcode_dict.items():
            if typ not in bool_dict.keys():
                bool_dict[typ] = {}
                corr_dict[typ] = {}
            mean_neuron_arr = np.nanmean(neuron_arr, axis = 0)[:len(ideal_barcorde_dict[typ])]
            corr = np.corrcoef(ideal_barcorde_dict[typ], mean_neuron_arr)[0, 1]
            corr_dict[typ][n] = corr

            if neuron_binary_code == binary_code:
                bool_dict[typ][n] = True
            else:
                bool_dict[typ][n] = False
        
    neurons_per_barcode_dict = {}
    for typ, bools in bool_dict.items():
        if typ not in neurons_per_barcode_dict.keys():
            neurons_per_barcode_dict[typ] = [key for key, value in bools.items()] # selected neurons

    return neurons_per_barcode_dict, bool_dict, corr_dict


def barcode_binary_score(vizstimfish, one_neuron_arr, base_length = 4, frames_motion_on = None, std_thresh = 1.8, num_responding_trials = 3):
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
        frames_motion_on = int(vizstimfish.img_hz*5)

    bool_dict_per_neuron = {key: 0 for key in vizstimfish.stim_order}
    for e, l in enumerate(vizstimfish.stim_start_frames):
        key = vizstimfish.stim_order[e]
        base_arr = np.nanmedian(one_neuron_arr[:, (l-base_length):l], axis = 1)
        base_std = np.nanstd(one_neuron_arr[:, (l-base_length):l], axis = 1)
        evoked_arr = np.nanmedian(one_neuron_arr[:, l:(l+ frames_motion_on)], axis =1 )
        count = 0
        for d in range(len(evoked_arr)):
            if evoked_arr[d] > (base_arr[d] + std_thresh*base_std[d]):
                count += 1
        if count >= num_responding_trials:
            bool_dict_per_neuron[key] = 1
    binary_code = [bool(v) for v in bool_dict_per_neuron.values()]

    return binary_code


# whit's version of barcoding
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