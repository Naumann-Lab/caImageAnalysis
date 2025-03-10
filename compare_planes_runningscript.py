"""
The running script for compare_planes. So far, the data depends on the csv out put of load_fish_data.py, but this
can be easily modified to directly fetch file from fish (will probably take forever).

@Zichen He 240313
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tqdm, json
from ast import literal_eval
import constants, plot_individual_plane
from fishy import BaseFish
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner
import compare_planes


def cleanup_csv(df, columns):
    """
    deal with the issues that all the lists/tuples/complex objects stored in the csv can't be properly read
    """
    for column in columns:
        if type(df[column].iloc[0]) == str:
            df[column] = [literal_eval(df[column].iloc[i]) if type(
                df[column].iloc[i]) == str else np.nan for i in range(len(df))]

    return df


def cycle_onefish_tail(fish, planerange, strength_boundary, stimulus_s):
    # gather all information space
    save_path_all = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish + '/'
    stimulus_dfs = pd.DataFrame()
    tail_dfs = pd.DataFrame()
    tail_bout_dfs = pd.DataFrame()
    frametimes_dfs = pd.DataFrame()
    maxframe = 0

    # Collect all information
    for plane in tqdm.tqdm(planerange, 'tail plane data collection'):
        save_path = save_path_all + 'plane_' + str(plane) + '/data/'

        # LOAD ALL DATA
        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]
        frametimes_dfs = pd.concat([frametimes_dfs, frametimes_df])
        frametimes_dfs = frametimes_dfs.reset_index(drop=True)

        tail_bout_df = pd.read_csv(save_path + 'tail_bout_df_proofread.csv', index_col=0)
        tail_bout_df = cleanup_csv(tail_bout_df, tail_bout_df.columns.drop('tail_stimuli'))
        tail_bout_df = tail_bout_df.set_index(np.add([plane * 0.01] * len(tail_bout_df), tail_bout_df.index))
        tail_bout_df['cont_tuples_imageframe'] = [(tail_bout_df.loc[row]['cont_tuples_imageframe'][0] + maxframe,
                                                  tail_bout_df.loc[row]['cont_tuples_imageframe'][1] + maxframe)
                                                  for row in tail_bout_df.index]
        tail_bout_df['plane'] = [plane] * len(tail_bout_df)
        tail_bout_dfs = pd.concat([tail_bout_dfs, tail_bout_df])

        stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)
        stimulus_df.frame = np.add(stimulus_df.frame, maxframe)
        stimulus_dfs = pd.concat([stimulus_dfs, stimulus_df])
        stimulus_dfs = stimulus_dfs.reset_index(drop=True)

        maxframe = maxframe + frametimes_df.shape[0]

    #ACTUAL ANALYSIS
    # re analyze the tail so i don't have to re write a code, but this graph is only for plotting purpose while
    # the data is not used for actual presentation/analysis
    tail_bout_dfs.to_csv(save_path_all + 'tail_bout_df_all.csv')
    stimuli_presenting_responding_df = plot_individual_plane.tail_angle_all(frametimes_dfs, stimulus_dfs,
                                                                 tail_bout_dfs, stimulus_s=stimulus_s)
    plt.savefig(save_path_all + '01_tail_bouts_acrossplane.png')
    stimuli_presenting_responding_df.to_csv(save_path_all + 'stimuli_presenting_responding_df.csv')
    plot_individual_plane.tail_angle_binocular(tail_bout_dfs)
    plt.savefig(save_path_all + '02_tail_binocular_acrossplane.png')
    # tail_clustering_result = compare_planes.planes_plot_tail_type_clusters(region_ysort_normf, tail_bout_dfs, tail_dfs, save_path_all, k_clusters=7)
    # tail_clustering_result.to_csv(save_path_all + '90_tail_clustering_result.csv')



def cycle_onefish_neuron(fish, planerange, refplane, top_percentage, artr):
    #gather all information space
    save_path_all = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish + '/'
    potential_region_list = ['PT', 'nMLF', 'HBr']#['FBr', 'HBr']
    region_ROIs = {region: {'xpos': [], 'ypos': [], 'zpos': []} for region in potential_region_list}
    stimulus_dfs = pd.DataFrame()
    tail_dfs = pd.DataFrame()
    tail_bout_dfs = pd.DataFrame()
    frametimes_dfs = pd.DataFrame()
    neuron_tail_dfs = pd.DataFrame()
    pearson_neuron_tail_dfs = pd.DataFrame()
    maxframe = 0
    wholebrain_ysort_rois = pd.DataFrame()
    wholebrain_ysort_normfs = pd.DataFrame()
    region_ysort_rois = {key: None for key in potential_region_list}
    region_ysort_normfs = {key: None for key in potential_region_list}
    vspn = np.load('/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish + '/vspn.npy')

    #Collect all information
    for plane in tqdm.tqdm(planerange, 'neuron plane data collection'):
        save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish + '/plane_' + str(plane) + '/data/'

        # LOAD ALL DATA
        if plane == refplane:
            refImg = pd.read_csv(save_path + 'refImg.csv', index_col=0)

        regions = pd.read_csv(save_path + 'regions.csv', index_col=0).values.flatten().tolist()
        regions = [r for r in regions if r != 'wholebrain']

        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]
        frametimes_dfs = pd.concat([frametimes_dfs, frametimes_df])
        frametimes_dfs = frametimes_dfs.reset_index(drop=True)

        # stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)
        # stimulus_df.frame = np.add(stimulus_df.frame, maxframe)
        # stimulus_dfs = pd.concat([stimulus_dfs, stimulus_df])
        # stimulus_dfs = stimulus_dfs.reset_index(drop=True)

        # tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        # tail_df['t_dt'] = pd.to_datetime(tail_df['t_dt'], format='%H:%M:%S.%f')
        # tail_df['t_dt'] = [x.time() for x in tail_df['t_dt']]
        # tail_df['plane'] = [plane] * len(tail_df)
        # tail_df['frame_plane'] = tail_df.frame
        # tail_df['tailindex_plane'] = tail_df.index
        # tail_df.frame = np.add(tail_df.frame, maxframe)
        # tail_dfs = pd.concat([tail_dfs, tail_df])
        # tail_dfs = tail_dfs.reset_index(drop=True)
        #
        tail_bout_df = pd.read_csv(save_path + 'tail_bout_df_proofread.csv', index_col=0)
        tail_bout_df = cleanup_csv(tail_bout_df, tail_bout_df.columns.drop('tail_stimuli'))
        tail_bout_df = tail_bout_df.set_index(np.add([plane * 0.01] * len(tail_bout_df), tail_bout_df.index))
        tail_bout_df['plane'] = [plane] * len(tail_bout_df)
        tail_bout_dfs = pd.concat([tail_bout_dfs, tail_bout_df])

        with open(save_path + 'region_cell_index.json') as file:
            region_cell_index = json.load(file)

        ysort_roi = pd.read_csv(save_path + "ysort_rois.csv", index_col=0)
        #ysort_roi = ysort_roi.set_index(np.add([plane * 0.01] * ysort_roi.shape[0], ysort_roi.index))
        wholebrain_ysort_roi = ysort_roi.loc[region_cell_index['wholebrain']]
        wholebrain_ysort_roi['zpos'] = [plane] * wholebrain_ysort_roi.shape[0]
        wholebrain_ysort_roi = wholebrain_ysort_roi.set_index(np.add([plane * 0.01] * wholebrain_ysort_roi.shape[0], wholebrain_ysort_roi.index))
        wholebrain_ysort_rois = pd.concat([wholebrain_ysort_rois, wholebrain_ysort_roi])

        ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
        #ysort_normf = ysort_normf.set_index(
        #    np.add([plane * 0.01] *ysort_normf.shape[0], ysort_normf.index))
        wholebrain_ysort_normf = ysort_normf.loc[region_cell_index['wholebrain']]
        wholebrain_ysort_normf = wholebrain_ysort_normf.set_index(np.add([plane * 0.01] * wholebrain_ysort_normf.shape[0], wholebrain_ysort_normf.index))
        wholebrain_ysort_normfs = pd.concat([wholebrain_ysort_normfs, wholebrain_ysort_normf])

        # if len(regions) > 0:
        #     for r in regions:  # find region cells that are actually within the whole brain roi, also re-save it so i dont have to do it again:)
                #region_cell_index = list(pd.read_csv(save_path + r + '_ysort_normf.csv', index_col=0).index)
                #region_cell_index  = [plane * 0.01 + i for i in region_cell_index]
                #region_cell_index[r] = [n for n in region_cell_index[r] if n in region_cell_index['wholebrain']]#okay i forgot to save it, in the future redump my json file here
                # if r == 'OT' or r == 'PT':
                #     r = 'FBr'
                # else:
                #     r = 'HBr'
                #region_ysort_rois[r] = pd.concat([region_ysort_rois[r], ysort_roi.loc[region_cell_index]], ignore_index = True, axis = 0)
                #region_ysort_normfs[r] = pd.concat([region_ysort_normfs[r], ysort_normf.loc[region_cell_index]], ignore_index = True, axis = 0)

        if len(regions) > 0:
            for r in regions:
                region_cell_index[r] = [n + plane*0.01 for n in region_cell_index[r] if n in region_cell_index['wholebrain']]
                region_ysort_roi = wholebrain_ysort_roi.loc[region_cell_index[r]]
                region_ysort_normf = wholebrain_ysort_normf.loc[region_cell_index[r]]
                try:
                    region_ysort_normfs[r] = pd.concat([region_ysort_normfs[r], region_ysort_normf])
                    region_ysort_rois[r] = pd.concat([region_ysort_rois[r], region_ysort_roi])
                except TypeError: #in the first plane we end up here
                    region_ysort_normfs[r] = region_ysort_normf
                    region_ysort_rois[r] = region_ysort_roi
                region_ROI = pd.read_csv(save_path[:-5] + 'ROIs/' + r + '.csv')
                region_ROIs[r]['xpos'] = region_ROIs[r]['xpos'] + list(region_ROI.loc[:, 'X'])
                region_ROIs[r]['ypos'] = region_ROIs[r]['ypos'] + list(region_ROI.loc[:, 'Y'])
                region_ROIs[r]['zpos'] = region_ROIs[r]['zpos'] + [plane] * region_ROI.shape[0]
        #
        neuron_tail_df = pd.read_csv(save_path + 'neuron_tail_df_stop.csv', index_col=0)
        neuron_tail_df = cleanup_csv(neuron_tail_df, neuron_tail_df.columns)
        neuron_tail_df = neuron_tail_df.set_index(np.add([plane * 0.01] * len(neuron_tail_df), neuron_tail_df.index))
        neuron_tail_dfs = pd.concat([neuron_tail_dfs, neuron_tail_df])
        # pearson_neuron_tail_df = pd.read_csv(save_path + 'neuron_tail_df_pearson.csv', index_col=0)
        # pearson_neuron_tail_df = pearson_neuron_tail_df.set_index(np.add([plane * 0.01] * len(pearson_neuron_tail_df), pearson_neuron_tail_df.index))
        # pearson_neuron_tail_dfs = pd.concat([pearson_neuron_tail_dfs, pearson_neuron_tail_df])

        maxframe = maxframe + frametimes_df.shape[0]

    # ACTUAL ANALYSIS
    # fig = compare_planes.volumetric_plot_pearson_tailneurons(pearson_neuron_tail_dfs, wholebrain_ysort_rois, top_percentage)
    # fig.write_html(save_path_all + "10_pearson_tail_neuron.html")
    #
    # fig, fig_response, fig_predict, fig_scatter = compare_planes.volumetric_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, refImg, top_percentage)
    # fig_scatter.savefig(save_path_all + "11_tail_neuron.png")
    # fig.write_html(save_path_all + "12_tail_neuron.html")
    # fig_response.write_html(save_path_all + "13_tail_respond_neuron.html")
    # fig_predict.write_html(save_path_all + "14_tail_predict_neuron.html")
    #
    # _ = compare_planes.plot_alltrace_tail_neuron_respondingonly(frametimes_dfs, tail_dfs, tail_bout_dfs, wholebrain_ysort_normfs,
    #                          neuron_tail_dfs, top_percentage = top_percentage, normalized_window = (-5, 20), normalized = False)
    # plt.savefig(save_path_all + '20_tail_neuron_trace(respondingonly_notnormalized).png')
    # _ = compare_planes.plot_alltrace_tail_neuron_respondingonly(frametimes_dfs, tail_dfs, tail_bout_dfs,wholebrain_ysort_normfs,
    #                             neuron_tail_dfs, top_percentage= top_percentage, normalized_window=(-5, 20), normalized=True)
    # plt.savefig(save_path_all + '21_tail_neuron_trace(respondingonly_normalized).png')
    # _ = compare_planes.plot_alltrace_tail_neuron_all(frametimes_dfs, tail_dfs, tail_bout_dfs, wholebrain_ysort_normfs,
    #                          neuron_tail_dfs, top_percentage = top_percentage, normalized_window = (-5, 20), normalized = False)
    # plt.savefig(save_path_all + '22_tail_neuron_trace(all_notnormalized).png')
    # _ = compare_planes.plot_alltrace_tail_neuron_all(frametimes_dfs, tail_dfs, tail_bout_dfs, wholebrain_ysort_normfs,
    #                         neuron_tail_dfs, top_percentage= top_percentage, normalized_window=(-5, 20), normalized=True)
    # plt.savefig(save_path_all + '23_tail_neuron_trace(all_normalized).png')
    #
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, top_percentage, 'omr_index')
    # plt.savefig(save_path_all + '30_tail_neuron_region_omrindex.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.05, 'omr_index')
    # plt.savefig(save_path_all + '30_tail_neuron_region_omrindex_5.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.99, 'omr_index')
    # plt.savefig(save_path_all + '30_tail_neuron_region_omrindex_all.png')
    #compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, top_percentage, 'duration')
    #plt.savefig(save_path_all + '31_stoptail_neuron_region_duration.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.05, 'duration')
    # plt.savefig(save_path_all + '31_tail_neuron_region_duration_5.png')
    #compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.99, 'duration')
    #plt.savefig(save_path_all + '31_stoptail_neuron_region_duration_all.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, top_percentage, 'direction')
    # plt.savefig(save_path_all + '33_tail_neuron_region_direction.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.05, 'direction')
    # plt.savefig(save_path_all + '33_tail_neuron_region_direction_5.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.99, 'direction')
    # plt.savefig(save_path_all + '33_tail_neuron_region_direction_all.png')
    #compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, top_percentage, 'stim_direction')
    #plt.savefig(save_path_all + '33_tail_neuron_region_stimdirection.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.05,'stim_direction')
    # plt.savefig(save_path_all + '33_tail_neuron_region_stimdirection_5.png')
    #compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.99,'stim_direction')
    #plt.savefig(save_path_all + '33_tail_neuron_region_stimdirection_all.png')
    # compare_planes.region_plot_tailneurons(neuron_tail_dfs, wholebrain_ysort_rois, region_ysort_rois, refImg, 0.2, 'cluster')
    # plt.savefig(save_path_all + '34_tail_neuron_region_cluster.png')
    midline = artr[3]
    lateralization_corr_df = compare_planes.artr_lateralization(vspn, midline, neuron_tail_dfs, wholebrain_ysort_rois, fish, planerange, top_percentage)
    lateralization_corr_df.to_csv(save_path_all + 'lateralization_corr_df.csv')
    # lateralization_corr_df = compare_planes.artr_lateralization_pop(artr, neuron_tail_dfs, wholebrain_ysort_rois, fish,
    #                                                             planerange, top_percentage)
    # lateralization_corr_df.to_csv(save_path_all + 'pop_lateralization_corr_df.csv')

    #cell dynamics
    #_, mean_on_duration = plot_individual_plane.plot_on(frametimes_df, region_ysort_normfs, region_ysort_rois, 'cluster', 0, 30,
    #                              refImg, False, tail_bout_dfs)
    #plt.savefig(save_path_all + 'cluster_on_tailonly.png')#baseline on
    #mean_on_duration['FBr'].to_csv(save_path_all + 'FBr_cluster_on.csv')
    #mean_on_duration['HBr'].to_csv(save_path_all + 'HBr_cluster_on.csv')
    #mean_on_duration['nMLF'].to_csv(save_path_all + 'nMLF_cluster_on.csv')
    #mean_on_duration['PT'].to_csv(save_path_all + 'PT_cluster_on.csv')
    #mean_on_duration['HBr'].to_csv(save_path_all + 'HBr_cluster_on.csv')

#[0: tail_planes, 1: neuron_planes, 2: strength_boundary, 3: stimulus_s, 4: refplane]
fish_dict = {
            #'danionella_fish0': [list(range(0, 19)), list(range(0, 19)), 0, 5, 12, [np.nan, np.nan, np.nan, np.nan]],
            #'danionella_fish3': [list(range(0, 22)), list(range(0, 9)) + list(range(10, 19)), 0, 5, 12, [np.nan, np.nan, np.nan, np.nan]],
            #'danionella_fish4': [list(range(0, 9)), list(range(0, 9)), 0, 5, 8, [np.nan, np.nan, np.nan, np.nan]],
            #'danionella_fish5': [list(range(0, 12)), list(range(0, 12)), 0, 5, 11, [np.nan, np.nan, np.nan, np.nan]],
            'danionella_fish10': [list(range(9, 20)), list(range(9, 19)), 0.25, 5, 10, [550, 620, 230, 265, 300]],
            'danionella_fish11': [list(range(0, 20)), list(range(1, 19)), 0.33, 5, 13, [585, 635, 250, 275, 300]],
             #'danionella_fish14': [[7, 8, 9, 10, 11, 18, 19], [7, 8, 9, 10, 11, 18, 19], 0.22, 15, 10],#REMEMBER SPECIAL TAIL PROCESSING
             #'danionella_fish17': [list(range(8, 10)), list(range(8, 10)), 0.1, 15, 9]
            #'zebrafish_fish0': [list(range(0, 13)), list(range(0, 13)), 0, 5, 12, [np.nan, np.nan, np.nan, np.nan]],
             # 'zebrafish_fish1': [list(range(0, 14)), list(range(3, 14)), 0.4, 5, 12, [620, 680, 275,315, 355]],
              'zebrafish_fish4': [list(range(0, 19)), list(range(2, 18)), 0.2, 5, 11, [600, 670, 255, 290, 325]],
              'zebrafish_fish5': [list(range(0, 16)), list(range(2, 16)), 0.25, 5, 5, [670, 750, 255, 305, 355]],
              'zebrafish_fish6': [list(range(0, 19)), list(range(1, 19)), 0.25, 5, 11, [580, 650, 280, 320, 360]]
              }


for fish, params in tqdm.tqdm(fish_dict.items(), 'fish progress'):
    #cycle_onefish_tail(fish = fish, planerange = params[0], strength_boundary= params[2], stimulus_s = params[3])
    cycle_onefish_neuron(fish = fish, planerange = params[1], top_percentage = 0.2, refplane = params[4],
                         artr = params[5])
