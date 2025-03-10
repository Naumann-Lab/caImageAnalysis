"""
Run through all planes and plot all graphs:)

Note that data needs to go through load_fish_data.py so they are in (faster) csv forms

@Zichen He 20240313
"""

import pandas as pd
from matplotlib import pyplot as plt
from fishy import BaseFish
import plot_individual_plane
import numpy as np
import json
import tqdm
from ast import literal_eval

hzReturner = BaseFish.hzReturner

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
    for plane in tqdm.tqdm(planerange, fish + ' tail plane progress'):
        """
        Make sure all fish entering here has tail df and a stimulus df (workingfish_tail)
        """
        fish_plane = fish + '/plane_' + str(plane)
        save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish_plane + '/data/'
        img_save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish_plane + '/'

        # LOAD ALL DATA
        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]

        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        tail_df['t_dt'] = pd.to_datetime(tail_df['t_dt'], format='%H:%M:%S.%f')
        tail_df['t_dt'] = [x.time() for x in tail_df['t_dt']]
        tail_hz = hzReturner(tail_df)

        stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)

        # ACTUALLY START ANALYSIS
        tail_bout_df = plot_individual_plane.analyze_tail(
            frametimes_df, stimulus_df, tail_df, tail_hz, stimulus_s=stimulus_s, strength_boundary=strength_boundary)
        plt.savefig(img_save_path + '00_tail.png')
        tail_bout_df.to_csv(save_path + '/tail_bout_df.csv')

        plot_individual_plane.tail_angle_all(frametimes_df, stimulus_df, tail_bout_df, stimulus_s=stimulus_s)
        plt.savefig(img_save_path + '01_tail_bouts.png')
        plot_individual_plane.tail_angle_binocular(tail_bout_df, stimuli_s=stimulus_s)
        plt.savefig(img_save_path + '02_tail_binocular.png')

        plt.close()

def cycle_onefish_neuron(fish, planerange, top_percentage, stimulus_s):
    """
    Make sure all fish entering here are workingfish_tail, also have their region ROIs besides wholebrain drawn.
    """
    for plane in tqdm.tqdm(planerange, fish + ' neural plane progress'):
        fish_plane = fish + '/plane_' + str(plane)
        save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' +  fish_plane + '/data/'
        img_save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/'  + fish_plane + '/'

        #LOAD ALL DATA
        regions = pd.read_csv(save_path + 'regions.csv', index_col=0).values.flatten().tolist()
        regions = [r for r in regions if r != 'wholebrain']
        try:
           refImg = pd.read_csv(save_path + 'refImg.csv', index_col=0)
        except:
           refImg = pd.DataFrame()
        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]

        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        tail_df['t_dt'] = pd.to_datetime(tail_df['t_dt'], format='%H:%M:%S.%f')
        tail_df['t_dt'] = [x.time() for x in tail_df['t_dt']]
        tail_bout_df = pd.read_csv(save_path + 'tail_bout_df_proofread_clustered.csv', index_col = 0)
        tail_bout_df = cleanup_csv(tail_bout_df, tail_bout_df.columns.drop('tail_stimuli'))
        tailhz = hzReturner(tail_df)
        for bout in tail_bout_df.index:
            bout_on = tail_bout_df.loc[bout, 'cont_tuples_tailindex'][0]
            accounting_bout = len([i[0] for i in tail_bout_df['cont_tuples_tailindex'] if
                                   (i[0] <= bout_on + 1 * tailhz) and (i[0] >= bout_on - 1 * tailhz)])
            tail_bout_df.loc[bout, 'bouts_nearby'] = accounting_bout
        tail_bout_df['tail_magnitude'] = tail_bout_df['tail_angle_posmax'] - tail_bout_df['tail_angle_negmin']

        #stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)

        ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
        ysort_rois = pd.read_csv(save_path + 'ysort_rois.csv', index_col=0)
        with open(save_path + 'region_cell_index.json') as file:
           region_cell_index = json.load(file)
        wholebrain_ysort_normf = ysort_normf.loc[region_cell_index['wholebrain']]
        wholebrain_ysort_rois = ysort_rois.loc[region_cell_index['wholebrain']]

        if len(regions) > 0:
            region_ysort_rois = {}
            region_ysort_normf = {}
            for r in regions:#find region cells that are actually within the whole brain roi, also re-save it so i dont have to do it again:)
                #region_cell_index = pd.read_csv(save_path + r + '_ysort_normf.csv', index_col = 0).index
                region_cell_index[r] = [n for n in region_cell_index[r] if n in region_cell_index['wholebrain']]#okay i forgot to save it, in the future redump my json file here
                region_ysort_rois[r] = ysort_rois.loc[region_cell_index[r]]
                region_ysort_normf[r] = wholebrain_ysort_normf.loc[region_cell_index[r]]
            #region_ysort_rois = {r: ysort_rois.loc[region_cell_index] for r in regions}
            #region_ysort_normf = {r: wholebrain_ysort_normf.loc[region_cell_index] for r in regions}

        # ACTUALLY START ANALYSIS
        # traces overview
        # tail_byframe = tail_df.drop('t_dt', axis=1).groupby('frame').std().tail_sum
        # plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, wholebrain_ysort_normf, 'norm F',
        #     wholebrain_ysort_rois, minbar=0, maxbar=1, byregion=False, stimulus_s=stimulus_s, tail_byframe=tail_byframe)
        # plt.savefig(img_save_path + '10_trace_normf.png')
        # plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, region_ysort_normf, 'norm F',
        #     region_ysort_rois, minbar=0, maxbar=1, byregion=True, stimulus_s=stimulus_s, tail_byframe=tail_byframe)
        # plt.savefig(img_save_path + '11_trace_normf_region.png')
        #
        # #example traces
        # plot_individual_plane.plot_trace_loc_example(frametimes_df, stimulus_df, refImg, wholebrain_ysort_normf, 'norm F',
        #     wholebrain_ysort_rois, minbar=0, maxbar=1, byregion=False, stimulus_s=stimulus_s, tail_byframe=tail_byframe)
        # plt.savefig(img_save_path + '20_exampletrace_normf.png')
        # plot_individual_plane.plot_trace_loc_example(frametimes_df, stimulus_df, refImg, region_ysort_normf, 'norm F',
        #     region_ysort_rois, minbar = 0, maxbar = 1, byregion = True, stimulus_s = stimulus_s, tail_byframe = tail_byframe)
        # plt.savefig(img_save_path + '21_exampletrace_normf_region.png')
        #
        # #correlation and correlation clustering
        # plot_individual_plane.corr(wholebrain_ysort_normf, region_ysort_normf, 'norm F')
        # plt.savefig(img_save_path + '30_corr_normf.png')
        # cluster_dict = plot_individual_plane.corr_clustering(frametimes_df, stimulus_df, refImg, wholebrain_ysort_normf, 'norm F',
        #     wholebrain_ysort_rois, minbar = 0, maxbar = 1, stimulus_s = stimulus_s, tail_byframe = tail_byframe)
        # plt.savefig(img_save_path + '31_corr_normf_clustering.png')
        # cluster_dict = pd.DataFrame.from_dict(cluster_dict)

        #cell dynamics
        # if suite2p output, smooth and set mean on time for 5 to avoid noise; if caiman output, no need to smooth
        # so far, not plotting the whole brain but only the ROIs of interest (PT, nMLF, HBr)
        # ntf = pd.read_csv(save_path + 'neuron_tail_df_during.csv', index_col=0)
        # ntf = cleanup_csv(ntf, ntf.columns)
        # plot_individual_plane.specialplot_on(ntf, frametimes_df, wholebrain_ysort_normf,  wholebrain_ysort_rois, 'cluster', 0, 10,
        #                               refImg, False)
        # plt.savefig(img_save_path + '40_on_mean.png')
        _, mean_on_duration, mean_on_tailduration, mean_on_trace_hz = plot_individual_plane.plot_on(frametimes_df, {'wb': wholebrain_ysort_normf}, {'wb':wholebrain_ysort_rois}, 'mean', 0, 50, refImg, True, tail_bout_df)
        #plt.savefig(img_save_path + '40_on_mean.png')
        mean_on_trace_hz['wb'].to_csv(img_save_path + 'data/mean_on_trace.csv')
        mean_on_duration['wb'].to_csv(img_save_path + 'data/mean_on_duration.csv')
        mean_on_tailduration['wb'].to_csv(img_save_path + 'data/mean_on_tailduration.csv')
        #plot_individual_plane.plot_on_cont(frametimes_df, region_ysort_normf,  region_ysort_rois, 'mean_cont', 0, 30, 3, refImg, False)
        #plt.savefig(img_save_path + '41_on_cont_mean.png')
        # _, mean_on_duration = plot_individual_plane.plot_on_cont(frametimes_df, {'wb': wholebrain_ysort_normf}, 'norm F',{'wb':wholebrain_ysort_rois}, 'peak_cont', 0, 30, 1, refImg, False)
        # mean_on_duration.to_csv(img_save_path + 'data/peak_mean_on_duration.csv')
        # plt.savefig(img_save_path + '42_on_cont_peak.png')

        #find tail_responding neurons
        #neuron_tail_df = plot_individual_plane.find_tail_neuron(frametimes_df, wholebrain_ysort_normf, tail_window_s=0.5, tail_bout_df=tail_bout_df, timescale = 'during')
        #neuron_tail_df.to_csv(img_save_path + 'data/neuron_tail_df_during.csv')
        # neuron_tail_df = plot_individual_plane.find_tail_neuron(frametimes_df, wholebrain_ysort_normf,tail_window_s=0.5, tail_bout_df=tail_bout_df, timescale = 'stop')
        # neuron_tail_df.to_csv(img_save_path + 'data/neuron_tail_df_stop.csv')
        # neuron_tail_df = plot_individual_plane.find_tail_neuron(frametimes_df, wholebrain_ysort_normf, tail_window_s=0.5, tail_bout_df=tail_bout_df, timescale = 'start')
        # neuron_tail_df.to_csv(img_save_path + 'data/neuron_tail_df_start.csv')
        # plot_individual_plane.plot_loc_tail_neuron(neuron_tail_df, refImg, wholebrain_ysort_rois, top_percentage=top_percentage)
        # plt.savefig(img_save_path + '50_neuron_tail_location_start.png')
        # plot_individual_plane.plot_trace_tail_neuron(tail_df, tail_bout_df, frametimes_df, neuron_tail_df, wholebrain_ysort_normf, top_percentage = top_percentage, tail_window_s = 5)
        # plt.savefig(img_save_path + '51_neuron_tail_trace_start.png')
        #neuron_tail_df = plot_individual_plane.find_pearson_tail_neuron(wholebrain_ysort_normf, tail_df)
        #neuron_tail_df.to_csv(img_save_path + 'data/neuron_tail_df_pearson.csv')
        #neuron_tail_df = plot_individual_plane.find_pearson_tail_neuron_stop(wholebrain_ysort_normf, tail_bout_df)
        #neuron_tail_df.to_csv(img_save_path + 'data/neuron_tail_df_pearson_stop.csv')
        # plot_individual_plane.plot_loc_pearson_tail_neuron(neuron_tail_df, refImg, ysort_rois, top_percentage = top_percentage)
        # plt.savefig(img_save_path + '52_neuron_tail_location_pearson.png')
            # maintainer = plot_individual_plane.plot_property_tail_neuron(refImg, ysort_rois, neuron_tail_df, top_percentage = top_percentage)
            # plt.savefig(img_save_path + '92_neuron_tail_property.png')
            # plot_individual_plane.plot_trace_tail_neuron(tail_df, tail_bout_df, frametimes_df, maintainer, ysort_normf, top_percentage=1, tail_window_s=5)
            # plt.savefig(img_save_path + '93_neuron_tail_trace_maintainer.png')


        plt.clf()

#[0: tail_planes, 1: neuron_planes, 2: strength_boundary, 3: stimulus_s]
fish_dict = {
            #'danionella_fish0': [list(range(0, 19)), list(range(0, 19)), 0, 5],
            #'danionella_fish3': [list(range(0, 22)), list(range(0, 19)), 0, 5],
            #'danionella_fish4': [list(range(0, 9)), list(range(0, 9)), 0, 5],
            #'danionella_fish5': [list(range(0, 12)), list(range(0, 12)), 0, 5],
            'danionella_fish10': [list(range(9, 20)), list(range(9, 19)), 0.25, 5],
            'danionella_fish11': [list(range(0, 20)), list(range(1, 19)), 0.33, 5],
            #'danionella_fish14': [[7, 8, 9, 10, 11, 18, 19], [7, 8, 9, 10, 18, 19], 0.16, 15],#[7, 8, 9, 10, 11, 18, 19]
            # 'danionella_fish17': [list(range(9, 10)), list(range(9, 10)), 0.1, 15]
            'zebrafish_fish1': [list(range(0, 14)), list(range(3, 14)), 0.4, 5],
             'zebrafish_fish4': [list(range(0, 19)), list(range(2, 18)), 0.2, 5],
             'zebrafish_fish5': [list(range(0, 16)), list(range(2, 16)), 0.25, 5],
             'zebrafish_fish6': [list(range(0, 19)), list(range(1, 19)), 0.25, 5]
              }

for fish, params in tqdm.tqdm(fish_dict.items(), 'fish progress'):
    #cycle_onefish_tail(fish = fish, planerange = params[0], strength_boundary= params[2], stimulus_s = params[3])
    cycle_onefish_neuron(fish = fish, planerange = params[1], top_percentage = 0.2, stimulus_s = params[3])