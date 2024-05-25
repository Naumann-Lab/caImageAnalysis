"""
The running script for compare_planes. So far, the data depends on the csv out put of load_fish_data.py, but this
can be easily modified to directly fetch file from fish (will probably take forever).

@Zichen He 240313
"""
from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import sys

# change this sys path to your own computer
sys.path.append(r'miniforge3/envs/naumann_lab/Codes/caImageAnalysis-kaitlyn/')

import constants, plot_individual_plane
from fishy import BaseFish
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner

import compare_planes

#gather all information space
planerange = list(range(9, 20))
fishtype = 'workingfish_tail'#"workingfish"， "workingfish_tail", "tailtrackedfish"
fish = 'danionella_fish10'
strength_boundary = 0.25
regionlist = ['PT', 'OT', 'HBr']
save_path_all = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/'
region_ROIs = {region: {'xpos': [], 'ypos': [], 'zpos': []} for region in regionlist}
stimulus_dfs = pd.DataFrame()
tail_dfs = pd.DataFrame()
frametimes_dfs = pd.DataFrame()
neuron_tail_dfs = pd.DataFrame()
maxframe = 0
mean_f = {key: None for key in planerange}
sdv_f = {key: None for key in planerange}
meancorr_normf = {key: None for key in planerange}
region_ysort_rois = {key: None for key in regionlist}
region_ysort_f = {key: None for key in regionlist}
region_ysort_normf = {key: None for key in regionlist}
region_meancorr_normf = {region: {key: None for key in planerange} for region in regionlist}
region_n_neuron = {key: {plane: None for plane in planerange} for key in regionlist}
# if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
#     region_stim_dict_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.allcolor_dict.keys()} for region in regionlist}
#     num_dir_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
#     perc_dir_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
#     num_dir_normf_baseline = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
#     perc_dir_normf_baseline = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
#     num_dir_normf_cluster = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
#     perc_dir_normf_cluster = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}


#Collect all information
for plane in planerange:
    print(plane)
    save_path = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/plane_' + str(plane) + '/data/'

    # LOAD ALL DATA
    dateparse = lambda x: pd.to_datetime(x, format='%H:%M:%S.%f').time()
    region_available = pd.read_csv(save_path + 'regions.csv', index_col=0).values.flatten().tolist()
    # ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
    frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0, parse_dates=[1], date_parser=dateparse)
    frametimes_dfs = pd.concat([frametimes_dfs, frametimes_df])
    frametimes_dfs = frametimes_dfs.reset_index(drop=True)
    refImg = pd.read_csv(save_path + 'refImg.csv', index_col=0)
    if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
         stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)
         stimulus_df.frame = np.add(stimulus_df.frame, maxframe)
         stimulus_dfs = pd.concat([stimulus_dfs, stimulus_df])
         stimulus_dfs = stimulus_dfs.reset_index(drop=True)
         stimlist = list(stimulus_df.stim_name.unique())
         offsets = pd.read_csv(save_path + 'offsets.csv', index_col=0)
         offsets = (offsets.iloc[0, 0], offsets.iloc[1, 0])
         # region_ysort_booldf_zdiff = {region: pd.DataFrame() for region in regionlist}
         # region_ysort_booldf_normf_baseline = {region: pd.DataFrame() for region in regionlist}
         # region_ysort_booldf_normf_cluster = {region: pd.DataFrame() for region in regionlist}
         for region in region_available:
             ysort_rois = pd.read_csv(save_path + region + "_ysort_rois.csv", index_col=0)
             ysort_rois['zpos'] = [plane] * ysort_rois.shape[0]
             ysort_rois = ysort_rois.set_index(np.add([plane * 0.01] * ysort_rois.shape[0], ysort_rois.index))
             ysort_f = pd.read_csv(save_path + region + '_ysort_f.csv', index_col=0)
             ysort_f = ysort_f.set_index(np.add([plane * 0.01] * ysort_f.shape[0], ysort_f.index))
             ysort_normf = pd.read_csv(save_path + region + '_ysort_normf.csv', index_col=0)
             ysort_normf = ysort_normf.set_index(np.add([plane * 0.01] * ysort_normf.shape[0], ysort_normf.index))
             region_n_neuron[region][plane] = ysort_normf.shape[0]
             try:
                 region_ysort_f[region] = pd.concat([region_ysort_f[region], ysort_f])
                 region_ysort_normf[region] = pd.concat([region_ysort_normf[region], ysort_normf])
                 region_ysort_rois[region] = pd.concat([region_ysort_rois[region], ysort_rois])
             except TypeError:
                 region_ysort_f[region] = ysort_f
                 region_ysort_normf[region] = ysort_normf
                 region_ysort_rois[region] = ysort_rois
             region_ROI = pd.read_csv(save_path[:-5] + 'ROIs/' + region + '.csv')
             region_ROIs[region]['xpos'] = region_ROIs[region]['xpos'] + list(region_ROI.loc[:, 'X'])
             region_ROIs[region]['ypos'] = region_ROIs[region]['ypos'] + list(region_ROI.loc[:, 'Y'])
             region_ROIs[region]['zpos'] = region_ROIs[region]['zpos'] + [plane] * region_ROI.shape[0]
         #     region_ysort_booldf_zdiff[region] = pd.read_csv(save_path + region + '_ysort_booldf_corr_zdiff.csv', index_col=0)
         #     region_ysort_booldf_normf_baseline[region] = pd.read_csv(save_path + region + '_ysort_booldf_baseline_normf.csv', index_col=0)
         #     region_ysort_booldf_normf_cluster[region] = pd.read_csv(save_path + region + '_ysort_booldf_cluster_normf.csv', index_col=0)
         # region_ysort_stim_dict_zdiff = {stim: {region: None for region in regionlist} for stim in stimlist}
    #     for stim in stimlist:
    #         for region in region_available:
    #             region_ysort_stim_dict_zdiff[stim][region] = pd.read_csv(
    #                 save_path + stim + region + '_ysort_stim_dict_zdiff.csv', index_col=0)
    if fishtype == 'tailtrackedfish' or fishtype == 'workingfish_tail':
        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0, parse_dates=['t_dt'], date_parser=dateparse)
        tail_df.frame = np.add(tail_df.frame, maxframe)
        tail_dfs = pd.concat([tail_dfs, tail_df])
        tail_dfs = tail_dfs.reset_index(drop=True)
        neuron_tail_df = pd.read_csv(save_path + '90_neuron_tail_df.csv', index_col=0)
        neuron_tail_df = neuron_tail_df.set_index(np.add([plane * 0.01] * len(neuron_tail_df), neuron_tail_df.index))
        neuron_tail_dfs = pd.concat([neuron_tail_dfs, neuron_tail_df])
    # supplement the non-tailtracked fish
    tail_hz = hzReturner(tail_df)

    #collect %dir responsive cells according to zdiff
    if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
        # collect raw fluorscence trace for every frame
        mean_f[plane] = ysort_f.mean()
        sdv_f[plane] = ysort_f.std()
    #     for region in region_available:
    #         for dir in constants.allcolor_dict.keys():
    #             region_stim_dict_zdiff[region][dir][plane] = region_ysort_stim_dict_zdiff[dir][region]
    #     for region in region_available:
    #         booldf_to_count_zdiff = region_ysort_booldf_zdiff[region]
    #         booldf_to_count_normf_baseline = region_ysort_booldf_normf_baseline[region]
    #         booldf_to_count_normf_cluster = region_ysort_booldf_normf_cluster[region]
    #         num_total = booldf_to_count_zdiff.shape[0]
    #         for dir in constants.monocular_dict.keys():
    #             num_response_zdiff = booldf_to_count_zdiff[booldf_to_count_zdiff[dir] == True].shape[0]
    #             num_dir_zdiff[region][dir][plane] = num_response_zdiff
    #             perc_dir_zdiff[region][dir][plane] = np.divide(num_response_zdiff, num_total)
    #             num_response_normf_baseline = booldf_to_count_normf_baseline[booldf_to_count_normf_baseline[dir] == True].shape[0]
    #             num_dir_normf_baseline[region][dir][plane] = num_response_normf_baseline
    #             perc_dir_normf_baseline[region][dir][plane] = np.divide(num_response_normf_baseline, num_total)
    #             num_response_normf_cluster = booldf_to_count_normf_cluster[booldf_to_count_normf_cluster[dir] == True].shape[0]
    #             num_dir_normf_cluster[region][dir][plane] = num_response_normf_cluster
    #             perc_dir_normf_cluster[region][dir][plane] = np.divide(num_response_normf_cluster, num_total)
    maxframe = maxframe + frametimes_df.shape[0]

#PLOTTING EVERYTHING
compare_planes.planes_plot_trace(frametimes_dfs, planerange, mean_f, sdv_f, 'raw F')
plt.savefig(save_path_all + '00_trace_rawF_acrossplane.png')
# i = 0
# for region in regionlist:
#     compare_planes.planes_plot_trace_stimuli(frametimes_dfs, offsets, region_stim_dict_zdiff[region], 'mean zdiff F', minbar = -2, maxbar = 2)
#     plt.savefig(save_path_all + '1' + str(i) + '_' + region + '_stimtrace_zdiff_acrossplane.png')
#     i += 1
#
# compare_planes.planes_plot_tuning_num(frametimes_dfs, regionlist, planerange, num_dir_zdiff, 'zdiff F corr')
# plt.savefig(save_path_all + '30_stimnum_zdiff_corr_acrossplane.png')
# compare_planes.planes_plot_tuning_num(frametimes_dfs, regionlist, planerange, num_dir_normf_baseline, 'norm F baseline')
# plt.savefig(save_path_all + '31_stimnum_normf_baseline_acrossplane.png')
# compare_planes.planes_plot_tuning_num(frametimes_dfs, regionlist, planerange, num_dir_normf_cluster, 'norm F cluster')
# plt.savefig(save_path_all + '32_stimnum_normf_cluster_acrossplane.png')
#
# compare_planes.planes_plot_tuning_perc(frametimes_dfs, regionlist, planerange, perc_dir_zdiff,'zdiff F corr')
# plt.savefig(save_path_all + '40_stimperc_zdiff_acrossplane.png')
# compare_planes.planes_plot_tuning_perc(frametimes_dfs, regionlist, planerange, perc_dir_normf_baseline,'norm F baseline')
# plt.savefig(save_path_all + '41_stimperc_normf_baseline_acrossplane.png')
# compare_planes.planes_plot_tuning_perc(frametimes_dfs, regionlist, planerange, perc_dir_normf_cluster,'norm F cluster')
# plt.savefig(save_path_all + '42_stimperc_normf_cluster_acrossplane.png')
# color_cutoff, mean_on_duration = plot_individual_plane.plot_on(frametimes_dfs, region_ysort_normf, 'norm F',
#                                                                region_ysort_rois, 'mean', 0, 30, refImg, True)
# plt.savefig(save_path_all + '50_on_mean_acrossplane.png')
# fig = compare_planes.volumetric_plot(region_ROIs, mean_on_duration, region_ysort_rois)
# fig.write_html(save_path_all + "51_on_mean_acrossplane.html")
#
# color_cutoff, mean_on_duration = plot_individual_plane.plot_on_cont(frametimes_dfs, region_ysort_normf, 'norm F',
#                                                              region_ysort_rois, 'mean_cont',0, 30, 3, refImg, False)
# plt.savefig(save_path_all + '52_on_cont_mean.png')
# fig = compare_planes.volumetric_plot(region_ROIs, mean_on_duration, region_ysort_rois)
# fig.write_html(save_path_all + "53_on_mean_cont_acrossplane.html")
#
# color_cutoff, mean_on_duration = plot_individual_plane.plot_on_cont(frametimes_dfs, region_ysort_normf, 'peak F',
#                                                              region_ysort_rois, 'mean_cont',0, 30, 8, refImg, False)
# plt.savefig(save_path_all + '54_on_cont_mean.png')
# fig = compare_planes.volumetric_plot(region_ROIs, mean_on_duration, region_ysort_rois)
# fig.write_html(save_path_all + "56_on_peak_cont_acrossplane.html")


if fishtype == 'tailtrackedfish' or fishtype == 'workingfish_tail':
    tail_bout_df = \
        plot_individual_plane.analyze_tail(frametimes_dfs, stimulus_dfs, tail_dfs, tail_hz,  strength_boundary = strength_boundary)
    tail_bout_df.to_csv(save_path_all + 'tail_bout_df.csv')
    plt.savefig(save_path_all + '80_tail_peak_acrossplane.png')
if fishtype == 'workingfish_tail':
    plot_individual_plane.tail_angle_all(stimulus_dfs, tail_bout_df, stimuli_s = 5)
    plt.savefig(save_path_all + '81_tail_all_acrossplane.png')
    plot_individual_plane.tail_angle_binocular(tail_bout_df, stimuli_s = 5)
    plt.savefig(save_path_all + '82_tail_binocular.png')
    # tail_clustering_result = \
    #     compare_planes.planes_plot_tail_type_clusters(region_ysort_normf, tail_bout_df, tail_dfs, save_path_all, k_clusters=7)
    # tail_clustering_result.to_csv(save_path_all + '90_tail_clustering_result.csv')
    ysort_rois = pd.DataFrame()
    for region in region_ysort_rois:
         ysort_rois = pd.concat([ysort_rois, region_ysort_rois[region]])
    fig, fig_response, fig_predict, fig_scatter = compare_planes.volumetric_plot_tailneurons(neuron_tail_dfs, ysort_rois, plane_toppercentage = 0.2)
    fig_scatter.savefig(save_path_all + "96_tail_neuron.png")
    fig.write_html(save_path_all + "97_tail_neuron.html")
    fig_response.write_html(save_path_all + "98_tail_respond_neuron.html")
    fig_predict.write_html(save_path_all + "99_tail_predict_neuron.html")