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
planerange = list(range(0, 19))#list(range(0, 12)) + list(range(13, 23))
fishtype = 'workingfish'#"workingfish"， "workingfish_tail", "tailtrackedfish"
fish = 'danionella_fish8'
regionlist = ['PT', 'OT', 'HBr']
save_path_all = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/'
region_ROIs = {region: {'xpos': [], 'ypos': [], 'zpos': []} for region in regionlist}
stimulus_dfs = pd.DataFrame()
tail_dfs = pd.DataFrame()
maxframe = 0
mean_f = {key: None for key in planerange}
sdv_f = {key: None for key in planerange}
region_ysort_rois = {key: None for key in regionlist}
region_ysort_f = {key: None for key in regionlist}
region_ysort_normf = {key: None for key in regionlist}
meancorr_normf = {key: None for key in planerange}
region_meancorr_normf = {region: {key: None for key in planerange} for region in regionlist}
if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
    region_stim_dict_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.allcolor_dict.keys()} for region in regionlist}
    num_dir_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
    perc_dir_zdiff = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
    num_dir_normf_baseline = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
    perc_dir_normf_baseline = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
    num_dir_normf_cluster = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}
    perc_dir_normf_cluster = {region: {dir: {key: None for key in planerange} for dir in constants.monocular_dict.keys()} for region in regionlist}


#Collect all information
for plane in planerange:
    print(plane)
    save_path = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/plane_' + str(plane) + '/data/'

    # LOAD ALL DATA
    dateparse = lambda x: pd.to_datetime(x, format='%H:%M:%S.%f').time()
    region_available = pd.read_csv(save_path + 'regions.csv', index_col=0).values.flatten().tolist()
    ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
    frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0, parse_dates=[1], date_parser=dateparse)
    refImg = pd.read_csv(save_path + 'refImg.csv', index_col=0)
    if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
        stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)
        stimulus_df.frame = np.add(stimulus_df.frame, maxframe)
        stimulus_dfs = pd.concat([stimulus_dfs, stimulus_df])
        stimulus_dfs.reset_index(drop=True)
        stimlist = list(stimulus_df.stim_name.unique())
        offsets = pd.read_csv(save_path + 'offsets.csv', index_col=0)
        offsets = (offsets.iloc[0, 0], offsets.iloc[1, 0])
        region_ysort_booldf_zdiff = {region: pd.DataFrame() for region in regionlist}
        region_ysort_booldf_normf_baseline = {region: pd.DataFrame() for region in regionlist}
        region_ysort_booldf_normf_cluster = {region: pd.DataFrame() for region in regionlist}
    if fishtype == 'tailtrackedfish' or fishtype == 'workingfish_tail':
        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        tail_df.frame = np.add(tail_df.frame, maxframe)
        tail_dfs = pd.concat([tail_dfs, tail_df])
        tail_dfs.reset_index(drop=True)
    for region in region_available:
        ysort_rois = pd.read_csv(save_path + region + "_ysort_rois.csv", index_col=0)
        ysort_rois['zpos'] = [plane] * ysort_rois.shape[0]
        ysort_f = pd.read_csv(save_path + region + '_ysort_f.csv', index_col=0)
        ysort_f = ysort_f.set_index(np.add([plane * 0.01] * ysort_f.shape[0], ysort_f.index))
        ysort_normf = pd.read_csv(save_path + region + '_ysort_normf.csv', index_col=0)
        ysort_normf = ysort_f.set_index(np.add([plane * 0.01] * ysort_normf.shape[0], ysort_normf.index))
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
       # if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
       #    region_ysort_booldf_zdiff[region] = pd.read_csv(save_path + region + '_ysort_booldf_corr_zdiff.csv', index_col=0)
       #    region_ysort_booldf_normf_baseline[region] = pd.read_csv(save_path + region + '_ysort_booldf_baseline_normf.csv', index_col=0)
       #    region_ysort_booldf_normf_cluster[region] = pd.read_csv(save_path + region + '_ysort_booldf_cluster_normf.csv', index_col=0)
    # if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
    #     region_ysort_stim_dict_zdiff = {stim: {region: None for region in regionlist} for stim in stimlist}
    #     for stim in stimlist:
    #         for region in region_available:
    #             region_ysort_stim_dict_zdiff[stim][region] = pd.read_csv(
    #                 save_path + stim + region + '_ysort_stim_dict_zdiff.csv', index_col=0)

#     # collect raw fluorscence trace for every frame
#     mean_f[plane] = ysort_f.mean()
#     sdv_f[plane] = ysort_f.std()
#
#     # collect normalized cell traces for each region
#     for region in region_available:
#         for dir in constants.allcolor_dict.keys():
#             region_stim_dict_zdiff[region][dir][plane] = region_ysort_stim_dict_zdiff[dir][region]
#
#     #collect correlation
#     corr_matrix, region_corr_matrix = plot_individual_plane.corr(ysort_normf, region_ysort_normf, 'norm F')
#     meancorr_normf[plane] = corr_matrix.flatten()
#     for region in region_available:
#         region_meancorr_normf[region][plane] = region_corr_matrix[region].flatten()
#
#     # collect %dir responsive cells according to zdiff
#     if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
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
# compare_planes.planes_plot_trace(frametimes_df, planerange, mean_f, sdv_f, 'raw F')
# plt.savefig(save_path_all + '00_trace_rawF_acrossplane.png')
# i = 0
# for region in regionlist:
#     compare_planes.planes_plot_trace_stimuli(frametimes_df, offsets, region_stim_dict_zdiff[region], 'mean zdiff F', minbar = -2, maxbar = 2)
#     plt.savefig(save_path_all + '1' + str(i) + '_' + region + '_stimtrace_zdiff_acrossplane.png')
#     i += 1
#
# compare_planes.planes_plot_corr_dist(planerange, regionlist, meancorr_normf, region_meancorr_normf, 'norm F')
# plt.savefig(save_path_all + '20_corr_normf_acrossplane.png')
#
# compare_planes.planes_plot_tuning_num(frametimes_df, regionlist, planerange, num_dir_zdiff, 'zdiff F corr')
# plt.savefig(save_path_all + '30_stimnum_zdiff_corr_acrossplane.png')
# compare_planes.planes_plot_tuning_num(frametimes_df, regionlist, planerange, num_dir_normf_baseline, 'norm F baseline')
# plt.savefig(save_path_all + '31_stimnum_normf_baseline_acrossplane.png')
# compare_planes.planes_plot_tuning_num(frametimes_df, regionlist, planerange, num_dir_normf_cluster, 'norm F cluster')
# plt.savefig(save_path_all + '32_stimnum_normf_cluster_acrossplane.png')
#
# compare_planes.planes_plot_tuning_perc(frametimes_df, regionlist, planerange, perc_dir_zdiff,'zdiff F corr')
# plt.savefig(save_path_all + '40_stimperc_zdiff_acrossplane.png')
# compare_planes.planes_plot_tuning_perc(frametimes_df, regionlist, planerange, perc_dir_normf_baseline,'norm F baseline')
# plt.savefig(save_path_all + '41_stimperc_normf_baseline_acrossplane.png')
# compare_planes.planes_plot_tuning_perc(frametimes_df, regionlist, planerange, perc_dir_normf_cluster,'norm F cluster')
# plt.savefig(save_path_all + '42_stimperc_normf_cluster_acrossplane.png')

color_cutoff, mean_on_duration = plot_individual_plane.plot_on(frametimes_df, region_ysort_f, 'norm F',
                                                               region_ysort_rois, 'mean', 0, 30, refImg, True)
plt.savefig(save_path_all + '50_on_mean_acrossplane.png')
fig = compare_planes.volumetric_plot(color_cutoff, region_ROIs, mean_on_duration, region_ysort_rois)
fig.write_html(save_path_all + "51_on_mean_acrossplane.html")

color_cutoff, mean_on_duration = plot_individual_plane.plot_on_cont(frametimes_df, region_ysort_normf, 'norm F',
                                                             region_ysort_rois, 'mean_cont',0, 30, 3, refImg, False)
plt.savefig(save_path_all + '52_on_cont_mean.png')
fig = compare_planes.volumetric_plot(color_cutoff, region_ROIs, mean_on_duration, region_ysort_rois)
fig.write_html(save_path_all + "53_on_mean_cont_acrossplane.html")

color_cutoff, mean_on_duration = plot_individual_plane.plot_on_cont(frametimes_df, region_ysort_normf, 'peak F',
                                                             region_ysort_rois, 'mean_cont',0, 30, 8, refImg, False)
plt.savefig(save_path_all + '54_on_cont_mean.png')
fig = compare_planes.volumetric_plot(color_cutoff, region_ROIs, mean_on_duration, region_ysort_rois)
fig.write_html(save_path_all + "56_on_peak_cont_acrossplane.html")

#打入冷宫
# color_cutoff, fr_per_min_max = plot_individual_plane.plot_fr(frametimes_df, region_ysort_normf, "norm F", region_ysort_rois, 0, 4.8, 25, refImg)
# plt.savefig(save_path_all + '54_fr.png')
# fig = compare_planes.volumetric_plot(color_cutoff, region_ROIs, fr_per_min_max, region_ysort_rois)
# fig.write_html(save_path_all + "55_fr_acrossplane.html")

# if fishtype == 'tailtrackedfish' or fishtype == 'workingfish_tail':
#     tail_byframe, tail_bout_df = \
#         plot_individual_plane.analyze_tail(frametimes_df, stimulus_dfs, tail_dfs, 5, 0.2)
#     plt.savefig(save_path_all + '80_tail_peak_acrossplane.png')
# if fishtype == 'workingfish_tail':
#     plot_individual_plane.tail_angle_all(stimulus_dfs, tail_bout_df, 5)
#     plt.savefig(save_path_all + '81_tail_all_acrossplane.png')
#     plot_individual_plane.tail_angle_binocular(tail_bout_df, stimuli_s = 5)
#     plt.savefig(save_path_all + '82_tail_binocular.png')