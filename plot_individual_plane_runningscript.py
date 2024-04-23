"""
Run through all planes and plot all graphs:)

Note that data needs to go through load_fish_data.py so they are in (faster) csv forms

@Zichen He 20240313
"""

import sys
import pandas as pd
from matplotlib import pyplot as plt
from fishy import BaseFish
import plot_individual_plane
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner

# change this sys path to your own computer
sys.path.append(r'miniforge3/envs/naumann_lab/Codes/caImageAnalysis-kaitlyn/')

for plane in list(range(0, 12)) + list(range(14, 23)):
    fish = 'danionella_fish6/plane_' + str(plane)#'danionella_20240322_fish3/plane_' + str(plane)
    save_path = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/data/'
    img_save_path = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/'

    #LOAD ALL DATA
    dateparse = lambda x: pd.to_datetime(x, format='%H:%M:%S.%f').time()
    regionlist = pd.read_csv(save_path + 'regions.csv', index_col=0).values.flatten().tolist()
    ysort_rois = pd.read_csv(save_path + 'ysort_rois.csv', index_col=0)
    ysort_f = pd.read_csv(save_path + 'ysort_f.csv', index_col=0)
    ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
    ysort_booldf_normf_baseline = pd.read_csv(save_path + 'ysort_booldf_baseline_normf.csv', index_col=0)
    ysort_booldf_normf_cluster = pd.read_csv(save_path + 'ysort_booldf_cluster_normf.csv', index_col=0)
    ysort_zdiff = pd.read_csv(save_path + 'ysort_zdiff.csv', index_col=0)
    ysort_booldf_zdiff_corr = pd.read_csv(save_path + 'ysort_booldf_corr_zdiff.csv', index_col=0)

    try:
        degree_zdiff = pd.read_csv(save_path + 'degree_zdiff.csv', index_col=0, dtype = float).iloc[:, 0].to_list()
        degreeval_zdiff = pd.read_csv(save_path + 'degreeval_zdiff.csv', index_col=0, dtype = float).iloc[:, 0].to_list()
    except:
        degree_zdiff = []
        degreeval_zdiff = []
    degree_dict_zdiff = pd.read_csv(save_path + 'degree_dict_zdiff.csv', index_col=0, dtype = float).to_dict(orient='list')
    degree_dict_zdiff = {int(key): degree_dict_zdiff[key] for key in degree_dict_zdiff.keys()}
    response_dict_zdiff = pd.read_csv(save_path + 'response_dict_zdiff.csv', index_col=0).to_dict(orient='list')
    response_dict_zdiff = {int(key): response_dict_zdiff[key] for key in response_dict_zdiff.keys()}
    try:
        degree_normf_baseline = pd.read_csv(save_path + 'degree_normf_baseline.csv', index_col=0, dtype=float).iloc[:, 0].to_list()
        degreeval_normf_baseline = pd.read_csv(save_path + 'degreeval_normf_baseline.csv', index_col=0, dtype=float).iloc[:, 0].to_list()
    except:
        degree_normf_baseline = []
        degreeval_normf_baseline = []
    degree_dict_normf_baseline = pd.read_csv(save_path + 'degree_dict_normf_baseline.csv', index_col=0, dtype=float).to_dict(
        orient='list')
    degree_dict_normf_baseline = {int(key): degree_dict_normf_baseline[key] for key in degree_dict_normf_baseline.keys()}
    response_dict_normf_baseline = pd.read_csv(save_path + 'response_dict_normf_baseline.csv', index_col=0).to_dict(orient='list')
    response_dict_normf_baseline = {int(key): response_dict_normf_baseline[key] for key in response_dict_normf_baseline.keys()}
    try:
        degree_normf_cluster = pd.read_csv(save_path + 'degree_normf_cluster.csv', index_col=0, dtype=float).iloc[:,0].to_list()
        degreeval_normf_cluster = pd.read_csv(save_path + 'degreeval_normf_cluster.csv', index_col=0, dtype=float).iloc[:,0].to_list()
    except:
        degree_normf_cluster = []
        degreeval_normf_cluster = []
    degree_dict_normf_cluster = pd.read_csv(save_path + 'degree_dict_normf_cluster.csv', index_col=0,dtype=float).to_dict(
        orient='list')
    degree_dict_normf_cluster = {int(key): degree_dict_normf_cluster[key] for key in degree_dict_normf_cluster.keys()}
    response_dict_normf_cluster = pd.read_csv(save_path + 'response_dict_normf_cluster.csv', index_col=0).to_dict(
        orient='list')
    response_dict_normf_cluster = {int(key): response_dict_normf_cluster[key] for key in response_dict_normf_cluster.keys()}
    frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0, parse_dates=[1], date_parser=dateparse)
    stimulus_df = pd.read_csv(save_path + 'stimulus_df.csv', index_col=0)
    offsets = pd.read_csv(save_path + 'offsets.csv', index_col=0)
    offsets = (offsets.iloc[0, 0], offsets.iloc[1, 0])
    refImg = pd.read_csv(save_path + 'refImg.csv', index_col=0)

    region_ysort_rois = {}
    region_ysort_f = {}
    region_ysort_normf = {}
    region_ysort_booldf_normf_baseline = {}
    region_ysort_booldf_normf_cluster = {}
    region_ysort_zdiff = {}
    region_ysort_booldf_zdiff_corr = {}

    region_degree_zdiff = {region: None for region in regionlist}
    region_degreeval_zdiff = {region: None for region in regionlist}
    region_degree_dict_zdiff = {region: None for region in regionlist}
    region_response_dict_zdiff = {region: None for region in regionlist}
    region_degree_normf_baseline = {region: None for region in regionlist}
    region_degreeval_normf_baseline = {region: None for region in regionlist}
    region_degree_dict_normf_baseline = {region: None for region in regionlist}
    region_response_dict_normf_baseline = {region: None for region in regionlist}
    region_degree_normf_cluster = {region: None for region in regionlist}
    region_degreeval_normf_cluster = {region: None for region in regionlist}
    region_degree_dict_normf_cluster = {region: None for region in regionlist}
    region_response_dict_normf_cluster = {region: None for region in regionlist}
    for region in regionlist:
        region_ysort_rois[region] = pd.read_csv(save_path + region + '_ysort_rois.csv', index_col=0)
        region_ysort_f[region] = pd.read_csv(save_path + region + '_ysort_f.csv', index_col=0)
        region_ysort_normf[region] = pd.read_csv(save_path + region + '_ysort_normf.csv', index_col=0)
        region_ysort_booldf_normf_baseline[region] = pd.read_csv(save_path + region + '_ysort_booldf_baseline_normf.csv', index_col=0)
        region_ysort_booldf_normf_cluster[region] = pd.read_csv(save_path + region + '_ysort_booldf_cluster_normf.csv', index_col=0)
        region_ysort_zdiff[region] = pd.read_csv(save_path + region + '_ysort_zdiff.csv', index_col=0)
        region_ysort_booldf_zdiff_corr[region] = pd.read_csv(save_path + region + '_ysort_booldf_corr_zdiff.csv', index_col=0)

        try:
            region_degree_zdiff[region] = pd.read_csv(save_path + region + '_degree_zdiff.csv', index_col=0, dtype = float).iloc[:, 0].to_list()
            region_degreeval_zdiff[region] = pd.read_csv(save_path + region + '_degreeval_zdiff.csv', index_col=0, dtype=float).iloc[:, 0].to_list()

        except:
            region_degree_zdiff[region] = []
            region_degreeval_zdiff[region] = []
        region_degree_dict_zdiff[region] = pd.read_csv(save_path + region + '_degree_dict_zdiff.csv',
                                                       index_col=0, dtype=float).to_dict(orient='list')
        region_degree_dict_zdiff[region] = {int(key): region_degree_dict_zdiff[region][key] for key in
                                            region_degree_dict_zdiff[region].keys()}
        region_response_dict_zdiff[region] = pd.read_csv(save_path + region + '_response_dict_zdiff.csv',
                                                         index_col=0).to_dict(orient='list')
        region_response_dict_zdiff[region] = {int(key): region_response_dict_zdiff[region][key] for key in
                                              region_response_dict_zdiff[region].keys()}
        try:
            region_degree_normf_baseline[region] = pd.read_csv(save_path + region + '_degree_normf_baseline.csv', index_col=0,
                                                  dtype=float).iloc[:, 0].to_list()
            region_degreeval_normf_baseline[region] = pd.read_csv(save_path + region + '_degreeval_normf_baseline.csv', index_col=0,
                                                     dtype=float).iloc[:, 0].to_list()
        except:
            region_degree_normf_baseline[region] = []
            region_degreeval_normf_baseline[region] = []
        region_degree_dict_normf_baseline[region] = pd.read_csv(save_path + region + '_degree_dict_normf_baseline.csv',
                                                       index_col=0, dtype=float).to_dict(orient='list')
        region_degree_dict_normf_baseline[region] = {int(key): region_degree_dict_normf_baseline[region][key] for key in
                                            region_degree_dict_normf_baseline[region].keys()}
        region_response_dict_normf_baseline[region] = pd.read_csv(save_path + region + '_response_dict_normf_baseline.csv',
            index_col=0).to_dict(orient='list')
        region_response_dict_normf_baseline[region] = {int(key): region_response_dict_normf_baseline[region][key] for
                                                      key in region_response_dict_normf_baseline[region].keys()}
        try:
            region_degree_normf_cluster[region] = pd.read_csv(save_path + region + '_degree_normf_cluster.csv',
                                                               index_col=0,dtype=float).iloc[:, 0].to_list()
            region_degreeval_normf_cluster[region] = pd.read_csv(save_path + region + '_degreeval_normf_cluster.csv',
                                                                  index_col=0, dtype=float).iloc[:, 0].to_list()
        except:
            region_degree_normf_cluster[region] = []
            region_degreeval_normf_cluster[region] = []
        region_degree_dict_normf_cluster[region] = pd.read_csv(save_path + region + '_degree_dict_normf_cluster.csv',
                                                                index_col=0, dtype=float).to_dict(orient='list')
        region_degree_dict_normf_cluster[region] = {int(key): region_degree_dict_normf_cluster[region][key] for key in
                                                     region_degree_dict_normf_cluster[region].keys()}
        region_response_dict_normf_cluster[region] = pd.read_csv(save_path + region + '_response_dict_normf_cluster.csv',
            index_col=0).to_dict(orient='list')
        region_response_dict_normf_cluster[region] = {int(key): region_response_dict_normf_cluster[region][key] for
                                                       key in region_response_dict_normf_cluster[region].keys()}
    ysort_stim_dict_f = {}
    ysort_stim_dict_normf = {}
    ysort_stim_dict_zdiff = {}
    stimlist = list(stimulus_df.stim_name.unique())
    region_ysort_stim_dict_f = {stim: {region: None for region in regionlist} for stim in stimlist}
    region_ysort_stim_dict_normf = {stim: {region: None for region in regionlist} for stim in stimlist}
    region_ysort_stim_dict_zdiff = {stim: {region: None for region in regionlist} for stim in stimlist}
    for stim in stimlist:
        ysort_stim_dict_f[stim] = pd.read_csv(save_path + stim + '_ysort_stim_dict_f.csv', index_col=0)
        ysort_stim_dict_normf[stim] = pd.read_csv(save_path + stim + '_ysort_stim_dict_normf.csv', index_col=0)
        ysort_stim_dict_zdiff[stim] = pd.read_csv(save_path + stim + '_ysort_stim_dict_zdiff.csv', index_col=0)
        for region in regionlist:
            region_ysort_stim_dict_f[stim][region] = pd.read_csv(save_path + stim + region + '_ysort_stim_dict_f.csv',
                                                                 index_col=0)
            region_ysort_stim_dict_normf[stim][region] = pd.read_csv(
                save_path + stim + region + '_ysort_stim_dict_normf.csv', index_col=0)
            region_ysort_stim_dict_zdiff[stim][region] = pd.read_csv(
                save_path + stim + region + '_ysort_stim_dict_zdiff.csv', index_col=0)

    #ACTUALLY START DRAWING
    plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, ysort_f, 'raw', ysort_rois, minbar = 0, maxbar = 300, byregion = False)
    plt.savefig(img_save_path + '00_trace_raw_all.png')
    plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, region_ysort_f, 'raw', region_ysort_rois, minbar = 0, maxbar = 300, byregion = True)
    plt.savefig(img_save_path + '01_trace_raw_region.png')
    plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, ysort_normf, 'norm F', ysort_rois, minbar = 0, maxbar = 1, byregion = False)
    plt.savefig(img_save_path + '02_trace_normf_all.png')
    plot_individual_plane.plot_trace_loc(frametimes_df, stimulus_df, refImg, region_ysort_normf, 'norm F', region_ysort_rois, minbar = 0, maxbar = 1, byregion = True)
    plt.savefig(img_save_path + '03_trace_normf_region.png')

    plot_individual_plane.plot_trace_loc_example(frametimes_df, stimulus_df, refImg, ysort_normf, 'norm F', ysort_rois, minbar = 0, maxbar = 1, byregion = False)
    plt.savefig(img_save_path + '10_exampletrace_normf_all.png')
    plot_individual_plane.plot_trace_loc_example(frametimes_df, stimulus_df, refImg, region_ysort_normf, 'norm F', region_ysort_rois, minbar = 0, maxbar = 1, byregion = True)
    plt.savefig(img_save_path + '11_exampletrace_normf_region.png')

    plot_individual_plane.corr(ysort_normf, region_ysort_normf, 'norm F')
    plt.savefig(img_save_path + '20_corr_normf.png')
    plot_individual_plane.corr(ysort_zdiff, region_ysort_zdiff, 'zdiff F')
    plt.savefig(img_save_path + '21_corr_zdiff.png')
    #plot_individual_plane.corr_clustering(frametimes_df, stimulus_df, refImg, ysort_normf, 'norm F', ysort_rois)
    #plt.savefig(img_save_path + '22_corr_normf_clustering.png')

    plot_individual_plane.plot_trace_stimuli(frametimes_df, offsets, ysort_stim_dict_normf, 'norm F', ysort_rois, minbar = 0, maxbar = 1, byregion = False)
    plt.savefig(img_save_path + '30_stimtrace_normf_all.png')
    plot_individual_plane.plot_trace_stimuli(frametimes_df, offsets, region_ysort_stim_dict_normf, 'norm F', region_ysort_rois, minbar = 0, maxbar = 1, byregion = True)
    plt.savefig(img_save_path + '31_stimtrace_normf_region.png')
    plot_individual_plane.plot_trace_stimuli(frametimes_df, offsets, ysort_stim_dict_zdiff, 'zdiff F', ysort_rois, minbar = -2, maxbar = 2, byregion = False)
    plt.savefig(img_save_path + '32_stimtrace_zdiff_all.png')
    plot_individual_plane.plot_trace_stimuli(frametimes_df, offsets, region_ysort_stim_dict_zdiff, 'zdiff F', region_ysort_rois, minbar = -2, maxbar = 2, byregion = True)
    plt.savefig(img_save_path + '33_stimtrace_zdiff_region.png')

    plot_individual_plane.plot_tuning_distribution_x('zdiff F corr', ysort_booldf_zdiff_corr, ysort_rois, degree_dict_zdiff,
                                                     response_dict_zdiff, region_ysort_booldf_zdiff_corr, region_ysort_rois,
                                                     region_degree_dict_zdiff, region_response_dict_zdiff, refImg)
    plt.savefig(img_save_path + '40_stimdist_zdiff_corr')
    plot_individual_plane.plot_tuning_distribution_x('norm F baseline', ysort_booldf_normf_baseline, ysort_rois, degree_dict_normf_baseline,
                                                     response_dict_normf_baseline, region_ysort_booldf_normf_baseline,
                                                     region_ysort_rois,
                                                     region_degree_dict_normf_baseline, region_response_dict_normf_baseline, refImg)
    plt.savefig(img_save_path + '41_stimdist_normf_baseline')
    plot_individual_plane.plot_tuning_distribution_x('norm F cluster', ysort_booldf_normf_cluster, ysort_rois, degree_dict_normf_cluster,
                                                     response_dict_normf_cluster, region_ysort_booldf_normf_cluster,
                                                     region_ysort_rois,
                                                     region_degree_dict_normf_cluster, region_response_dict_normf_cluster, refImg)
    plt.savefig(img_save_path + '42_stimdist_normf_cluster')

    plot_individual_plane.plot_degree_tuned_distribution(degree_zdiff, degreeval_zdiff, region_degree_zdiff, region_degreeval_zdiff, 'zdiff F corr')
    plt.savefig(img_save_path + '50_tuneddegree_zdiff_corr')
    plot_individual_plane.plot_degree_tuned_distribution(degree_normf_baseline, degreeval_normf_baseline, region_degree_normf_baseline,
                                                         region_degreeval_normf_baseline, 'norm F baseline')
    plt.savefig(img_save_path + '51_tuneddegree_normf_baseline.png')
    plot_individual_plane.plot_degree_tuned_distribution(degree_normf_cluster, degreeval_normf_cluster, region_degree_normf_cluster,
                                                         region_degreeval_normf_cluster, 'norm F cluster')
    plt.savefig(img_save_path + '52_tuneddegree_normf_cluster.png')

    plot_individual_plane.plot_tuning_specificity(degree_dict_zdiff, response_dict_zdiff, ysort_booldf_zdiff_corr,
                            region_degree_dict_zdiff, region_response_dict_zdiff, region_ysort_booldf_zdiff_corr, 'zdiff F corr')
    plt.savefig(img_save_path + '60_stimgeneralization_zdiff_corr.png')
    plot_individual_plane.plot_tuning_specificity(degree_dict_normf_baseline, response_dict_normf_baseline, ysort_booldf_normf_baseline,
                                                  region_degree_dict_normf_baseline, region_response_dict_normf_baseline,
                                                  region_ysort_booldf_normf_baseline, 'norm F baseline')
    plt.savefig(img_save_path + '61_stimgeneralization_normf_baseline.png')
    plot_individual_plane.plot_tuning_specificity(degree_dict_normf_cluster, response_dict_normf_cluster, ysort_booldf_normf_cluster,
                                                  region_degree_dict_normf_cluster, region_response_dict_normf_cluster,
                                                  region_ysort_booldf_normf_cluster, 'norm F cluster')
    plt.savefig(img_save_path + '62_stimgeneralization_normf_cluster.png')

    plot_individual_plane.plot_on(frametimes_df, region_ysort_f, 'norm F', region_ysort_rois, 'mean', 30, refImg)
    plt.savefig(img_save_path + '70_on_mean.png')


    plt.clf()