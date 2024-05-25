"""
Load all planes of a fish to csv format so it can run through plot_individual_plane analysis pipeline. The code also
sorts the neurons with their y coordinates.

This was created mainly so I don't need to carry the massive amount of data in the harddrive around and to speed up the
process. If desired, one can easily modify the codes so that they can directly load data from fish to
the plot_individual_plane pipeline. I'm just lazy lol

@Zichen He 20240313
"""

from pathlib import Path
import pandas as pd

import sys

# change this sys path to your own computer
sys.path.append(r'miniforge3/envs/naumann_lab/caImageAnalysis/')

import stimuli
from fishy import WorkingFish, WorkingFish_Tail, TailTrackedFish

def sort_y(wf, signal, fishtype):
    """
    Sory all cells by their y location
        wf: working fish
        signal: the kind of signal that can be sorted and the way that booldf is generated
         ('raw', 'zdiff_corr', 'norm_baseline', 'norm_cluster')
        region: weather to sort cells by region
        fishtype: the fish type to decide if to sort stimulus array or not
    return:
        ysort_rois: the cell rois sorted by y
        ysort_cells: the cell signal sorted by y, in the form of dataframe
        region_ysort_rois: the dictionary contains dataframes of ysort_rois for each region
        region_ysort_cells: the dictionary contains dataframes of ysort_cells for each region
        ysort_stim_dict: the dictionary contains each stimulus containing the dataframes of cell traces in response window
        region_ysort_stim_dict: the dictionary contains each stimulus, which is a dictionary contains the dataframes of cell traces in response window for each region
        ysort_booldf: the ysorted dataframe of if cells are selective to certain direction
        region_ysort_booldf: the dictionary contains each region, in which contains ysort_booldf
    """
    cell_loc = pd.DataFrame(wf.return_cell_rois(range(len(wf.zdiff_cells))), columns = ['xpos', 'ypos'])
    ysort_rois = cell_loc.sort_values(by = ['ypos'])
    if signal == 'raw':
        cells = pd.DataFrame(wf.f_cells)
        # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
        #     ysort_stim_dict = {key: pd.DataFrame(wf.f_stim_dict[key]).T for key in wf.f_stim_dict.keys()}
        #     booldf = pd.DataFrame(data = None)
    elif signal == 'norm_cluster':
        cells = pd.DataFrame(wf.normf_cells)
        # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
        #     ysort_stim_dict = {key: pd.DataFrame(wf.normf_stim_dict[key]).T for key in wf.normf_stim_dict.keys()}
        #     booldf = pd.DataFrame(data=wf.normf_cluster_booldf)
    elif signal == 'norm_baseline':
        cells = pd.DataFrame(wf.normf_cells)
        # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
        #     ysort_stim_dict = {key: pd.DataFrame(wf.normf_stim_dict[key]).T for key in wf.normf_stim_dict.keys()}
        #     booldf = pd.DataFrame(data = wf.normf_baseline_booldf)
    elif signal == 'zdiff_corr':
        cells = pd.DataFrame(wf.zdiff_cells)
        # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
        #     ysort_stim_dict = {key: pd.DataFrame(wf.zdiff_stim_dict[key]).T for key in wf.zdiff_stim_dict.keys()}
        #     booldf = pd.DataFrame(data = wf.zdiff_corr_booldf)

    #sort cell traces
    cells['ypos'] = cell_loc['ypos']
    ysort_cells = cells.sort_values(by = ['ypos'])
    #orient ROIs from anterior to posterior, not necessary
    wf.roi_dict = dict(sorted(wf.roi_dict.items(), key = lambda x: x[1]['Y'].mean()))
    #get cell index from ROIs
    rois_cell_index = {key: wf.return_cells_by_location_imagej(wf.roi_dict[key])
                        for key in wf.roi_dict.keys()}
    region_ysort_rois = {key: ysort_rois.loc[rois_cell_index[key]].sort_values(by = ['ypos'])
                        for key in wf.roi_dict.keys()}
    region_ysort_cells = {key: ysort_cells.loc[rois_cell_index[key]].sort_values(by = ['ypos']).drop(['ypos'], axis = 1)
                        for key in wf.roi_dict.keys()}
    ysort_cells = ysort_cells.drop(['ypos'], axis = 1)

    #sort response to stimuli
    # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
    #     for stim in ysort_stim_dict.keys():
    #          ysort_stim_dict[stim]['ypos'] = cell_loc['ypos']
    #          ysort_stim_dict[stim] = ysort_stim_dict[stim].sort_values(by = ['ypos'])
    #     #divide cells to regions
    #     region_ysort_stim_dict = {key: {region: None for region in wf.roi_dict.keys()} for key in ysort_stim_dict.keys()}
    #     for stim in region_ysort_stim_dict.keys():
    #         for region in region_ysort_stim_dict[stim].keys():
    #             region_ysort_stim_dict[stim][region] = ysort_stim_dict[stim].loc[rois_cell_index[region]]
    #             region_ysort_stim_dict[stim][region] = region_ysort_stim_dict[stim][region].sort_values(by = ['ypos'])
    #             region_ysort_stim_dict[stim][region] = region_ysort_stim_dict[stim][region].drop(['ypos'], axis = 1)
    #         ysort_stim_dict[stim] = ysort_stim_dict[stim].drop(['ypos'], axis = 1)
    #
    #     #sort booldf to stimuli
    #     booldf['ypos'] = cell_loc['ypos']
    #     ysort_booldf = booldf.sort_values(by = ['ypos'])
    #     #divide cells to regions
    #     region_ysort_booldf = {region: None for region in wf.roi_dict.keys()}
    #     for region in region_ysort_booldf.keys():
    #         region_ysort_booldf[region] = ysort_booldf.loc[rois_cell_index[region]]
    #         region_ysort_booldf[region] = region_ysort_booldf[region].sort_values(by = ['ypos'])
    #         region_ysort_booldf[region] = region_ysort_booldf[region].drop(['ypos'], axis = 1)
    #     ysort_booldf = ysort_booldf.drop(['ypos'], axis = 1)

    # if fishtype == 'workingfish' or fishtype =='workingfish_tail':
    #     return ysort_rois, ysort_cells, region_ysort_rois, region_ysort_cells, ysort_stim_dict, region_ysort_stim_dict, \
    #            ysort_booldf, region_ysort_booldf
    # else:
    return ysort_rois, ysort_cells, region_ysort_rois, region_ysort_cells

def degree_tuned(wf, signal, roi, region_roi):
    """
    Return the degree tunning of all cells. Noted that results are probably not y-sorted, as i didn't bother to adjust it at this level lol
        wf: the workingfish object
        signal: the type of response we are using ('normf_baseline', 'normf_cluster' and 'zdiff_corr')
        roi: the dataframe containing all cells' locations, noted that the index column should match the cell identifier
        region_roi: the dictionary that contains all region and their correspoinding roi dataframe
    return:
    """
    degree, degreeval, degree_dict, response_dict = wf.return_degree_vectors(roi.index, signal)
    region_degree = {region: None for region in region_roi.keys()}
    region_degreeval = {region: None for region in region_roi.keys()}
    region_degree_dict = {region: None for region in region_roi.keys()}
    region_response_dict = {region: None for region in region_roi.keys()}
    for region in region_roi.keys():
        region_degree[region], region_degreeval[region], region_degree_dict[region], region_response_dict[region] =\
            wf.return_degree_vectors(region_roi[region].index, signal)
    return degree, degreeval, degree_dict, response_dict, \
           region_degree, region_degreeval, region_degree_dict, region_response_dict

for plane in list(range(0, 2)):
    fish = 'zebrafish_fish6/plane_' + str(plane)
    fishtype = 'workingfish_tail' #workingfish or workingfish_tail or tailtrackedfish
    example_folder = Path(r'/Volumes/WD_BLACK/20240507_ZF_elavl3g7f_wholebrain_simplified/fish6_ZF_wholebrain/plane_' + str(plane))
    save_path = '/Users/zichenhe/Desktop/Naumann Lab/' + fish + '/data/'
    midnight_noon = 'noon'#midnight or noon

    # make a WorkingFish object
    if fishtype =='tailtrackedfish':
        wf = TailTrackedFish(folder_path = example_folder,
                         frametimes_key="frametimes",
                         midnight_noon= midnight_noon,
                         invert= False,
                         )
    elif fishtype == 'workingfish':
        wf = WorkingFish(folder_path = example_folder,
                         frametimes_key="frametimes",
                         midnight_noon= midnight_noon,
                         stim_key="pstim",
                         stim_fxn=stimuli.pandastim_to_df,
                         invert= False,
                         used_offsets = (-7, 25),#(-7, 25),
                         stim_offset = 12,
                         baseline_offset = -5,
                         )
    elif fishtype == 'workingfish_tail':
        wf = WorkingFish_Tail(folder_path=example_folder,
                         frametimes_key="frametimes",
                         midnight_noon= midnight_noon, #dedicate for planes imaged 00:00-1:00AM
                         stim_key="pstim",
                         stim_fxn=stimuli.pandastim_to_df,
                         invert=False,
                         used_offsets=(-7, 25),  # (-7, 25),
                         stim_offset=12,
                         baseline_offset=-5,
                         )

    if fishtype == 'workingfish_tail' or fishtype == 'tailtrackedfish':
        wf.tail_df.to_csv(save_path + 'tail_df.csv')


    if fishtype == 'workingfish_tail' or fishtype == 'workingfish':
        # ysort_rois, ysort_f, region_ysort_rois, region_ysort_f, ysort_stim_dict_f, region_ysort_stim_dict_f, \
        #        ysort_booldf_f, region_ysort_booldf_f = sort_y(wf, 'raw', fishtype)
        # ysort_rois, ysort_normf, region_ysort_rois, region_ysort_normf, ysort_stim_dict_normf, region_ysort_stim_dict_normf, \
        #        ysort_booldf_baseline_normf, region_ysort_booldf_baseline_normf = sort_y(wf, 'norm_baseline', fishtype)
        # ysort_rois, ysort_normf, region_ysort_rois, region_ysort_normf, ysort_stim_dict_normf, region_ysort_stim_dict_normf, \
        #     ysort_booldf_cluster_normf, region_ysort_booldf_cluster_normf = sort_y(wf, 'norm_cluster', fishtype)
        # ysort_rois, ysort_zdiff, region_ysort_rois, region_ysort_zdiff, ysort_stim_dict_zdiff, region_ysort_stim_dict_zdiff, \
        #        ysort_booldf_corr_zdiff, region_ysort_booldf_corr_zdiff = sort_y(wf, 'zdiff_corr', fishtype)
    #else:
        ysort_rois, ysort_f, region_ysort_rois, region_ysort_f = sort_y(wf, 'raw', fishtype)
        ysort_rois, ysort_normf, region_ysort_rois, region_ysort_normf = sort_y(wf, 'norm_baseline', fishtype)
        ysort_rois, ysort_zdiff, region_ysort_rois, region_ysort_zdiff = sort_y(wf, 'zdiff_corr', fishtype)

    ysort_rois.to_csv(save_path + 'ysort_rois.csv')
    ysort_f.to_csv(save_path + 'ysort_f.csv')
    ysort_normf.to_csv(save_path + 'ysort_normf.csv')
    ysort_zdiff.to_csv(save_path + 'ysort_zdiff.csv')
    wf.frametimes_df.to_csv(save_path + 'frametimes_df.csv', date_format='%H:%M:%S.%f')
    pd.DataFrame(wf.ops["refImg"]).to_csv(save_path + 'refImg.csv')
    pd.DataFrame(wf.roi_dict.keys()).to_csv(save_path + 'regions.csv')

    if fishtype == 'workingfish_tail' or fishtype =='workingfish':
        # degree_zdiff, degreeval_zdiff, degree_dict_zdiff, response_dict_zdiff, \
        # region_degree_zdiff, region_degreeval_zdiff, region_degree_dict_zdiff, region_response_dict_zdiff = \
        #     degree_tuned(wf, 'zdiff_corr', ysort_rois, region_ysort_rois)
        # degree_normf_baseline, degreeval_normf_baseline, degree_dict_normf_baseline, response_dict_normf_baseline, \
        # region_degree_normf_baseline, region_degreeval_normf_baseline, region_degree_dict_normf_baseline, \
        # region_response_dict_normf_baseline = \
        #     degree_tuned(wf, 'normf_baseline', ysort_rois, region_ysort_rois)
        # degree_normf_cluster, degreeval_normf_cluster, degree_dict_normf_cluster, response_dict_normf_cluster, \
        # region_degree_normf_cluster, region_degreeval_normf_cluster, region_degree_dict_normf_cluster, \
        # region_response_dict_normf_cluster = \
        #     degree_tuned(wf, 'normf_cluster', ysort_rois, region_ysort_rois)
        # ysort_booldf_baseline_normf.to_csv(save_path + 'ysort_booldf_baseline_normf.csv')
        # ysort_booldf_cluster_normf.to_csv(save_path + 'ysort_booldf_cluster_normf.csv')
        # ysort_booldf_baseline_normf.to_csv(save_path + 'ysort_booldf_baseline_normf.csv')
        # ysort_booldf_cluster_normf.to_csv(save_path + 'ysort_booldf_cluster_normf.csv')
        # ysort_booldf_corr_zdiff.to_csv(save_path + 'ysort_booldf_corr_zdiff.csv')
        #
        # pd.DataFrame(degree_zdiff).to_csv(save_path + 'degree_zdiff.csv')
        # pd.DataFrame(degreeval_zdiff).to_csv(save_path + 'degreeval_zdiff.csv')
        # pd.DataFrame(degree_dict_zdiff).to_csv(save_path + 'degree_dict_zdiff.csv')
        # pd.DataFrame(response_dict_zdiff).to_csv(save_path + 'response_dict_zdiff.csv')
        # pd.DataFrame(degree_normf_baseline).to_csv(save_path + 'degree_normf_baseline.csv')
        # pd.DataFrame(degreeval_normf_baseline).to_csv(save_path + 'degreeval_normf_baseline.csv')
        # pd.DataFrame(degree_dict_normf_baseline).to_csv(save_path + 'degree_dict_normf_baseline.csv')
        # pd.DataFrame(response_dict_normf_baseline).to_csv(save_path + 'response_dict_normf_baseline.csv')
        # pd.DataFrame(degree_normf_cluster).to_csv(save_path + 'degree_normf_cluster.csv')
        # pd.DataFrame(degreeval_normf_cluster).to_csv(save_path + 'degreeval_normf_cluster.csv')
        # pd.DataFrame(degree_dict_normf_cluster).to_csv(save_path + 'degree_dict_normf_cluster.csv')
        # pd.DataFrame(response_dict_normf_cluster).to_csv(save_path + 'response_dict_normf_cluster.csv')
        wf.stimulus_df.to_csv(save_path + 'stimulus_df.csv')
        pd.DataFrame(wf.offsets).to_csv(save_path + 'offsets.csv')

    for region in region_ysort_rois.keys():
        region_ysort_rois[region].to_csv(save_path + region + '_ysort_rois.csv')
        region_ysort_f[region].to_csv(save_path + region + '_ysort_f.csv')
        region_ysort_normf[region].to_csv(save_path + region + '_ysort_normf.csv')
        region_ysort_zdiff[region].to_csv(save_path + region + '_ysort_zdiff.csv')
        # if fishtype == 'workingfish_tail' or fishtype =='workingfish':
        #     region_ysort_booldf_baseline_normf[region].to_csv(save_path + region + '_ysort_booldf_baseline_normf.csv')
        #     region_ysort_booldf_cluster_normf[region].to_csv(save_path + region + '_ysort_booldf_cluster_normf.csv')
        #     region_ysort_booldf_corr_zdiff[region].to_csv(save_path + region + '_ysort_booldf_corr_zdiff.csv')

            # pd.DataFrame(region_degree_zdiff[region]).to_csv(save_path + region + '_degree_zdiff.csv')
            # pd.DataFrame(region_degreeval_zdiff[region]).to_csv(save_path + region + '_degreeval_zdiff.csv')
            # pd.DataFrame(region_degree_dict_zdiff[region]).to_csv(save_path + region + '_degree_dict_zdiff.csv')
            # pd.DataFrame(region_response_dict_zdiff[region]).to_csv(save_path + region + '_response_dict_zdiff.csv')
            # pd.DataFrame(region_degree_normf_baseline[region]).to_csv(save_path + region + '_degree_normf_baseline.csv')
            # pd.DataFrame(region_degreeval_normf_baseline[region]).to_csv(save_path + region + '_degreeval_normf_baseline.csv')
            # pd.DataFrame(region_degree_dict_normf_baseline[region]).to_csv(save_path + region + '_degree_dict_normf_baseline.csv')
            # pd.DataFrame(region_response_dict_normf_baseline[region]).to_csv(save_path + region + '_response_dict_normf_baseline.csv')
            # pd.DataFrame(region_degree_normf_cluster[region]).to_csv(save_path + region + '_degree_normf_cluster.csv')
            # pd.DataFrame(region_degreeval_normf_cluster[region]).to_csv(save_path + region + '_degreeval_normf_cluster.csv')
            # pd.DataFrame(region_degree_dict_normf_cluster[region]).to_csv(save_path + region + '_degree_dict_normf_cluster.csv')
            # pd.DataFrame(region_response_dict_normf_cluster[region]).to_csv(save_path + region + '_response_dict_normf_cluster.csv')
            # for stim in ysort_stim_dict_f.keys():
            #     ysort_stim_dict_f[stim].to_csv(save_path + stim + '_ysort_stim_dict_f.csv')
            #     ysort_stim_dict_normf[stim].to_csv(save_path + stim + '_ysort_stim_dict_normf.csv')
            #     ysort_stim_dict_zdiff[stim].to_csv(save_path + stim + '_ysort_stim_dict_zdiff.csv')
            #     for region in region_ysort_rois.keys():
            #         region_ysort_stim_dict_f[stim][region].to_csv(save_path + stim + region + '_ysort_stim_dict_f.csv')
            #         region_ysort_stim_dict_normf[stim][region].to_csv(save_path + stim + region + '_ysort_stim_dict_normf.csv')
            #         region_ysort_stim_dict_zdiff[stim][region].to_csv(save_path + stim + region + '_ysort_stim_dict_zdiff.csv')