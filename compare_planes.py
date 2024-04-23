"""
Functions to plot a bunch of graphs among planes from the same fish. Run with compare_planes_runningscript.py

@Zichen He 240313
"""

from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utilities import arrutils
import plotly.graph_objects as go
import plot_individual_plane

import sys

# change this sys path to your own computer
sys.path.append(r'miniforge3/envs/naumann_lab/Codes/caImageAnalysis-kaitlyn/')

import constants, plot_individual_plane
import cmasher as cmr
from fishy import BaseFish
from datetime import datetime as dt
hzReturner = BaseFish.hzReturner

#plot fluorscence traces across all planes
def planes_plot_trace(frametimes_df, planerange, mean_trace, sdv_trace, tracename):
    """
    Plot the mean traces across all cells in each frame
    frametimes_df: the dataframe of the frames and their corresponding real time, could be from any of the plane as long as they are consistent
    planerange: a list that indicates the planes to cover
    mean_trace: a dictionary for all planes and their corresponding list of mean trace across time
    sdv_trace: a dictionary for all planes and their corresponding list of trace standard deviation across time
    tracename: the name of the trace
    """
    fig, ax = plt.subplots(1, 1, figsize = (40, 10), dpi = 240)
    frame_count = frametimes_df.shape[0]#sometimes the last plane has less time and would need to fix this line of code:(
    for plane in planerange:
        c = cmr.get_sub_cmap('bone', 0, 0.8)(plane/len(planerange))
        index = [int(x) for x in mean_trace[plane].index]
        ax.plot(np.add(index, plane * frame_count), mean_trace[plane], linewidth = 0.5, c = c)
        ax.fill_between(np.add(index, plane * frame_count),
                        list(np.subtract(mean_trace[plane], sdv_trace[plane])),
                        list(np.add(mean_trace[plane], sdv_trace[plane])), alpha=0.2, color = c)
        ax.scatter(np.mean(np.add(index, plane * frame_count)), np.mean(mean_trace[plane]), s = 200, color = c)
    ax.set_xlabel('frame * plane')
    ax.set_ylabel(tracename)

#plot stimulus trace for each region across all planes
def planes_plot_trace_stimuli(frametimes_df, offsets, stim_dict, tracename, minbar = 0, maxbar = 1):
    """
    Plot the mean cell traces heatmap in the response window for each stimuli, with their corresponding plane
        frametimes_df: the dataframe of the frames and their corresponding real time, could be from any of the plane as long as they are consistent
        offsets: the tuple that contains the frame number for the response window to look at, could be from any plane as long as they are consistent
        stim_dict: the dictionary of stimulus with their corresponiding cell responses. Each stimulus key is corresponding to a dictionary containing all regions
        tracename: the cell trace name, used to label the graph
        minbar: the minimal value of the cbar, default 0
        maxbar: the max value of the cbar, dfault 1
    Return: None
    """
    hz = hzReturner(frametimes_df)
    #initiate figure plotting space
    region_count = len(stim_dict['forward'].keys()) + 1
    stim_count = len(stim_dict.keys()) + 2
    fig, ax= plt.subplots(region_count, stim_count, figsize = (18, 32), dpi = 240,
                              gridspec_kw={'height_ratios': [1 * (region_count - 1)] + [20] * (region_count -1),
                                           'width_ratios': [1] + [5] * (stim_count -2) + [1],
                                            'hspace': 0, 'wspace': 0.05})
    #turn off unnecessary axis
    ax[0,0].axis('off')
    ax[0,stim_count - 1].axis('off')
    #prepare cbar axis
    gs = ax[1, 0].get_gridspec()
    for axes in ax[1:, 0]:
        axes.axis('off')
        axes.remove()
    ax_cbar = fig.add_subplot(gs[1:, 0])
    stim_col = 1
    for stim in constants.dir_sort: #sort stim bar according to a sequence that makes more sense
        #plot stimulus traces
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 0 + 5*hz/2, facecolor = c[0], alpha = 0.5)
        ax[0, stim_col].axvspan(0 + 5*hz/2 + 0.05, 0 + 5*hz, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].sharex(ax[2, stim_col])
        ax[0, stim_col].axis('off')
        #plot heatmap for each region
        for region_row in range(1, region_count):
            regionname = list(stim_dict[stim].keys())[region_row - 1]
            trace_to_plot = stim_dict[stim][regionname]
            ax_heatmap = ax[region_row, stim_col]
            if region_row == region_count - 1 and stim_col == 1:#if first heatmap on the last row, plot cbar and x axis
                try:
                    sns.heatmap(trace_to_plot, ax = ax_heatmap,
                        cmap = 'viridis', vmax = maxbar, vmin = minbar, yticklabels = False, cbar_ax = ax_cbar,
                        xticklabels = False, cbar_kws =dict(location="left", shrink = 0.3, label = 'mean ' + tracename))
                    ax_heatmap.set_xticks([0, offsets[1] - offsets[0]], labels = [0, offsets[1] - offsets[0]])
                    ax_heatmap.set_xlabel('stim on (frame)')
                except:
                    sns.heatmap(np.zeros((offsets[1] - offsets[0], offsets[1] - offsets[0])), ax=ax_heatmap,
                                cmap='viridis', vmax=maxbar, vmin=minbar, yticklabels=False, cbar_ax=ax_cbar,
                                xticklabels=False,
                                cbar_kws=dict(location="left", shrink=0.3, label='mean ' + tracename))
                    ax_heatmap.set_xticks([0, offsets[1] - offsets[0]], labels=[0, offsets[1] - offsets[0]])
                    ax_heatmap.set_xlabel('stim on (frame)')
            else:
                try:
                    sns.heatmap(trace_to_plot, ax = ax_heatmap,
                        cmap = 'viridis', vmax = maxbar, vmin = minbar, cbar = False, yticklabels = False,
                            xticklabels = False)
                except:
                    sns.heatmap(np.zeros((offsets[1] - offsets[0], offsets[1] - offsets[0])), ax=ax_heatmap,
                                cmap='viridis', vmax=maxbar, vmin=minbar, cbar=False, yticklabels=False,
                                xticklabels=False)
            #plot white dash lines between regions
            if region_row != 1:
                ax_heatmap.axhline(y = 0, xmax = offsets[1] - offsets[0], color='white', linewidth=1, linestyle = ':')
            #if last region row, plot region scatter
            if stim_col == stim_count - 2:
                ax_scatter = ax[region_row, stim_count - 1]
                ax_scatter.axvspan(0, 1, color = plt.colormaps['bone'](regionname/(region_count - 1)), alpha = 0.5)
                ax_scatter.set_xticks([])
                ax_scatter.set_yticks([])
                plt.setp(ax_scatter.spines.values(), visible=False)
                ax_scatter.set_ylabel('plane ' + str(regionname))
                ax_scatter.yaxis.set_label_position("right")
        stim_col += 1

#plot distribution of correlation for each region across all planes
def planes_plot_corr_dist(planerange, regionlist, meancorr_trace, region_meancorr_trace, tracename):
    """
    Plot the distribution of correlation values for each region for each plane
        planerange: the list of planes that are included
        regionlist: the list of regions that are included
        meancorr_trace: the list containing correlation values for all regions across each plane
        region_meancorr_trace: the dictionary for each region, which each contains the list containing correlation values for all regions across each plane
        tracename: the name of the trace to be plotted
    """
    fig, ax = plt.subplots(len(regionlist) + 1, 2, figsize = (10, 30), dpi = 240, gridspec_kw = {'hspace': 0.2,
                                                                                                 'width_ratios': [20, 1]})
    planerange_str = [str(plane) for plane in planerange]
    for row in range(0, len(regionlist) + 1):
        ax_plot = ax[row, 0]
        if row == 0:
            regionname = 'all'
            list_to_plot = list(meancorr_trace.values())
            c = 'grey'
        else:
            regionname = regionlist[row - 1]
            list_to_plot = list(region_meancorr_trace[regionname].values())
            c = constants.cmaplist[regionname](0.5)
        list_to_plot = [[] if value is None else value for value in list_to_plot]
        bp = ax_plot.boxplot(list_to_plot, vert = True, notch = True, labels = planerange_str, sym = '', patch_artist = True)
        alpha_plane = np.arange(0.1, 1, 0.9/len(planerange))
        for patch, alpha in zip(bp['boxes'], alpha_plane):
            patch.set_facecolor(c)
            patch.set_alpha(alpha)
        for median, whisker, cap in zip(bp['medians'], bp['whiskers'], bp['caps']):
            median.set_color(c)
            whisker.set_color('grey')
            cap.set_color('grey')
        ax_plot.set_ylabel(tracename + ' r2')
        ax_plot.set_yticklabels([])
        ax_plot.set_ylim([-1, 1])
        ax_plot.set_xlabel('')
        ax_plot.spines['right'].set_visible(False)
        ax_plot.spines['top'].set_visible(False)
        if row == 0:
            ax_plot.set_yticks([-1, 0, 1])
            ax_plot.set_yticklabels([-1, 0, 1])
        if row == len(regionlist):
            ax_plot.set_xlabel('plane')
        else:
            ax_plot.set_xticklabels([])
        #plot region colormap
        ax_scatter = ax[row, 1]
        ax_scatter.axvspan(0, 1, color= c, alpha = 0.5)
        ax_scatter.set_xticks([])
        ax_scatter.set_yticks([])
        plt.setp(ax_scatter.spines.values(), visible=False)
        ax_scatter.set_ylabel(regionname)
        ax_scatter.yaxis.set_label_position("right")

#plot number of stimuli responsive cell for each region across all planes
def planes_plot_tuning_num(frametimes_df, regionlist, planerange, num_dir, tracename):
    """
    Plot the number of cells tuned to each stimuli for each region across all planes
        frametimes_df: the dataframe of the frames and their corresponding real time, could be from any of the plane as long as they are consistent
        regionlist: the list of regions to be plotted
        planerange: a list that indicates the planes to cover
        num_dir: the dictionary containing each region, each stimuli, and corresponding number of cells tuned for each stimuli
        tracename: the type of signal that is used to sort cells
    """
    fig, ax = plt.subplots(len(regionlist) + 1, len(constants.monocular_dict.keys()) + 1, figsize = (20, 10), dpi = 240,
                           gridspec_kw = {'height_ratios': [1 * len(regionlist)] + [20] * len(regionlist), 'hspace': 0.2,
                                          'width_ratios': [12] * len(constants.monocular_dict.keys()) + [1]})
    hz = hzReturner(frametimes_df)
    planerange_str = [str(plane) for plane in planerange]
    stim_col = 0
    ax[0, len(constants.monocular_dict.keys())].axis('off')
    for stim in constants.monocular_dict.keys(): #sort stim bar according to a sequence that makes more sense
        #plot stimulus traces
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 0 + 5*hz/2, facecolor = c[0], alpha = 0.5)
        ax[0, stim_col].axvspan(0 + 5*hz/2 + 0.05, 0 + 5*hz, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].set_xlim([0, 32])
        ax[0, stim_col].axis('off')
        region_row = 1
        for region in regionlist:
            list_to_plot = list(num_dir[region][stim].values())
            c = constants.cmaplist[region](0.8)
            list_to_plot = [0 if value is None else value for value in list_to_plot]
            bar = ax[region_row, stim_col].bar(planerange_str, list_to_plot, color = c)
            alpha_plane = np.arange(0.1, 1, 1/len(planerange))
            i = 0
            for b, alpha in zip(bar, alpha_plane):
                b.set_alpha(alpha)
                i += 1
            ax[region_row, stim_col].set_ylim([0, 500])
            ax[region_row, stim_col].set_xticks([])
            ax[region_row, stim_col].spines['right'].set_visible(False)
            ax[region_row, stim_col].spines['top'].set_visible(False)
            if stim_col == 0:
                ax[region_row, stim_col].set_ylabel(tracename + ' #cells')
                ax[region_row, stim_col].set_yticks([0, 500])
                ax[region_row, stim_col].set_yticklabels([0, 500])
            else:
                ax[region_row, stim_col].set_ylabel('')
                ax[region_row, stim_col].set_yticks([])
                ax[region_row, stim_col].set_yticklabels([])
            if region_row == len(regionlist):
                ax[region_row, stim_col].set_xlabel('plane')
            else:
                ax[region_row, stim_col].set_xticklabels([])
            # plot region colormap
            if stim_col == len(constants.monocular_dict.keys()) - 1:
                ax_scatter = ax[region_row, stim_col + 1]
                ax_scatter.axvspan(0, 1, color=c, alpha=0.5)
                ax_scatter.set_xticks([])
                ax_scatter.set_yticks([])
                plt.setp(ax_scatter.spines.values(), visible=False)
                ax_scatter.set_ylabel(region)
                ax_scatter.yaxis.set_label_position("right")
            region_row += 1
        stim_col += 1

#plot percentage of stimuli responsive cell for each region across all planes
def planes_plot_tuning_perc(frametimes_df, regionlist, planerange, perc_dir, tracename):
    """
       Plot the percentage of cells within the region tuned to each stimuli for each region across all planes
           frametimes_df: the dataframe of the frames and their corresponding real time, could be from any of the plane as long as they are consistent
           regionlist: the list of regions to be plotted
           planerange: a list that indicates the planes to cover
           num_dir: the dictionary containing each region, each stimuli, and corresponding number of cells tuned for each stimuli
           tracename: the type of signal that is used to sort cells
       """
    fig, ax = plt.subplots(len(regionlist) + 1, len(constants.monocular_dict.keys()) + 1, figsize = (20, 10), dpi = 240,
                           gridspec_kw = {'height_ratios': [1 * len(regionlist)] + [20] * len(regionlist), 'hspace': 0.2,
                                          'width_ratios': [12] * len(constants.monocular_dict.keys()) + [1]})
    hz = hzReturner(frametimes_df)
    planerange_str = [str(plane) for plane in planerange]
    stim_col = 0
    ax[0, len(constants.monocular_dict.keys())].axis('off')
    for stim in constants.monocular_dict.keys(): #sort stim bar according to a sequence that makes more sense
        #plot stimulus traces
        c = constants.allcolor_dict[stim]
        ax[0, stim_col].axvspan(0, 0 + 5*hz/2, facecolor = c[0], alpha = 0.5)
        ax[0, stim_col].axvspan(0 + 5*hz/2 + 0.05, 0 + 5*hz, facecolor = c[1], alpha = 0.5)
        ax[0, stim_col].set_xlim([0, 32])
        ax[0, stim_col].axis('off')
        region_row = 1
        for region in regionlist:
            list_to_plot = list(perc_dir[region][stim].values())
            c = constants.cmaplist[region](0.8)
            list_to_plot = [0 if value is None else value for value in list_to_plot]
            list_to_plot = np.multiply(list_to_plot, 100)
            bar = ax[region_row, stim_col].bar(planerange_str, list_to_plot, color=c)
            alpha_plane = np.arange(0.1, 1, 1/len(planerange))
            i = 0
            for b, alpha in zip(bar, alpha_plane):
                b.set_alpha(alpha)
                i += 1
            ax[region_row, stim_col].set_ylim([0, 50])
            ax[region_row, stim_col].set_xticks([])
            ax[region_row, stim_col].spines['right'].set_visible(False)
            ax[region_row, stim_col].spines['top'].set_visible(False)
            if stim_col == 0:
                ax[region_row, stim_col].set_ylabel(tracename + ' %cells')
                ax[region_row, stim_col].set_yticks([0, 50])
                ax[region_row, stim_col].set_yticklabels([0, 50])
            else:
                ax[region_row, stim_col].set_ylabel('')
                ax[region_row, stim_col].set_yticks([])
                ax[region_row, stim_col].set_yticklabels([])
            if region_row == len(regionlist):
                ax[region_row, stim_col].set_xlabel('plane')
            else:
                ax[region_row, stim_col].set_xticklabels([])
            # plot region colormap
            if stim_col == len(constants.monocular_dict.keys()) - 1:
                ax_scatter = ax[region_row, stim_col + 1]
                ax_scatter.axvspan(0, 1, color=c, alpha=0.5)
                ax_scatter.set_xticks([])
                ax_scatter.set_yticks([])
                plt.setp(ax_scatter.spines.values(), visible=False)
                ax_scatter.set_ylabel(region)
                ax_scatter.yaxis.set_label_position("right")
            region_row += 1
        stim_col += 1

def planes_plot_on(frametimes_df, region_trace, tracename, loc, on_method, cutoff_s, refImg):
    """
    Note that the traces are normalized and smoothed when selecting "on periods". The smoothing factor is determined by
    frame rate * 10.
        frametimes_df: the dataframe for all frames and their corresponding raw time
        region_trace: dictionary containing dataframe containing all traces for all regions, raw F!
        tracename: the cell trace name, used to label the graph
        loc: a dataframe containing all regions and their corresponding ROIs for each cell
        on_method: the method to selecton "on periods" ('cluster', 'mean', 'diff_peak')
        cutoff_s: the cut off seconds to differentiate on/off and peaky neurons
        refImg: the dataframe to plot the original fish plane

    Return:
        color_cutoff: the max duration in second that matches the colorbar
        mean_on_duration: the CLIPPD and NORMALIZED duration in seconds matching the colorbar
    """
    hz = hzReturner(frametimes_df)
    cbar = plt.get_cmap('rainbow')

    #preparing figure space
    region_count = len(region_trace.keys())
    fig, ax= plt.subplots(region_count + 1, 4,
                          gridspec_kw={'hspace': 0, 'wspace': 0.2, 'height_ratios':  [20] * region_count + [1]},
            figsize = (15, 6), dpi = 240, sharex = 'col')
    # add gridspace for all neuron correlation plot
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_scatter = fig.add_subplot(gs[0:, 3])
    ax_scatter.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    ax_scatter.set_yticks([])
    ax_scatter.set_xticks([])
    # plot cbar
    color_cutoff = 75
    for i in range(0, 3):
        ax_cbar = ax[region_count, i]
        #if heatmap ax cbar, transfer to frames
        if i == 1:
            color_cutoff_hz = int(color_cutoff * hz)
            plotting_frame = int(np.floor(150 * hz) + 1)
            ax_cbar.scatter(np.linspace(0, color_cutoff_hz, color_cutoff_hz + 1), [0] * (color_cutoff_hz + 1),
                            c=np.linspace(0, color_cutoff_hz, color_cutoff_hz + 1), cmap=cbar)
            ax_cbar.scatter(np.linspace(color_cutoff_hz + 1, plotting_frame,
                                        plotting_frame - color_cutoff_hz),
                            [0] * (plotting_frame - color_cutoff_hz), c='red')
        #else, stay with seconds
        else:
            ax_cbar.scatter(np.linspace(0, color_cutoff, color_cutoff + 1), [0] * (color_cutoff + 1),
                        c=np.linspace(0, color_cutoff, color_cutoff + 1), cmap=cbar)
            ax_cbar.scatter(np.linspace(color_cutoff + 1, 150, 150 - color_cutoff), [0] * (150 - color_cutoff), c='red')
        ax_cbar.spines['top'].set_color('white')
        ax_cbar.spines['right'].set_color('white')
        ax_cbar.spines['bottom'].set_color('white')
        ax_cbar.spines['left'].set_color('white')
        ax_cbar.set_yticks([])
        ax_cbar.sharex(ax[0, i])
        ax_cbar.set_xlabel('time (s)')
    mean_on_trace = {key: np.empty((region_trace[key].shape[0], frametimes_df.shape[0])) for key in region_trace.keys()}
    mean_on_duration = {key: np.empty(region_trace[key].shape[0]) for key in region_trace.keys()}
    region_row = 0
    for region in region_trace.keys():
        #process data for each region
        trace = region_trace[region]
        for neuron in trace.index:
            trace.loc[neuron, :] = arrutils.pretty(trace.loc[neuron, :], int(10 * hz))
        trace = arrutils.norm_0to1(trace.to_numpy())
        if on_method == 'cluster':
            mean_on_trace[region], mean_on_duration[region] = plot_individual_plane.cluster_on(frametimes_df, trace)
        elif on_method == 'mean':
            mean_on_trace[region], mean_on_duration[region] = plot_individual_plane.mean_on(frametimes_df, trace)
        elif on_method == 'diff_peak':
            mean_on_trace[region], mean_on_duration[region] = plot_individual_plane.diff_peak_on(frametimes_df, trace)
        #transfer everything to seconds
        mean_on_duration[region] = np.divide(mean_on_duration[region], hz)
        #sort neuron by duration
        sort_duration = np.argsort(mean_on_duration[region])
        sort_mean_on_trace = mean_on_trace[region][sort_duration]
        # plot heatmap
        region_heat_ax = ax[region_row, 1]
        sns.heatmap(sort_mean_on_trace[:, :plotting_frame], ax=region_heat_ax, cmap='viridis',
                    vmin=0, vmax=1, cbar=False)
        region_heat_ax.set_yticks([])
        region_heat_ax.set_xticks([])
        # plot histogram for each region
        region_hist_ax = ax[region_row, 2]
        peaky = mean_on_duration[region][np.where(mean_on_duration[region] < cutoff_s)[0]]
        onoff = mean_on_duration[region][np.where(mean_on_duration[region] >= cutoff_s)[0]]
        bins = np.linspace(0, 150, 11)
        peaky = np.divide(np.histogram(peaky, bins)[0], len(mean_on_duration[region])) *100#transfer to percentage
        onoff = np.divide(np.histogram(onoff, bins)[0], len(mean_on_duration[region])) *100
        bins_center = np.linspace(7.5, 142.5, 10)
        region_hist_ax.bar(bins_center, peaky, color = 'deepskyblue', alpha = 0.5, edgecolor = 'deepskyblue',
                           width = 15)
        region_hist_ax.bar(bins_center, onoff, color='coral', alpha=0.5, edgecolor='coral',
                          width = 15)
        region_hist_ax.set_ylim(0, 70)
        region_hist_ax.set_yticks([0, 70])
        region_hist_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth = 1)
        region_hist_ax.spines['top'].set_visible(False)
        region_hist_ax.spines['right'].set_visible(False)
        region_hist_ax.spines['bottom'].set_visible(False)
        #plot line plot for each region, plot after the other graphs because the duration is clipped
        mean_on_duration[region] = np.clip(mean_on_duration[region], 0, color_cutoff)/color_cutoff
        region_line_ax = ax[region_row, 0]
        for neuron in range(0, mean_on_trace[region].shape[0]):
            region_line_ax.plot(np.arange(0, 150, 1/hz), mean_on_trace[region][neuron][:plotting_frame],
                                c = cbar(mean_on_duration[region][neuron]), linewidth = 0.1)
        region_line_ax.axvline(cutoff_s, linestyle = ':', color = 'black', linewidth =1)
        region_line_ax.set_xlim(0, 150)
        region_line_ax.set_ylim(0, 1)
        region_line_ax.set_xticks([])
        region_line_ax.set_yticks([0, 1])
        region_line_ax.set_ylabel(region)
        region_line_ax.spines['top'].set_visible(False)
        region_line_ax.spines['right'].set_visible(False)
        region_line_ax.spines['bottom'].set_visible(False)
        if region_row == 0:
            region_line_ax.set_yticklabels([0, 1])
            region_heat_ax.set_ylabel(tracename)
            region_hist_ax.set_yticklabels([0, 70])
            region_hist_ax.set_ylabel('cell%')
        else:
            region_line_ax.set_yticklabels([])
            region_hist_ax.set_ylabel('')
            region_hist_ax.set_yticklabels([])
        ax_scatter.scatter(loc[region]['xpos'], loc[region]['ypos'], c = mean_on_duration[region], s = 0.1, cmap = cbar)
        region_row += 1
    return color_cutoff, mean_on_duration

def volumetric_plot_on(color_cutoff, region_ROIs, mean_on_duration, loc):
    """
    Plot the 3d html of the location of peaky and on/off cells, with color corresponding to their mean on durations.
            color_cutoff: the max mean_on_duration in seconds that reaches the peak of the color
            region_ROIs: the dictionary that contains all regions, as well as a dataframe containing their all ROIs including
             "xpos", "ypos", "zpos"
            mean_on_duration: the CLIPPED and NORMALIZED mean_on_duration for each cell in seconds. Note that the list
             is CLIPPED at color_cutoff and NORMALIZED to the range of 0-1 regards to percentage color_cutoff
            loc: a dataframe containing all regions and their corresponding ROIs for each cell
    """
    scatter = []
    mesh = []
    for region in region_ROIs.keys():
        scatter = scatter + \
            [go.Scatter3d(x=loc[region].loc[:, 'xpos'], y=loc[region].loc[:, 'ypos'], z=loc[region].loc[:, 'zpos'],
                                    mode='markers', opacity = 0.5,
                          marker=dict(size=3, symbol="circle", color=mean_on_duration[region],
                                                                colorscale = "rainbow" ))]
    for region in region_ROIs.keys():
        color = 'rgb(' + str(constants.cmaplist[region](0.8)[0] * 255) + ',' \
                + str(constants.cmaplist[region](0.8)[1] * 255) + ','\
                + str(constants.cmaplist[region](0.8)[2] * 255) + ')'
        mesh = mesh + \
               [go.Mesh3d(x=region_ROIs[region]['xpos'], y=region_ROIs[region]['ypos'], z=region_ROIs[region]['zpos'],
                          opacity=0.1, alphahull=0, color = color)]
    fig = go.Figure(data=mesh + scatter)
    fig.update_layout(plot_bgcolor='white',
                       coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=1))
    return fig

