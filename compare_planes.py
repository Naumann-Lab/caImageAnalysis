"""
Functions to plot a bunch of graphs among planes from the same fish. Run with compare_planes_runningscript.py

@Zichen He 240313
"""

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import Point, Polygon
import numpy as np
import pandas as pd
import seaborn as sns
from utilities import arrutils
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import matplotlib.cm as cm
from scipy.stats import pearsonr
from ast import literal_eval

import constants, plot_individual_plane, angles
import cmasher as cmr
from fishy import BaseFish
from datetime import datetime as dt
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

def volumetric_plot(region_ROIs, clip_variable, loc, byregion =True):
    """
    Plot the 3d html of the location of peaky and on/off cells, with color corresponding to their mean on durations.
            color_cutoff: the max mean_on_duration in seconds that reaches the peak of the color
            region_ROIs: the dictionary that contains all regions, as well as a dataframe containing their all ROIs including
             "xpos", "ypos", "zpos"
            clip_variable: the CLIPPED (and NORMALIZED) variable for each cell that determine their color. Note that the
             list is CLIPPED at color_cutoff and NORMALIZED to the range of 0-1 regards to percentage color_cutoff
            loc: a dataframe containing all regions and their corresponding ROIs for each cell
            byregion: if the input of loc is by region as a dictionary
    """
    scatter = []
    mesh = []
    if byregion:
        for region in region_ROIs.keys():
            scatter = scatter + \
                [go.Scatter3d(x=loc[region].loc[:, 'xpos'], y=loc[region].loc[:, 'ypos'], z=loc[region].loc[:, 'zpos'],
                    mode='markers', opacity = 0.5, marker=dict(size=2, symbol="circle", color=clip_variable[region],
                                                                        colorscale = "rainbow" ))]
    else:
        scatter = scatter + \
                  [go.Scatter3d(x=loc.loc[:, 'xpos'], y=loc.loc[:, 'ypos'], z=loc.loc[:, 'zpos'],
                #mode='markers', opacity=0.5, marker=dict(size=2, symbol="circle", color=clip_variable, colorscale="rainbow"))]
                mode='markers', opacity=0.5, marker=dict(size=2, symbol="circle", color= clip_variable))]
    for region in region_ROIs.keys():
        color = 'rgb(' + str(constants.cmaplist[region](0.8)[0] * 255) + ',' \
                + str(constants.cmaplist[region](0.8)[1] * 255) + ','\
                + str(constants.cmaplist[region](0.8)[2] * 255) + ')'
        mesh = mesh + \
               [go.Mesh3d(x=region_ROIs[region]['xpos'], y=region_ROIs[region]['ypos'], z=region_ROIs[region]['zpos'],
                          opacity=0.1, alphahull=0, color = color)]
    fig = go.Figure(data=mesh + scatter)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout( coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))
    return fig

def plot_tail_type_cluster_traces(cluster_df, tail_df, region_trace, save_path, cluster):
    """
    Plot the calcium traces and raw tail_sum for each K-means clusters input.
     cluster_df: the dataframe of the tail cluster, with each row being a bout and most importantly, a columns with their
      on and off index
     region_trace: the (assuming HBr only, normF) trace to be plotted across all planes
     tail_df: the tail dataframe that includes the corresponding FRAME and the raw tail poisition (in rad)
     save_path: the image save path
     cluster: the cluster number
    """
    total_cycle = cluster_df.shape[0]//20 + 1
    region_trace['plane'] = [str(i).split('.')[1] for i in region_trace.index]
    frame_perplane = region_trace.shape[1]
    for cycle in range(0, total_cycle):
        bout_left = cluster_df.shape[0] - 20 * cycle
        max_bout = 20
        width_ratio = np.add(cluster_df.iloc[20 * cycle:20 * (cycle + 1)]['tail_duration_s'], 30)
        #if the bouts in this cluster is less than 20, turn off the rest of trhe axis
        if bout_left < 20:
            max_bout = bout_left
            width_ratio = list(np.add(cluster_df.iloc[20 * cycle:]['tail_duration_s'], 30))
            width_ratio = width_ratio + [30] * (20 - max_bout)
        fig, ax = plt.subplots(2, 20, figsize=(30, 5), dpi=240, gridspec_kw={'hspace': 0.1, 'width_ratios': width_ratio})
        if bout_left < 20:
            for empty_space in range(bout_left, 20):
                ax[0, empty_space].axis('off')
                ax[1, empty_space].axis('off')
        for bout in range(0, max_bout):
            ax_bout = ax[0, bout]
            bout_tuple = cluster_df.iloc[bout + 20 * cycle]['cont_tuples_imageframe']
            plot_raw_tail = tail_df[
                (tail_df.frame >= bout_tuple[0] - 5) & (tail_df.frame < bout_tuple[1] + 30)].tail_sum
            ax_bout.set_ylim([-5, 5])
            ax_bout.plot(list(plot_raw_tail), linewidth=0.5, color = 'black')
            ax_bout.set_xticks([])
            ax_bout.set_xticklabels([])
            ax_bout.set_yticks([])
            ax_bout.set_yticklabels([])
            ax_bout.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

            # plot calcium signals in HBr only
            # find plane
            tuple_plane = bout_tuple[0] // frame_perplane + 4#+9
            if tuple_plane%10 == 0:
                tuple_plane = str(tuple_plane)[0]
            elif tuple_plane > 10:
                tuple_plane = str(tuple_plane)
            elif tuple_plane < 10 :
                tuple_plane = '0' + str(tuple_plane)

            tuple_frame_on =bout_tuple[0] % frame_perplane
            tuple_frame_off = bout_tuple[1] % frame_perplane

            ax_neuron = ax[1, bout]
            plane_cell_toplot = region_trace[region_trace.plane == tuple_plane].drop(['plane'], axis = 1)
            plane_cell_toplot = plane_cell_toplot.iloc[:, tuple_frame_on - 5:tuple_frame_off + 30 ]
            #sort cells by their total fluorscence
            plane_cell_toplot['sum'] = plane_cell_toplot.sum()
            plane_cell_toplot = plane_cell_toplot.sort_values(by = ['sum']).drop(['sum'], axis = 1)
            sns.heatmap(plane_cell_toplot,
                ax=ax_neuron, cmap='viridis', vmin=0, vmax=1, cbar=False)
            ax_neuron.axvline(5, linestyle=':', color='pink', linewidth=1)
            ax_neuron.axvline(5 + tuple_frame_off - tuple_frame_on, linestyle=':', color='pink', linewidth=1)
            ax_neuron.set_yticks([])
            ax_neuron.set_yticklabels([])
            ax_neuron.set_xticks([0, 5, bout_tuple[1] - bout_tuple[0] + 30])
            ax_neuron.set_xticklabels(['-5', bout_tuple[1] - bout_tuple[0], '+30'], rotation = 0)
            if bout == 0:
                ax_neuron.set_xlabel('frame')
        fig.savefig(save_path + '/tail_cluster/cluster_' + str(cluster) + '(' + str(cycle) + ').png')

def planes_plot_tail_type_clusters(region_trace, tail_bout_df, tail_df, save_path, k_clusters = 6):
    """
    cluster the types of tail events with PCA and K-Means Clustering.
        region_trace: the traces of neurons for all planes, with their index labeled "plane.n_index" (default normF)
        tail_bout_df: A dataframe with each row being a tail event, including "cont_tuples", "tail_stimuli", the positive
         component, negative component, standard deviation, duration (s) and frequency (times/s) for each event
        tail_df: the dataframe for raw tail traces and their matching frames
        k_clusters: Ideally, use the elbow method (commented out below) to find the optimal k_means_clusters. Default around
         6 for all tail data here seems to yield a good result.
        save_path: the path to save for the plot in string
    """
    fig, ax = plt.subplots(2, 1, figsize=(5, 10), dpi = 240)
    # standardize the data
    tail_bout_df_pcainput = tail_bout_df.drop(['cont_tuples_tailindex', 'cont_tuples_imageframe', 'tail_stimuli'], axis=1)
    scaler = StandardScaler()
    tail_bout_df_pcainput = scaler.fit_transform(tail_bout_df_pcainput)
    # make the PCA
    pca = PCA(n_components=4)
    principalComponents = pca.fit_transform(tail_bout_df_pcainput)
    tail_bout_df_pcaoutput = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2', 'principal component 3',
                                        'principal component 4'])
    # plot bout unrelated scatteres
    colormap = cm.get_cmap('winter')
    cmap = np.divide(list(range(0, tail_bout_df.shape[0])), tail_bout_df.shape[0])
    ax[0].scatter(tail_bout_df_pcaoutput['principal component 1'], tail_bout_df_pcaoutput['principal component 2'],
               color=colormap(cmap))
    ax[0].set_title('PCA across bouts')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[0].set_xticks([])
    ax[0].set_xticklabels([])
    ax[0].set_yticks([])
    ax[0].set_yticklabels([])
    ax[0].spines[['right', 'top']].set_visible(False)

    #use elbow method to find the best k means clusters numbers
    # wcss = []
    # for i in range(1, 21):
    #     kmeans_pca = KMeans(n_clusters=i, init='k-means++', random_state=42)
    #     kmeans_pca.fit(tail_bout_df_pcaoutput)
    #     wcss.append(kmeans_pca.inertia_)

    #K Means Clustering
    kmeans_pca = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42)
    kmeans_pca.fit(tail_bout_df_pcaoutput)
    final_result = pd.concat([tail_bout_df, tail_bout_df_pcaoutput, pd.DataFrame(kmeans_pca.labels_)], axis=1)
    final_result.columns.values[-1] = 'K Means Cluster'
    sns.scatterplot(data=final_result, x='principal component 1', y='principal component 2', hue='K Means Cluster',
                    ax = ax[1], palette = 'cool')
    sns.move_legend(ax[1], 'lower center', bbox_to_anchor=(.5, -.2), title = None, frameon = False, ncol = k_clusters)
    ax[1].set_xticks([])
    ax[1].set_xticklabels([])
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    ax[1].spines[['right', 'top']].set_visible(False)
    ax[1].set_title('K Means Clustering')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')
    plt.savefig(save_path + '90_tail_clustering')

    #for each cluster, plot HBr calcium traces and raw tail_sum
    cluster_n = final_result['K Means Cluster'].max() + 1
    for cluster in range(0, cluster_n):
        cluster_df = final_result[final_result['K Means Cluster'] == cluster]
        plot_tail_type_cluster_traces(cluster_df, tail_df, region_trace['HBr'], save_path, cluster)

    return final_result

def volumetric_plot_tailneurons(neuron_tail_dfs, loc, refImg, plane_toppercentage):
    """
    Plot the 3d html of the location of peaky and on/off cells, with color corresponding to their mean on durations.
            color_cutoff: the max mean_on_duration in seconds that reaches the peak of the color
            region_ROIs: the dictionary that contains all regions, as well as a dataframe containing their all ROIs including
             "xpos", "ypos", "zpos"
            tail_neuron_df: the dataframe containing all neurons for each plane and their relevant tail responding/predict
            rate
            loc: the neuron index and the x, y, and zpos of the neurons
            plane_toppercentage: the percentile of top ail responder for each plane to plot
    """
    # DECIDE TOP RESPONDERS
    plane_toppercentage = 1 - plane_toppercentage
    neuron_tail_dfs[
        'tail_total_predictresponse_rate'] = neuron_tail_dfs.tail_total_predict_rate + neuron_tail_dfs.tail_total_response_rate
    # top_responder_cutoff = neuron_tail_dfs.tail_total_response_rate.quantile(plane_toppercentage)
    # top_predictor_cutoff = neuron_tail_dfs.tail_total_predict_rate.quantile(plane_toppercentage)
    top_all_cutoff = neuron_tail_dfs['tail_total_predictresponse_rate'].quantile(plane_toppercentage)
    top_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predictresponse_rate'] >= top_all_cutoff]
    # top_neurons = neuron_tail_dfs[(neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff) & (
    #                 neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff)]

    #PREPARE COLOR SCHEMES
    response_portion = list(np.divide(neuron_tail_dfs['tail_total_response_rate'],
                                    np.add(neuron_tail_dfs['tail_total_response_rate'], neuron_tail_dfs['tail_total_predict_rate'])))
    response_portion = [0 if x != x else x for x in response_portion]
    predict_portion = list(np.divide(neuron_tail_dfs['tail_total_predict_rate'],
                                    np.add(neuron_tail_dfs['tail_total_response_rate'], neuron_tail_dfs['tail_total_predict_rate'])))
    predict_portion = [0 if x != x else x for x in predict_portion]
    blues_cmap = plt.get_cmap('Reds')(neuron_tail_dfs['tail_total_response_rate'])#0.5
    reds_cmap = plt.get_cmap('Blues')(neuron_tail_dfs['tail_total_predict_rate'])# 0.5
    # scattercolor = np.add([np.multiply(blues_cmap[i, :], response_portion[i]) for i in range(len(response_portion))],
    #                          [np.multiply(reds_cmap[i, :], predict_portion[i]) for i in range(len(predict_portion))])[:, :3]
    scattercolor = plt.get_cmap('RdPu')(neuron_tail_dfs['tail_total_predictresponse_rate'])
    #scattercolor = [[1, 1, 1] if i.all() == 0 else i for i in scattercolor ]
    scattercolor_df = pd.DataFrame(index = neuron_tail_dfs.index, data = scattercolor)
    scatter = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'], z=loc.loc[neuron_tail_dfs.index, 'zpos'],
                mode='markers', opacity=0.8, marker=dict(size=3, symbol="circle", color= scattercolor_df.loc[neuron_tail_dfs.index]))

    # top_neurons = []
    # for plane in planes:
    #     neuron_tail_df = neuron_tail_dfs[np.round((neuron_tail_dfs.index%1), 2) * 100 == plane]
    #     top_responder_cutoff = neuron_tail_df.tail_total_response_rate.quantile(plane_toppercentage)
    #     top_predictor_cutoff = neuron_tail_df.tail_total_predict_rate.quantile(plane_toppercentage)
    #     top_neuron = neuron_tail_df[(neuron_tail_df['tail_total_response_rate'] >= top_responder_cutoff) & (
    #                 neuron_tail_df['tail_total_predict_rate'] >= top_predictor_cutoff)].index
    #     top_neurons = top_neurons + list(top_neuron)
    topscatter = go.Scatter3d(x=loc.loc[top_neurons.index, 'xpos'], y=loc.loc[top_neurons.index, 'ypos'], z=loc.loc[top_neurons.index, 'zpos'],
                mode='markers', opacity=0.5, marker=dict(size=5, symbol="circle", color= scattercolor_df.loc[top_neurons.index]))
    data = [scatter] + [topscatter]
    fig = go.Figure(data=data)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    #PLOT RESPONSE RATES
    #scattercolor_response = (blues_cmap[:3, :] * 2).T
    scattercolor_response = blues_cmap[:, :3]
    scattercolor_response_df = pd.DataFrame(index=neuron_tail_dfs.index, data=scattercolor_response)
    scatter_response = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                           z=loc.loc[neuron_tail_dfs.index, 'zpos'],
                           mode='markers', opacity=0.5,
                           marker=dict(size=2, symbol="circle", color=scattercolor_response_df.loc[neuron_tail_dfs.index]))
    # determine top responders in each plane
    #top_response_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff].index
    # top_response_neurons = []
    # for plane in planes:
    #     neuron_tail_df = neuron_tail_dfs[np.round((neuron_tail_dfs.index % 1), 2) * 100 == plane]
    #     top_responder_cutoff = neuron_tail_df.tail_total_response_rate.quantile(plane_toppercentage)
    #     top_response_neuron = neuron_tail_df[neuron_tail_df['tail_total_response_rate'] >= top_responder_cutoff].index
    #     top_response_neurons = top_response_neurons + list(top_response_neuron)
    #topscatter_response = go.Scatter3d(x=loc.loc[top_response_neurons, 'xpos'], y=loc.loc[top_response_neurons, 'ypos'],
    #                          z=loc.loc[top_response_neurons, 'zpos'],
    #                          mode='markers', opacity=0.5,
    #                          marker=dict(size=5, symbol="circle", color=scattercolor_response_df.loc[top_response_neurons]))
    #data_response = [scatter_response] + [topscatter_response]
    data_response = scatter_response
    fig_response = go.Figure(data=data_response)
    fig_response.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig_response.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    #PLOT PREDICT RATES
    #scattercolor_predict = (reds_cmap[:3, :] * 2).T
    scattercolor_predict = reds_cmap[:, :3]
    scattercolor_predict_df = pd.DataFrame(index=neuron_tail_dfs.index, data=scattercolor_predict)
    scatter_predict = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                                    z=loc.loc[neuron_tail_dfs.index, 'zpos'], mode='markers', opacity=0.5,
                                    marker=dict(size=2, symbol="circle", color=scattercolor_predict_df.loc[neuron_tail_dfs.index]))
    #top_predict_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff].index
    # top_predict_neurons = []
    # for plane in planes:
    #     neuron_tail_df = neuron_tail_dfs[np.round((neuron_tail_dfs.index % 1), 2) * 100 == plane]
    #     top_predictor_cutoff = neuron_tail_df.tail_total_predict_rate.quantile(plane_toppercentage)
    #     top_predict_neuron = neuron_tail_df[neuron_tail_df['tail_total_predict_rate'] >= top_predictor_cutoff].index
    #     top_predict_neurons = top_predict_neurons + list(top_predict_neuron)
    #topscatter_predict = go.Scatter3d(x=loc.loc[top_predict_neurons, 'xpos'], y=loc.loc[top_predict_neurons, 'ypos'],
    #                                   z=loc.loc[top_predict_neurons, 'zpos'],mode='markers', opacity=0.5,
    #                                   marker=dict(size=5, symbol="circle", color=scattercolor_predict_df.loc[top_predict_neurons]))
    #data_predict = [scatter_predict] + [topscatter_predict]
    data_predict = scatter_predict
    fig_predict= go.Figure(data=data_predict)
    fig_predict.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig_predict.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    #ON 2D, PLOT SCATTER PLOT FOR BOTH INDEX
    fig_scatter, ax = plt.subplots(3, 7, dpi = 240, figsize = (28, 16), gridspec_kw= {'width_ratios': [1, 6, 4, 4, 4, 4, 4], 'wspace': 0.1,
                                                                                    'height_ratios': [6, 1, 6], 'hspace': 0.1})
    ax[1, 0].axis('off')
    ax[0, 2].axis('off')
    ax[1, 2].axis('off')
    ax_combine_hist_response = ax[1, 1]
    plot = sns.histplot(x = neuron_tail_dfs['tail_total_response_rate'], binwidth=0.1, binrange=(0, 1), ax=ax_combine_hist_response,
                        stat = 'percent')
    blues = sns.color_palette("Reds", 10)
    for bin_, i in zip(plot.patches, blues):
        bin_.set_facecolor(i)
        bin_.set_edgecolor('white')
    ax_combine_hist_predict = ax[0, 0]
    plot = sns.histplot(y = neuron_tail_dfs['tail_total_predict_rate'], binwidth=0.1, binrange=(0, 1), ax=ax_combine_hist_predict,
                        stat = 'percent')
    reds = sns.color_palette("Blues", 10)
    for bin_, i in zip(plot.patches, reds):
        bin_.set_facecolor(i)
        bin_.set_edgecolor('white')
    ax_combine_scatter = ax[0, 1]
    ax_combine_scatter.scatter(neuron_tail_dfs['tail_total_response_rate'], neuron_tail_dfs['tail_total_predict_rate'], s=0.2,
                               c =scattercolor_df.loc[neuron_tail_dfs.index], marker = 'o')
    ax_combine_scatter.scatter(top_neurons['tail_total_response_rate'], top_neurons['tail_total_predict_rate'], s=2,
                               c=scattercolor_df.loc[top_neurons.index], marker = 'o')
    top_responder_cutoff = min(top_neurons['tail_total_response_rate'])
    top_predictor_cutoff = min(top_neurons['tail_total_predict_rate'])
    ax_combine_scatter.axvline(top_responder_cutoff, c='firebrick', linestyle=':', linewidth=3, alpha = 0.8)
    ax_combine_scatter.axhline(top_predictor_cutoff, c='royalblue', linestyle=':', linewidth=3, alpha = 0.8)
    ax_combine_scatter.set_xlim([0, 1])
    ax_combine_scatter.set_ylim([0, 1])
    ax_combine_scatter.set_xticks([])
    ax_combine_scatter.set_xticklabels([])
    ax_combine_scatter.set_yticks([])
    ax_combine_scatter.set_yticklabels([])
    ax_combine_scatter.spines[['top', 'right']].set_visible(False)
    ax_combine_scatter.set_aspect('equal', adjustable='box')
    ax_combine_hist_response.set_xlim([0, 1])
    ax_combine_hist_response.yaxis.tick_right()
    ax_combine_hist_response.invert_yaxis()
    ax_combine_hist_response.get_xaxis().set_ticks([])
    ax_combine_hist_response.get_yaxis().set_ticks([])
    ax_combine_hist_response.set_xlabel('neuron predicted by %tail')
    ax_combine_hist_response.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax_combine_hist_predict.invert_xaxis()
    ax_combine_hist_predict.set_ylim([0, 1])
    ax_combine_hist_predict.get_xaxis().set_ticks([])
    ax_combine_hist_predict.get_yaxis().set_ticks([])
    ax_combine_hist_predict.set_ylabel('neuron predicting %tail')
    ax_combine_hist_predict.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    #ON 2D, PLOT ALL THE OTHER POTENTIALLY FUN PARAMETERS
    def plot_ax(ax, label, variable, top_variable, cax):
        ax.scatter(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                    s= 2, alpha = 0.5, edgecolor = None,  marker = '.', color = 'darkgrey')#c=variable, cmap = 'cividis', vmin = 0, vmax = 1)
        s = ax.scatter(x=loc.loc[top_neurons.index, 'xpos'], y=loc.loc[top_neurons.index, 'ypos'],
                    s= 20, c=top_variable, cmap = 'plasma', edgecolor = None,  marker = '.', vmin = 0,
                  vmax = np.quantile(top_variable, 0.9))
        # ax.scatter(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
        #            s=20, c=variable, cmap='plasma', edgecolor=None, marker='.', vmin=0,
        #            vmax=np.quantile(variable, 0.9))
        fix_ax(ax, label)

    def plot_ax_z(ax, label, variable, top_variable, cax):
        ax.scatter(x=loc.loc[neuron_tail_dfs.index, 'zpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                    s= 2, alpha = 0.5, edgecolor = None,  marker = '.', color = 'darkgrey')#c=variable, cmap = 'cividis', vmin = 0, vmax = 1)
        s = ax.scatter(x=loc.loc[top_neurons.index, 'zpos'], y=loc.loc[top_neurons.index, 'ypos'],
                    s= 20, c=top_variable, cmap = 'plasma', edgecolor = None,  marker = '.', vmin = 0,
                  vmax = np.quantile(top_variable, 0.9))
        plt.colorbar(s, orientation = 'horizontal', ticks = [0, np.quantile(top_variable, 0.9)])
        # ax.scatter(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
        #            s=20, c=variable, cmap='plasma', edgecolor=None, marker='.', vmin=0,
        #            vmax=np.quantile(variable, 0.9))
        fix_ax(ax, None)

    def fix_ax(ax, label):
        ax.invert_yaxis()
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

    for i in range(2, 7):
        ax[1, i].axis('off')

    ax_index = ax[0, 2]
    ax_index.scatter(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                     s=0.5, c=scattercolor_df.loc[neuron_tail_dfs.index])
    ax_index.scatter(x=loc.loc[top_neurons.index, 'xpos'], y=loc.loc[top_neurons.index, 'ypos'],
                     s = 5, c=scattercolor_df.loc[top_neurons.index])
    ax_index.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
    fix_ax(ax_index, 'index')

    ax_index = ax[2, 2]
    ax_index.scatter(x=loc.loc[neuron_tail_dfs.index, 'zpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                     s=0.5, c=scattercolor_df.loc[neuron_tail_dfs.index])
    ax_index.scatter(x=loc.loc[top_neurons.index, 'zpos'], y=loc.loc[top_neurons.index, 'ypos'],
                     s=5, c=scattercolor_df.loc[top_neurons.index])
    fix_ax(ax_index, 'index')

    duration = [np.nan] * len(neuron_tail_dfs)
    min_duration = [np.nan] * len(neuron_tail_dfs)
    max_duration = [np.nan] * len(neuron_tail_dfs)
    omr_index = [np.nan] * len(neuron_tail_dfs)
    i = 0
    for n in neuron_tail_dfs.index:
        r = neuron_tail_dfs.loc[n, 'response_bout_duration_s']
        if r != []:
            duration[i] = np.mean(r)
            min_duration[i] = np.quantile(r, 0.1)
            max_duration[i] = np.quantile(r, 0.9)
            omr_index[i] = 1-np.divide(neuron_tail_dfs.loc[n, 'response_bout_stim'].count('spontaneous'),
                           len(neuron_tail_dfs.loc[n, 'response_bout_stim']))
        i += 1

    top_location = [i for i in range(len(neuron_tail_dfs)) if neuron_tail_dfs.index[i] in top_neurons.index]

    plot_ax(ax[0, 3], 'mean duration', duration, np.array(duration)[top_location], ax[1, 3])
    plot_ax(ax[0, 4], 'min duration', min_duration, np.array(min_duration)[top_location], ax[1, 4])
    plot_ax(ax[0, 5], 'max duration', max_duration, np.array(max_duration)[top_location], ax[1, 5])
    plot_ax(ax[0, 6], 'OMR index', omr_index, np.array(omr_index)[top_location], ax[1, 6])
    plot_ax_z(ax[2, 3], 'mean duration', duration, np.array(duration)[top_location], None)
    plot_ax_z(ax[2, 4], 'min duration', min_duration, np.array(min_duration)[top_location], None)
    plot_ax_z(ax[2, 5], 'max duration', max_duration, np.array(max_duration)[top_location], None)
    plot_ax_z(ax[2, 6], 'OMR index', omr_index, np.array(omr_index)[top_location], None)

    return fig, fig_response, fig_predict, fig_scatter

def volumetric_plot_pearson_tailneurons(neuron_tail_dfs, loc, plane_toppercentage):
    """
    Plot the 3d html of the location of peaky and on/off cells, with color corresponding to their mean on durations.
            color_cutoff: the max mean_on_duration in seconds that reaches the peak of the color
            region_ROIs: the dictionary that contains all regions, as well as a dataframe containing their all ROIs including
             "xpos", "ypos", "zpos"
            tail_neuron_df: the dataframe containing all neurons for each plane and their relevant tail responding/predict
            rate
            loc: the neuron index and the x, y, and zpos of the neurons
            plane_toppercentage: the percentile of top ail responder for each plane to plot
    """
    #PREPARE COLOR SCHEMES
    scatter = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'], z=loc.loc[neuron_tail_dfs.index, 'zpos'],
                mode='markers', opacity=0.5, marker=dict(size=2, symbol="circle", color= neuron_tail_dfs.loc[neuron_tail_dfs.index, 'statistics']))

    #DECIDE TOP RESPONDERS
    plane_toppercentage = 1 - plane_toppercentage
    top_all_cutoff = neuron_tail_dfs['statistics'].quantile(plane_toppercentage)
    top_neurons = neuron_tail_dfs[neuron_tail_dfs['statistics'] >= top_all_cutoff]

    topscatter = go.Scatter3d(x=loc.loc[top_neurons.index, 'xpos'], y=loc.loc[top_neurons.index, 'ypos'], z=loc.loc[top_neurons.index, 'zpos'],
                mode='markers', opacity=0.5, marker=dict(size=5, symbol="circle", color= neuron_tail_dfs.loc[top_neurons.index, 'statistics']))
    data = [scatter] + [topscatter]
    fig = go.Figure(data=data)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))
    return fig

def region_plot_tailneurons(neuron_tail_dfs, loc, region_loc, refImg, toppercentage, var, ):
    """
    Plot the distribution of tail neurons in region of interest
    """
    region_loc['wholebrain'] = loc

    #DECIDE TOP RESPONDERS
    toppercentage = 1 - toppercentage
    neuron_tail_dfs['tail_total_predictresponse_rate'] = neuron_tail_dfs.tail_total_predict_rate + neuron_tail_dfs.tail_total_response_rate
    top_all_cutoff = neuron_tail_dfs['tail_total_predictresponse_rate'].quantile(toppercentage)
    top_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predictresponse_rate'] >= top_all_cutoff]

    #DECIDE MATRICES TO PLOT
    matrices = pd.DataFrame(index = top_neurons.index, columns = [var])
    for n in top_neurons.index:
        if 'duration' in var:
            r = top_neurons.loc[n, 'predict_bout_duration_s']
            r2 = top_neurons.loc[n, 'predict_peakf']
            if r != [] and r2 != np.nan:
                if var == 'duration':
                    matrices.loc[n, var] = np.average(r, weights = r2)#np.mean(r)
                elif var == 'min_duration':
                    matrices.loc[n, var] = np.quantile(r, 0.1)
                elif var == 'max_duration':
                    matrices.loc[n, var] = np.quantile(r, 0.9)
        elif var == 'omr_index':
            matrices.loc[n, var] = top_neurons.loc[n, 'predict_bout_omrindex']
        elif var == 'direction':
            r = top_neurons.loc[n, 'predict_bout_avg']
            r2 = top_neurons.loc[n, 'predict_peakf']
            if r != []:
                matrices.loc[n, var] = angles.weighted_mean_angle(r, r2)#np.nanmean(r)
        elif var == 'stim_direction':
            r = top_neurons.loc[n, 'predict_bout_stim']
            r = [constants.deg_dict[i] for i in r if i in constants.monocular_dict.keys()]
            r2 = top_neurons.loc[n, 'predict_peakf']
            if r != []:
                matrices.loc[n, var] = angles.weighted_mean_angle(r, r2)
        elif var == 'cluster':
            r = top_neurons.loc[n, 'response_bout_cluster']
            r = [i for i in r if i in [1, 2, 4, 5]]
            if r != []:
                matrices.loc[n, var] = Counter(r).most_common(1)[0][0]
            else:
                matrices.loc[n, var] = np.nan
    matrices_min = np.nanquantile(matrices[var], 0.2)
    matrices_max = np.nanquantile(matrices[var], 0.8)
    colormap = 'plasma'
    if var == 'direction':
        matrices_min = -max(np.abs(matrices_min), np.abs(matrices_max))
        matrices_max = -matrices_min
        colors = [(0, 0.6, 1), (0, 0.6, 1), (1, 1, 1), (0, 1, 0), (1, 1, 1),(1, 0, 0.2), (1, 0, 0.2)]  # first color is black, last is red
        colormap = LinearSegmentedColormap.from_list("LFR", colors, N=1000)
    elif var == 'stim_direction':
        matrices_min = -180
        matrices_max = 180
        colors = [(1, 0, 1), (0, 0, 1), (0, 0.9, 0), (1, 0, 0), (1, 0, 1)]  # first color is black, last is red
        colormap = LinearSegmentedColormap.from_list("all", colors, N=1000)
    elif var == 'cluster':
        matrices_min = 0
        matrices_max = 72
        colormap = 'rainbow'

    # DECIDE MATRICES TO PLOT: NORMALIZED WITHIN PLANES
    matrices_norm = pd.DataFrame(index=top_neurons.index, columns=[var])
    for n in top_neurons.index:
        if 'duration' in var:
            r = top_neurons.loc[n, 'predict_bout_duration_norm']
            if r != []:
                if var == 'duration':
                    matrices_norm.loc[n, var] = np.mean(r)
                elif var == 'min_duration':
                    matrices_norm.loc[n, var] = np.quantile(r, 0.1)
                elif var == 'max_duration':
                    matrices_norm.loc[n, var] = np.quantile(r, 0.9)
        if var == 'omr_index':
            matrices_norm.loc[n, var] = top_neurons.loc[n, 'predict_bout_omrindex_norm']
        elif var == 'direction':
            r = top_neurons.loc[n, 'predict_bout_avg']
            if r != []:
                matrices_norm.loc[n, var] = np.mean(r)
        elif var == 'cluster':
            r = top_neurons.loc[n, 'predict_bout_cluster']
            if r != []:
                matrices_norm.loc[n, var] = Counter(r).most_common(1)[0][0]
    matrices_norm_min = np.nanquantile(matrices_norm[var], 0.2)
    matrices_norm_max = np.nanquantile(matrices_norm[var], 0.8)
    matrices_norm_max = max(abs(matrices_norm_min), abs(matrices_norm_max))
    matrices_norm_min = -matrices_norm_max
    region_top_neuron_count = {r: 0 for r in region_loc.keys()}
    if var == 'cluster':
        matrices_norm_min = 0
        matrices_norm_max = 5
        colormap = 'rainbow'

    #PREPARE FIGURES
    region_count = len(region_loc.keys())
    fig, ax = plt.subplots(region_count * 2, 7, dpi = 680, figsize = (20, 12), gridspec_kw = {'width_ratios': [0.1, 1, 1, 0.1, 1, 1, 1]})


    #PLOT NOT NORMALIZED WITHIN PLANE MATRICES
    region_n = 0
    #prepare cbar ax
    gs = ax[0, 0].get_gridspec()
    for axes in ax[0:, 0]:
        axes.remove()
    ax_cbar = fig.add_subplot(gs[0:, 0])
    cb = mpl.colorbar.ColorbarBase(ax_cbar, orientation='vertical', cmap=colormap, ticks = [0, 1])
    cb.set_ticklabels([round(matrices_min, 2), round(matrices_max, 2)])
    for region, region_rois in region_loc.items():
        if region_rois is not None:
            region_top_neurons_index = [n for n in top_neurons.index if n in region_rois.index]
            #plot x y scatterplot
            gs = ax[region_n * 2, 1].get_gridspec()
            for axes in ax[region_n * 2:region_n * 2 + 2, 1]:
                axes.remove()
            ax_scatterxy = fig.add_subplot(gs[region_n * 2:region_n * 2 + 2, 1])
            ax_scatterxy.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
            ax_scatterxy.axis('off')
            #ax_scatterxy.scatter(region_rois['xpos'], region_rois['ypos'], s = 0.2, alpha = 0.8, color = 'gray')
            ax_scatterxy.scatter(region_rois.loc[region_top_neurons_index, 'xpos'],
                                 region_rois.loc[region_top_neurons_index, 'ypos'],
                                 c = matrices.loc[region_top_neurons_index, var],
                                 s =1, marker = '.', alpha = 0.8, cmap = colormap, vmin = matrices_min, vmax = matrices_max)
            #plot z y scatterplot
            ax_scatterzy = ax[region_n * 2 + 1, 2]
            ax_scatterzy.scatter(region_rois['ypos'], region_rois['zpos'], s=0.2, alpha = 0.8, color='lightgray')
            ax_scatterzy.scatter(region_rois.loc[region_top_neurons_index, 'ypos'],
                                 region_rois.loc[region_top_neurons_index, 'zpos'],
                                 c =matrices.loc[region_top_neurons_index, var],
                                 s =1, marker = '.', alpha = 0.8, cmap= colormap, vmin=matrices_min, vmax=matrices_max)
            ax_scatterzy.spines[['top', 'bottom', 'right']].set_visible(False)
            ax_scatterzy.set_xticks([])
            ax_scatterzy.set_xticklabels([])
            ax_scatterzy.set_ylabel('plane')
            #plot percentage among own region
            ax_pie = ax[region_n * 2, 2]
            region_top_neuron_count[region] = len(region_top_neurons_index)
            ax_pie.pie([region_top_neuron_count[region], len(region_rois) - region_top_neuron_count[region]],
                       colors = [constants.cmaplist[region](0.8), 'lightgray'],
                       labels = [region, ' '])
        region_n += 1

    # PLOT NORMALIZED WITHIN PLANE MATRICES
    # prepare cbar ax
    region_n = 0
    gs = ax[0, 3].get_gridspec()
    for axes in ax[0:, 3]:
        axes.remove()
    ax_cbar = fig.add_subplot(gs[0:, 3])
    cb = mpl.colorbar.ColorbarBase(ax_cbar, orientation='vertical', cmap='plasma', ticks=[0, 1])
    cb.set_ticklabels([round(matrices_norm_min, 2), round(matrices_norm_max, 2)])
    for region, region_rois in region_loc.items():
        if region_rois is not None:
            region_top_neurons_index = [n for n in top_neurons.index if n in region_rois.index]
            # plot x y scatterplot
            gs = ax[region_n * 2, 4].get_gridspec()
            for axes in ax[region_n * 2:region_n * 2 + 2, 4]:
                axes.remove()
            ax_scatterxy = fig.add_subplot(gs[region_n * 2:region_n * 2 + 2, 4])
            ax_scatterxy.imshow(refImg, cmap='grey', alpha=0.8, vmax=100)
            ax_scatterxy.axis('off')
            #ax_scatterxy.scatter(region_rois['xpos'], region_rois['ypos'], s=0.2, alpha=0.8, color='gray')
            ax_scatterxy.scatter(region_rois.loc[region_top_neurons_index, 'xpos'],
                                 region_rois.loc[region_top_neurons_index, 'ypos'],
                                 c=matrices_norm.loc[region_top_neurons_index, var],
                                 s=0.5, marker='.', alpha=0.8, cmap=colormap, vmin=matrices_norm_min, vmax=matrices_norm_max)
            # plot z y scatterplot
            ax_scatterzy = ax[region_n * 2 + 1, 5]
            ax_scatterzy.scatter(region_rois['ypos'], region_rois['zpos'], s=0.2, alpha=0.8, color='lightgray')
            ax_scatterzy.scatter(region_rois.loc[region_top_neurons_index, 'ypos'],
                                 region_rois.loc[region_top_neurons_index, 'zpos'],
                                 c=matrices_norm.loc[region_top_neurons_index, var],
                                 s=0.5, marker='.', alpha=0.8, cmap=colormap, vmin=matrices_norm_min, vmax=matrices_norm_max)
            ax_scatterzy.spines[['top', 'bottom', 'right']].set_visible(False)
            ax_scatterzy.set_xticks([])
            ax_scatterzy.set_xticklabels([])
            ax_scatterzy.set_ylabel('plane')
            # plot percentage among own region
            ax_pie = ax[region_n * 2, 5]
            region_top_neuron_count[region] = len(region_top_neurons_index)
            ax_pie.pie([region_top_neuron_count[region], len(region_rois) - region_top_neuron_count[region]],
                       colors=[constants.cmaplist[region](0.8), 'lightgray'],
                        labels=[region, ' '])
        region_n += 1


    # plot percentage among all regions
    gs = ax[0, 6].get_gridspec()
    for axes in ax[0:, 6]:
        axes.remove()
    ax_allpie = fig.add_subplot(gs[0:, 6])
    ax_allpie.pie(region_top_neuron_count.values(),
                  colors = [constants.cmaplist[r](0.8) for r in region_top_neuron_count.keys()],
                  labels = region_top_neuron_count.keys(), labeldistance = None)
    ax_allpie.set_title(var)
    ax_allpie.legend(loc = 'lower center', ncols = len(region_top_neuron_count.keys()), frameon = False)


def plot_alltrace_tail_neuron_respondingonly(frametimes_dfs, tail_dfs, tail_bout_dfs, trace, neuron_tail_dfs,
                                             top_percentage = 0.1, normalized_window = (-5, 20), normalized = False):
    """
    Plot all the neuron dynamics for the top percentage of tail correlated neurons during tail events. But only neurons that peak near the tail event (responding to tail) are plotted.
        frametimes_dfs: the dataframe for all frames and their corresponding raw time, used to calculate imaging speed
        tail_dfs:
        tail_bout_dfs: a dataframe containing each bout as a row, and all the relevant information as columns. Note that this dataframe also includes information about the plane that the bout is captured in columns "plane", "frame_plane", "tailindex_plane"
        trace: dataframe containing all traces for all regions, norm f
        neuron_tail_dfs: a dataframe contain each neuron with their neuron index as index column, and one column "total response rate", and how many percentage of those bouts this neuron firing during [PREDICT NEURON ACTIVITY FROM TAIL], and one column named "total success rate", which contains how good each neuron peak calcium events predict wheather the tail is moving or not [PREDICT TAIL FROM NEURON ACTIVITY] Also Note that because the neurons seem to be tonically firing in smaller peaks, only larger peaks (peaks > 0.2 in normalized traces) are participating in this analysis
        top_percentage: the top percentage of tail predicting/responding cells to selecte
        normalized_window: the image frame window to be plotted, the first number indicates the frames before the tail onset that the trace is normalized to (trace - avg(x frame before tail trace))
        normalized: weather to normalize the trace to the avg trace before the tail onset
    Return:
        accumulator: a massive dataframe contains the neuron index, tail bout index (within each plane), plane, and bout.plane as multiindexes, and also the NORMALIZED neuron trace within the normalized window
    """
    #get top neurons
    plane_toppercentage = 1 - top_percentage
    top_responder_cutoff = neuron_tail_dfs.tail_total_response_rate.quantile(plane_toppercentage)
    top_predictor_cutoff = neuron_tail_dfs.tail_total_predict_rate.quantile(plane_toppercentage)
    top_neurons = neuron_tail_dfs[(neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff) & (
                         neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff)]
    top_neurons = top_neurons.copy()
    top_neurons['plane'] = [np.rint(n_index%1 * 100) for n_index in top_neurons.index]

    def fix_tailax(axes, title):
        """
        make tail axis prettier
            title: the title (time range of selected tail events)
        """
        axes.set_title(title)
        axes.set_ylim(-4, 4)
        axes.set_yticks([-4, 0, 4])
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    def fix_nax(axes):
        """
        make neuron trace window prettier
        """
        axes.set_ylim(0, 1)
        axes.set_yticks([0, 1])
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    hz = hzReturner(frametimes_dfs)
    tail_hz = hzReturner(tail_dfs)
    normalized_tailwindow = (int(normalized_window[0] * tail_hz/hz), int(normalized_window[1] * tail_hz/hz))
    fig, ax = plt.subplots(5, 4, figsize = (15, 10), dpi = 400)

    accumulator = top_neurons.explode(['response_bout_index']).loc[:, ['response_bout_index', 'plane']]
    accumulator.loc[:, 'n_index'] = list(accumulator.index)
    accumulator.loc[:, 'plane_bout'] = accumulator['response_bout_index'] + accumulator['plane'] * 0.01
    accumulator = accumulator.set_index(['n_index', 'response_bout_index', 'plane', 'plane_bout'])
    accumulator = pd.DataFrame(data = np.full((accumulator.shape[0],
            int(-normalized_window[0] + normalized_window[1] + np.ceil(500 * hz))), np.nan), index = accumulator.index)
    accumulator = accumulator.sort_index()

    for bout in range(len(tail_bout_dfs)):
        bout_frame_tuple = tail_bout_dfs['cont_tuples_imageframe'].iloc[bout]
        bout_tailframe_tuple = tail_bout_dfs['cont_tuples_tailindex'].iloc[bout]
        bout_duration = tail_bout_dfs['tail_duration_s'].iloc[bout]
        bout_plane = tail_bout_dfs['plane'].iloc[bout]
        if bout_duration <= 0.5:
            ax_tail = ax[0, 0]
            col = 0
        elif bout_duration > 0.5 and bout_duration <= 1.5:
            ax_tail = ax[0, 1]
            col = 1
        elif bout_duration > 1.5 and bout_duration <= 3:
            ax_tail = ax[0, 2]
            col = 2
        elif bout_duration > 3:
            ax_tail = ax[0, 3]
            col = 3
        ax_tail.plot(list(tail_dfs[tail_dfs['plane'] == bout_plane]['tail_sum'].iloc[bout_tailframe_tuple[0] +
            normalized_tailwindow[0]:bout_tailframe_tuple[1]+ normalized_tailwindow[1]]), linewidth = 0.1, c = 'grey', alpha = 0.5)
        for x in range(0, 5):
            ax[x, col].axvspan(-normalized_window[0], -normalized_window[0] + bout_frame_tuple[1] - bout_frame_tuple[0],
                            color = 'lavenderblush', alpha = 0.02, lw = 0)

    #create a multi-index dataframe to contain all the neuron trace, so one can choose from it to plot later
    plane_bout_1 = []
    plane_bout_2 = []
    plane_bout_3 = []
    plane_bout_4 = []
    for n_index in top_neurons.index:
        plane = np.rint(n_index%1 * 100)
        tail_bout_dfs_plane = tail_bout_dfs[tail_bout_dfs['plane'] == plane]
        n_f = list(trace.loc[n_index, :])
        bout_1 = []
        bout_2 = []
        bout_3 = []
        bout_4 = []
        for bout in top_neurons.loc[n_index, 'response_bout_index']:
            bout_plane = bout + plane * 0.01
            bout_frame_tuple = tail_bout_dfs_plane['cont_tuples_imageframe'].iloc[bout]
            plane_bout_frame_tuple = (bout_frame_tuple[0], bout_frame_tuple[1])
            n_f_bout = n_f[plane_bout_frame_tuple[0] + normalized_window[0]:plane_bout_frame_tuple[1] + normalized_window[1]]
            if normalized:#normalized to before baseline
                n_f_bout = list(np.subtract(n_f_bout, np.mean(n_f_bout[:-normalized_window[0]])))
            accumulator.loc[(n_index, bout, plane, bout_plane), :] = n_f_bout + (accumulator.shape[1] - len(n_f_bout)) * [np.nan]
            bout_duration = tail_bout_dfs_plane['tail_duration_s'].iloc[bout]
            #gather the bout index for each duration for this specific neuron
            if bout_duration <= 0.5:
                ax_n = ax[1, 0]
                bout_1 = bout_1 + [bout]
                plane_bout_1 = plane_bout_1 + [bout_plane]
            elif bout_duration > 0.5 and bout_duration <= 1.5:
                ax_n = ax[1, 1]
                bout_2 = bout_2 + [bout]
                plane_bout_2 = plane_bout_2 + [bout_plane]
            elif bout_duration > 1.5 and bout_duration <= 3:
                ax_n = ax[1, 2]
                bout_3 = bout_3 + [bout]
                plane_bout_3 = plane_bout_3 + [bout_plane]
            elif bout_duration > 3:
                ax_n = ax[1, 3]
                bout_4 = bout_4 + [bout]
                plane_bout_4 = plane_bout_4 + [bout_plane]
            ax_n.plot(n_f_bout, linewidth = 0.02, c = 'grey', alpha = 0.5)
        #4th row: plot avg response of this neuron to each kind of bouts
        ax[3, 0].plot(accumulator.loc[(n_index, bout_1, plane, slice(None)), :].mean(), linewidth = 0.01, c = 'grey', alpha = 1)
        ax[3, 1].plot(accumulator.loc[(n_index, bout_2, plane, slice(None)), :].mean(), linewidth = 0.01, c = 'grey', alpha = 1)
        ax[3, 2].plot(accumulator.loc[(n_index, bout_3, plane, slice(None)), :].mean(), linewidth = 0.01, c = 'grey', alpha = 1)
        ax[3, 3].plot(accumulator.loc[(n_index, bout_4, plane, slice(None)), :].mean(), linewidth = 0.01, c = 'grey', alpha = 1)

    #3rd row: for each bout, plot average responses
    for bout in tail_bout_dfs.index:
        bout_plane_index = bout//1
        bout_plane = tail_bout_dfs['plane'].loc[bout]
        bout_duration = tail_bout_dfs['tail_duration_s'].loc[bout]
        if bout_duration <= 0.5:
            ax_tail_sum = ax[2, 0]
            imageframe = 0.5 * hz
        elif bout_duration > 0.5 and bout_duration <= 1.5:
            ax_tail_sum = ax[2, 1]
            imageframe = 1.5 * hz
        elif bout_duration > 1.5 and bout_duration <= 3:
            ax_tail_sum = ax[2, 2]
            imageframe = 3 * hz
        elif bout_duration > 3:
            ax_tail_sum = ax[2, 3]
            imageframe = 10 * hz
        try:
            bout_mean_f = accumulator.loc[(slice(None), bout_plane_index, bout_plane, slice(None)), :].mean()
            bout_mean_f = bout_mean_f[:-normalized_window[0] + normalized_window[1] + int(imageframe)]
        except KeyError:# if no neurons respond during this bout
            bout_mean_f = np.nan
        ax_tail_sum.plot(bout_mean_f, linewidth = 0.1, c = 'grey', alpha = 0.5)

    #5th row: plot average across all neurons and all trials
    ax[4, 0].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_1), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 1].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_2), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 2].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_3), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 3].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_4), :].mean(), linewidth = 1, c = 'grey', alpha = 1)

    fix_tailax(ax[0, 0], '0-0.5s')
    fix_tailax(ax[0, 1], '0.5-1.5s')
    fix_tailax(ax[0, 2], '1.5-3s')
    fix_tailax(ax[0, 3], '3s+')
    for x in range(0, 4):
        fix_nax(ax[1, x])
        fix_nax(ax[2, x])
        fix_nax(ax[3, x])
        fix_nax(ax[4, x])
    ax[0, 0].set_ylabel('tail_sum (rad)')
    ax[1, 0].set_ylabel('norm F')
    ax[2, 0].set_ylabel('norm F (per tail)')
    ax[3, 0].set_ylabel('norm F (per neuron)')
    ax[4, 0].set_ylabel('norm F (avg all)')
    ax[4, 0].set_xlabel('frame')

    return accumulator

def plot_alltrace_tail_neuron_all(frametimes_dfs, tail_dfs, tail_bout_dfs, trace, neuron_tail_dfs, top_percentage = 0.1,
                                  normalized_window = (-5, 20), normalized = False):
    """
    Plot all the neuron dynamics for the top percentage of tail correlated neurons during tail events. All tail responding neurons during all tail events are plotted
        frametimes_dfs: the dataframe for all frames and their corresponding raw time, used to calculate imaging speed
        tail_dfs:
        tail_bout_dfs: a dataframe containing each bout as a row, and all the relevant information as columns. Note that this dataframe also includes information about the plane that the bout is captured in columns "plane", "frame_plane", "tailindex_plane"
        trace: dataframe containing all traces for all regions, norm f
        neuron_tail_dfs: a dataframe contain each neuron with their neuron index as index column, and one column "total response rate", and how many percentage of those bouts this neuron firing during [PREDICT NEURON ACTIVITY FROM TAIL], and one column named "total success rate", which contains how good each neuron peak calcium events predict wheather the tail is moving or not [PREDICT TAIL FROM NEURON ACTIVITY] Also Note that because the neurons seem to be tonically firing in smaller peaks, only larger peaks (peaks > 0.2 in normalized traces) are participating in this analysis
        top_percentage: the top percentage of tail predicting/responding cells to selecte
        normalized_window: the image frame window to be plotted, the first number indicates the frames before the tail onset that the trace is normalized to (trace - avg(x frame before tail trace))
        normalized: weather to normalize the trace to the avg trace before the tail onset
    Return:
        accumulator: a massive dataframe contains the neuron index, tail bout index (within each plane), plane, and bout.plane as multiindexes, and also the NORMALIZED neuron trace within the normalized window
    """

    #get top neurons
    plane_toppercentage = 1 - top_percentage
    top_responder_cutoff = neuron_tail_dfs.tail_total_response_rate.quantile(plane_toppercentage)
    top_predictor_cutoff = neuron_tail_dfs.tail_total_predict_rate.quantile(plane_toppercentage)
    top_neurons = neuron_tail_dfs[(neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff) & (
                         neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff)]
    top_neurons.loc[:, 'plane'] = [np.rint(n_index%1 * 100) for n_index in top_neurons.index]

    def fix_tailax(axes, title):
        """
        make tail axis prettier
            title: the title (time range of selected tail events)
        """
        axes.set_title(title)
        axes.set_ylim(-4, 4)
        axes.set_yticks([-4, 0, 4])
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    def fix_nax(axes):
        """
        make neuron trace window prettier
        """
        axes.set_ylim(0, 1)
        axes.set_yticks([0, 1])
        axes.set_xticks([])
        axes.set_xticklabels([])
        axes.spines[['top', 'bottom', 'left', 'right']].set_visible(False)

    hz = hzReturner(frametimes_dfs)
    tail_hz = hzReturner(tail_dfs)
    normalized_tailwindow = (int(normalized_window[0] * tail_hz/hz), int(normalized_window[1] * tail_hz/hz))
    fig, ax = plt.subplots(5, 4, figsize = (15, 10), dpi = 400)

    #build accumulator
    planes_bouts, plane_bout_counts = np.unique(tail_bout_dfs['plane'], return_counts=True)
    bout_count_dict = dict(zip(planes_bouts, plane_bout_counts))
    planes_ns, plane_n_counts = np.unique(top_neurons['plane'], return_counts = True)
    n_count_dict = dict(zip(planes_ns, plane_n_counts))
    accumulator = []
    for plane in planes_ns:
        accumulator = accumulator + n_count_dict[plane] * [list(range(bout_count_dict[plane]))]
    top_neurons = top_neurons.copy()
    top_neurons["all_bouts"] = accumulator
    accumulator = top_neurons.explode(['all_bouts']).loc[:, ['all_bouts', 'plane']]
    accumulator.loc[:, 'n_index'] = list(accumulator.index)
    accumulator.loc[:, 'plane_bout'] = accumulator['all_bouts'] + accumulator['plane'] * 0.01
    accumulator = accumulator.set_index(['n_index', 'all_bouts', 'plane', 'plane_bout'])
    accumulator = pd.DataFrame(data = np.full((accumulator.shape[0], int(-normalized_window[0] + normalized_window[1]
                                                                + np.ceil(500 * hz))), np.nan), index = accumulator.index)
    accumulator = accumulator.sort_index()

    for bout in range(len(tail_bout_dfs)):
        bout_frame_tuple = tail_bout_dfs['cont_tuples_imageframe'].iloc[bout]
        bout_tailframe_tuple = tail_bout_dfs['cont_tuples_tailindex'].iloc[bout]
        bout_duration = tail_bout_dfs['tail_duration_s'].iloc[bout]
        bout_plane = tail_bout_dfs['plane'].iloc[bout]
        if bout_duration <= 0.5:
            ax_tail = ax[0, 0]
            col = 0
        elif bout_duration > 0.5 and bout_duration <= 1.5:
            ax_tail = ax[0, 1]
            col = 1
        elif bout_duration > 1.5 and bout_duration <= 3:
            ax_tail = ax[0, 2]
            col = 2
        elif bout_duration > 3:
            ax_tail = ax[0, 3]
            col = 3
        ax_tail.plot(list(tail_dfs[tail_dfs['plane'] == bout_plane]['tail_sum'].iloc[bout_tailframe_tuple[0]
            + normalized_tailwindow[0]:bout_tailframe_tuple[1]+ normalized_tailwindow[1]]),
            linewidth = 0.1, c = 'grey', alpha = 0.5)
        for x in range(0, 5):
            ax[x, col].axvspan(-normalized_window[0], -normalized_window[0] + bout_frame_tuple[1] - bout_frame_tuple[0],
                            color = 'lavenderblush', alpha = 0.02, lw = 0)

    #create a multi-index dataframe to contain all the neuron trace, so one can choose from it to plot later
    plane_bout_1 = []
    plane_bout_2 = []
    plane_bout_3 = []
    plane_bout_4 = []
    for n_index in top_neurons.index:
        plane = np.rint(n_index%1 * 100)
        tail_bout_dfs_plane = tail_bout_dfs[tail_bout_dfs['plane'] == plane]
        n_f = list(trace.loc[n_index, :])
        bout_1 = []
        bout_2 = []
        bout_3 = []
        bout_4 = []
        for bout in top_neurons.loc[n_index, 'all_bouts']:
            bout_plane = bout + plane * 0.01
            bout_frame_tuple = tail_bout_dfs_plane['cont_tuples_imageframe'].iloc[bout]
            plane_bout_frame_tuple = (bout_frame_tuple[0], bout_frame_tuple[1])
            n_f_bout = n_f[plane_bout_frame_tuple[0] + normalized_window[0]:plane_bout_frame_tuple[1] + normalized_window[1]]
            if normalized:#normalized to before baseline
                n_f_bout = list(np.subtract(n_f_bout, np.mean(n_f_bout[:-normalized_window[0]])))
            accumulator.loc[(n_index, bout, plane, bout_plane), :] = n_f_bout + (accumulator.shape[1] - len(n_f_bout)) * [np.nan]
            bout_duration = tail_bout_dfs_plane['tail_duration_s'].iloc[bout]
            #gather the bout index for each duration for this specific neuron
            if bout_duration <= 0.5:
                ax_n = ax[1, 0]
                bout_1 = bout_1 + [bout]
                plane_bout_1 = plane_bout_1 + [bout_plane]
            elif bout_duration > 0.5 and bout_duration <= 1.5:
                ax_n = ax[1, 1]
                bout_2 = bout_2 + [bout]
                plane_bout_2 = plane_bout_2 + [bout_plane]
            elif bout_duration > 1.5 and bout_duration <= 3:
                ax_n = ax[1, 2]
                bout_3 = bout_3 + [bout]
                plane_bout_3 = plane_bout_3 + [bout_plane]
            elif bout_duration > 3:
                ax_n = ax[1, 3]
                bout_4 = bout_4 + [bout]
                plane_bout_4 = plane_bout_4 + [bout_plane]
            ax_n.plot(n_f_bout, linewidth = 0.02, c = 'grey', alpha = 0.5)
        #4th row: plot avg response of this neuron to each kind of bouts
        ax[3, 0].plot(accumulator.loc[(n_index, bout_1, plane, slice(None)), :].mean(), linewidth = 0.02, c = 'grey', alpha = 1)
        ax[3, 1].plot(accumulator.loc[(n_index, bout_2, plane, slice(None)), :].mean(), linewidth = 0.02, c = 'grey', alpha = 1)
        ax[3, 2].plot(accumulator.loc[(n_index, bout_3, plane, slice(None)), :].mean(), linewidth = 0.02, c = 'grey', alpha = 1)
        ax[3, 3].plot(accumulator.loc[(n_index, bout_4, plane, slice(None)), :].mean(), linewidth = 0.02, c = 'grey', alpha = 1)

    #3rd row: for each bout, plot average responses
    for bout in tail_bout_dfs.index:
        bout_plane_index = bout//1
        bout_plane = tail_bout_dfs['plane'].loc[bout]
        bout_duration = tail_bout_dfs['tail_duration_s'].loc[bout]
        if bout_duration <= 0.5:
            ax_tail_sum = ax[2, 0]
            imageframe = 0.5 * hz
        elif bout_duration > 0.5 and bout_duration <= 1.5:
            ax_tail_sum = ax[2, 1]
            imageframe = 1.5 * hz
        elif bout_duration > 1.5 and bout_duration <= 3:
            ax_tail_sum = ax[2, 2]
            imageframe = 3 * hz
        elif bout_duration > 3:
            ax_tail_sum = ax[2, 3]
            imageframe = 10 * hz
        try:
            bout_mean_f = accumulator.loc[(slice(None), bout_plane_index, bout_plane, slice(None)), :].mean()
            bout_mean_f = bout_mean_f[:-normalized_window[0] + normalized_window[1] + int(imageframe)]
        except KeyError:# if no neurons respond during this bout
            bout_mean_f = np.nan
        ax_tail_sum.plot(bout_mean_f, linewidth = 0.1, c = 'grey', alpha = 0.5)

    #5th row: plot average across all neurons and all trials
    ax[4, 0].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_1), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 1].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_2), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 2].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_3), :].mean(), linewidth = 1, c = 'grey', alpha = 1)
    ax[4, 3].plot(accumulator.loc[(slice(None), slice(None), slice(None), plane_bout_4), :].mean(), linewidth = 1, c = 'grey', alpha = 1)


    fix_tailax(ax[0, 0], '0-0.5s')
    fix_tailax(ax[0, 1], '0.5-1.5s')
    fix_tailax(ax[0, 2], '1.5-3s')
    fix_tailax(ax[0, 3], '3s+')
    for x in range(0, 4):
        fix_nax(ax[1, x])
        fix_nax(ax[2, x])
        fix_nax(ax[3, x])
        fix_nax(ax[4, x])
    ax[0, 0].set_ylabel('tail_sum (rad)')
    ax[1, 0].set_ylabel('norm F')
    ax[2, 0].set_ylabel('norm F (per tail)')
    ax[3, 0].set_ylabel('norm F (per neuron)')
    ax[4, 0].set_ylabel('norm F (avg all)')
    ax[4, 0].set_xlabel('frame')

    return accumulator

def artr_lateralization(vspn, midline, neuron_tail_dfs, loc, fish, planerange, toppercentage):
    """
    Calculate the Pearson's correlation coefficient with left/right swim for the left and right anatomy region
        artr: the list indicates the region of interest in the format of [ymin, ymax, xmin, xmid, xmax]
        neuron_tail_dfs: the neuron index across the whole brain as well as the tail decoding accuracy and related parameters
        loc: the x and y location for all neurons
        fish: the string of fish name for reference of file location
        planerange: the range of plane used for analysis
        toppercentage: top percentage of tail neurons to look atb
    """
    # DECIDE TOP RESPONDERS
    toppercentage = 1 - toppercentage
    neuron_tail_dfs[
        'tail_total_predictresponse_rate'] = neuron_tail_dfs.tail_total_predict_rate + neuron_tail_dfs.tail_total_response_rate
    top_all_cutoff = neuron_tail_dfs['tail_total_predictresponse_rate'].quantile(toppercentage)
    top_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predictresponse_rate'] >= top_all_cutoff]

    # get all the neurons that are 1) in 20% motor correlated and 2) within ARTR ROI
    l_roi_index = []
    r_roi_index = []
    vspn_obj = Polygon(vspn)
    for n in loc.index:
        point_obj = Point(loc.loc[n])
        if vspn_obj.contains(point_obj):
            if loc.loc[n, 'xpos'] < midline:
                l_roi_index = l_roi_index + [n]
            else:
                r_roi_index = r_roi_index + [n]
    # l_roi_index = loc[
    #     (loc['xpos'] >= artr[2]) & (loc['xpos'] <= artr[3]) &
    #     (loc['ypos'] >= artr[0]) & (loc['ypos'] <= artr[1])].index
    # l_roi_index = [n for n in l_roi_index if n in top_neurons.index]
    # r_roi_index = loc[
    #     (loc['xpos'] > artr[3]) & (loc['xpos'] <= artr[4]) &
    #     (loc['ypos'] >= artr[0]) & (loc['ypos'] <= artr[1])].index
    # r_roi_index = [n for n in r_roi_index if n in top_neurons.index]
    corr_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([l_roi_index, ['tail_l', 'tail_r']], names=['n_index', 'tail_side']),
        columns=['r2', 'p'])
    corr_df['anatomy_side'] = ['left'] * len(corr_df)
    corr_df_r = pd.DataFrame(
        index=pd.MultiIndex.from_product([r_roi_index, ['tail_l', 'tail_r']], names=['n_index', 'tail_side']),
        columns=['r2', 'p'])
    corr_df_r['anatomy_side'] = ['right'] * len(corr_df_r)
    corr_df = pd.concat([corr_df, corr_df_r])
    for plane in planerange:
        l_roi_index_plane = [int(n // 1) for n in l_roi_index if np.rint(n % 1 * 100) == plane]
        r_roi_index_plane = [int(n // 1) for n in r_roi_index if np.rint(n % 1 * 100) == plane]
        # gather all other relevant data
        fish_plane = fish + '/plane_' + str(plane)
        save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish_plane + '/data/'
        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]
        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        tail_df['t_dt'] = pd.to_datetime(tail_df['t_dt'], format='%H:%M:%S.%f')
        tail_df['t_dt'] = [x.time() for x in tail_df['t_dt']]
        tail_df = tail_df.ffill()
        ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
        # collect neuron traces on the left and right of the midline
        l_roi_normf = ysort_normf.loc[l_roi_index_plane]
        r_roi_normf = ysort_normf.loc[r_roi_index_plane]
        # get left and right tail regressor
        tail_byplane = tail_df.groupby(['frame'])['tail_sum'].mean()
        empty_frame = [i for i in range(1, ysort_normf.shape[1] + 1) if i not in tail_byplane.index]
        for frame in empty_frame:
            tail_byplane[frame] = np.nan
        tail_byplane = list(tail_byplane)
        med_tail = np.nanmean(tail_byplane)
        r_tail = [i if i > med_tail else med_tail for i in tail_byplane]
        l_tail = [i if i < med_tail else med_tail for i in tail_byplane]
        # perform regression
        for n in l_roi_normf.index:
            neuron_trace = l_roi_normf.loc[n]
            n_index = n + 0.01 * plane
            corr_df.loc[(n_index, 'tail_l'), 'r2'], corr_df.loc[(n_index, 'tail_l'), 'p'] = pearsonr(neuron_trace,
                                                                                                     l_tail)
            corr_df.loc[(n_index, 'tail_r'), 'r2'], corr_df.loc[(n_index, 'tail_r'), 'p'] = pearsonr(neuron_trace,
                                                                                                     r_tail)
        for n in r_roi_normf.index:
            neuron_trace = r_roi_normf.loc[n]
            n_index = n + 0.01 * plane
            corr_df.loc[(n_index, 'tail_l'), 'r2'], corr_df.loc[(n_index, 'tail_l'), 'p'] = pearsonr(neuron_trace,
                                                                                                     l_tail)
            corr_df.loc[(n_index, 'tail_r'), 'r2'], corr_df.loc[(n_index, 'tail_r'), 'p'] = pearsonr(neuron_trace,
                                                                                                     r_tail)
    return corr_df

def artr_lateralization_pop(artr, neuron_tail_dfs, loc, fish, planerange, toppercentage):
    """
    Calculate the Pearson's correlation coefficient with left/right swim for the left and right anatomy region
        artr: the list indicates the region of interest in the format of [ymin, ymax, xmin, xmid, xmax]
        neuron_tail_dfs: the neuron index across the whole brain as well as the tail decoding accuracy and related parameters
        loc: the x and y location for all neurons
        fish: the string of fish name for reference of file location
        planerange: the range of plane used for analysis
        toppercentage: top percentage of tail neurons to look atb
    """
    # DECIDE TOP RESPONDERS
    toppercentage = 1 - toppercentage
    neuron_tail_dfs[
        'tail_total_predictresponse_rate'] = neuron_tail_dfs.tail_total_predict_rate + neuron_tail_dfs.tail_total_response_rate
    top_all_cutoff = neuron_tail_dfs['tail_total_predictresponse_rate'].quantile(toppercentage)
    top_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predictresponse_rate'] >= top_all_cutoff]

    # get all the neurons that are 1) in 20% motor correlated and 2) within ARTR ROI
    l_roi_index = loc[
        (loc['xpos'] >= artr[2]) & (loc['xpos'] <= artr[3]) &
        (loc['ypos'] >= artr[0]) & (loc['ypos'] <= artr[1])].index
    l_roi_index = [n for n in l_roi_index if n in top_neurons.index]
    r_roi_index = loc[
        (loc['xpos'] > artr[3]) & (loc['xpos'] <= artr[4]) &
        (loc['ypos'] >= artr[0]) & (loc['ypos'] <= artr[1])].index
    r_roi_index = [n for n in r_roi_index if n in top_neurons.index]
    corr_df = pd.DataFrame(
        index=pd.MultiIndex.from_product([l_roi_index, ['tail_l', 'tail_r']], names=['n_index', 'tail_side']),
        columns=['r2', 'p'])
    corr_df['anatomy_side'] = ['left'] * len(corr_df)
    corr_df = pd.DataFrame(index = pd.MultiIndex.from_product([['left', 'right'], planerange,
                        ['tail_l', 'tail_r']], names = ['anatomy_side', 'plane', 'tail_side']), columns = ['r2', 'p'])
    for plane in planerange:
        l_roi_index_plane = [int(n // 1) for n in l_roi_index if np.rint(n % 1 * 100) == plane]
        r_roi_index_plane = [int(n // 1) for n in r_roi_index if np.rint(n % 1 * 100) == plane]
        # gather all other relevant data
        fish_plane = fish + '/plane_' + str(plane)
        save_path = '/Users/zichenhe/Desktop/Naumann Lab/danionella_zebrafish_comparative/' + fish_plane + '/data/'
        frametimes_df = pd.read_csv(save_path + 'frametimes_df.csv', index_col=0)
        frametimes_df['time'] = pd.to_datetime(frametimes_df['time'], format='%H:%M:%S.%f')
        frametimes_df['time'] = [x.time() for x in frametimes_df['time']]
        tail_df = pd.read_csv(save_path + 'tail_df.csv', index_col=0)
        tail_df['t_dt'] = pd.to_datetime(tail_df['t_dt'], format='%H:%M:%S.%f')
        tail_df['t_dt'] = [x.time() for x in tail_df['t_dt']]
        tail_df = tail_df.ffill()
        ysort_normf = pd.read_csv(save_path + 'ysort_normf.csv', index_col=0)
        # collect neuron traces on the left and right of the midline
        l_roi_normf = ysort_normf.loc[l_roi_index_plane]
        r_roi_normf = ysort_normf.loc[r_roi_index_plane]
        # get left and right tail regressor
        tail_byplane = tail_df.groupby(['frame'])['tail_sum'].mean()
        empty_frame = [i for i in range(1, ysort_normf.shape[1] + 1) if i not in tail_byplane.index]
        for frame in empty_frame:
            tail_byplane[frame] = np.nan
        tail_byplane = list(tail_byplane)
        med_tail = np.nanmean(tail_byplane)
        tail_byplane = [i - med_tail for i in tail_byplane]
        r_tail = [i if i > 0 else 0 for i in tail_byplane]
        l_tail = [-i if i < 0 else 0 for i in tail_byplane]
        # perform regression for population of neuron
        if not l_roi_index_plane == []:
            neuron_trace_l = l_roi_normf.mean()
            corr_df.loc[('left', plane, 'tail_l'), 'r2'], corr_df.loc[('left', plane, 'tail_l'), 'p'] = pearsonr(
                neuron_trace_l, l_tail)
            corr_df.loc[('left', plane, 'tail_r'), 'r2'], corr_df.loc[('left', plane, 'tail_r'), 'p'] = pearsonr(
                neuron_trace_l, r_tail)
        if not r_roi_index_plane == []:
            neuron_trace_r = r_roi_normf.mean()
            corr_df.loc[('right', plane, 'tail_l'), 'r2'], corr_df.loc[('right', plane, 'tail_l'), 'p'] = pearsonr(
                neuron_trace_r, l_tail)
            corr_df.loc[('right', plane, 'tail_r'), 'r2'], corr_df.loc[('right', plane, 'tail_r'), 'p'] = pearsonr(
                neuron_trace_r, r_tail)
    return corr_df