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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.cm as cm


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

def volumetric_plot_tailneurons(neuron_tail_dfs, loc, plane_toppercentage):
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
    blues_cmap = plt.get_cmap('Blues')(list(neuron_tail_dfs['tail_total_response_rate']))
    blues_cmap = np.array(blues_cmap.T) * 0.5
    reds_cmap = plt.get_cmap('Reds')(list(neuron_tail_dfs['tail_total_predict_rate']))
    reds_cmap = np.array(reds_cmap.T) * 0.5
    scattercolor = np.add(blues_cmap, reds_cmap)[:3, :].T
    scattercolor_df = pd.DataFrame(index = neuron_tail_dfs.index, data = scattercolor)
    scatter = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'], z=loc.loc[neuron_tail_dfs.index, 'zpos'],
                mode='markers', opacity=0.5, marker=dict(size=2, symbol="circle", color= scattercolor_df.loc[neuron_tail_dfs.index]))
    #determine top responders in each plane
    #get range of planes from loc.zpos
    #planes = np.unique(loc['zpos'])
    plane_toppercentage = 1 - plane_toppercentage
    top_responder_cutoff = neuron_tail_dfs.tail_total_response_rate.quantile(plane_toppercentage)
    top_predictor_cutoff = neuron_tail_dfs.tail_total_predict_rate.quantile(plane_toppercentage)
    top_neurons = neuron_tail_dfs[(neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff) & (
                     neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff)]
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
    fig.update_layout( coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    #plot fig_response
    scattercolor_response = (blues_cmap[:3, :] * 2).T
    scattercolor_response_df = pd.DataFrame(index=neuron_tail_dfs.index, data=scattercolor_response)
    scatter_response = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                           z=loc.loc[neuron_tail_dfs.index, 'zpos'],
                           mode='markers', opacity=0.5,
                           marker=dict(size=2, symbol="circle", color=scattercolor_response_df.loc[neuron_tail_dfs.index]))
    # determine top responders in each plane
    top_response_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_response_rate'] >= top_responder_cutoff].index
    # top_response_neurons = []
    # for plane in planes:
    #     neuron_tail_df = neuron_tail_dfs[np.round((neuron_tail_dfs.index % 1), 2) * 100 == plane]
    #     top_responder_cutoff = neuron_tail_df.tail_total_response_rate.quantile(plane_toppercentage)
    #     top_response_neuron = neuron_tail_df[neuron_tail_df['tail_total_response_rate'] >= top_responder_cutoff].index
    #     top_response_neurons = top_response_neurons + list(top_response_neuron)
    topscatter_response = go.Scatter3d(x=loc.loc[top_response_neurons, 'xpos'], y=loc.loc[top_response_neurons, 'ypos'],
                              z=loc.loc[top_response_neurons, 'zpos'],
                              mode='markers', opacity=0.5,
                              marker=dict(size=5, symbol="circle", color=scattercolor_response_df.loc[top_response_neurons]))
    data_response = [scatter_response] + [topscatter_response]
    fig_response = go.Figure(data=data_response)
    fig_response.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig_response.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    #plot fig_predict
    scattercolor_predict = (reds_cmap[:3, :] * 2).T
    scattercolor_predict_df = pd.DataFrame(index=neuron_tail_dfs.index, data=scattercolor_predict)
    scatter_predict = go.Scatter3d(x=loc.loc[neuron_tail_dfs.index, 'xpos'], y=loc.loc[neuron_tail_dfs.index, 'ypos'],
                                    z=loc.loc[neuron_tail_dfs.index, 'zpos'], mode='markers', opacity=0.5,
                                    marker=dict(size=2, symbol="circle", color=scattercolor_predict_df.loc[neuron_tail_dfs.index]))
    top_predict_neurons = neuron_tail_dfs[neuron_tail_dfs['tail_total_predict_rate'] >= top_predictor_cutoff].index
    # top_predict_neurons = []
    # for plane in planes:
    #     neuron_tail_df = neuron_tail_dfs[np.round((neuron_tail_dfs.index % 1), 2) * 100 == plane]
    #     top_predictor_cutoff = neuron_tail_df.tail_total_predict_rate.quantile(plane_toppercentage)
    #     top_predict_neuron = neuron_tail_df[neuron_tail_df['tail_total_predict_rate'] >= top_predictor_cutoff].index
    #     top_predict_neurons = top_predict_neurons + list(top_predict_neuron)
    topscatter_predict = go.Scatter3d(x=loc.loc[top_predict_neurons, 'xpos'], y=loc.loc[top_predict_neurons, 'ypos'],
                                       z=loc.loc[top_predict_neurons, 'zpos'],mode='markers', opacity=0.5,
                                       marker=dict(size=5, symbol="circle", color=scattercolor_predict_df.loc[top_predict_neurons]))
    data_predict = [scatter_predict] + [topscatter_predict]
    fig_predict= go.Figure(data=data_predict)
    fig_predict.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig_predict.update_layout(coloraxis_showscale=True, scene_aspectmode='manual', scene_aspectratio=dict(x=1, y=1.8, z=0.6))

    fig_scatter, ax_combine_scatter = plt.subplots(1, 1, dpi = 240, figsize = (5, 5))
    ax_combine_scatter.scatter(neuron_tail_dfs['tail_total_response_rate'], neuron_tail_dfs['tail_total_predict_rate'], s=0.5,
                               c=scattercolor_df.loc[neuron_tail_dfs.index])
    ax_combine_scatter.scatter(top_neurons['tail_total_response_rate'], top_neurons['tail_total_predict_rate'], s=2,
                               c=scattercolor_df.loc[top_neurons.index])
    ax_combine_scatter.axvline(top_responder_cutoff, c='royalblue', linestyle=':', linewidth=2)
    ax_combine_scatter.axhline(top_predictor_cutoff, c='firebrick', linestyle=':', linewidth=2)
    ax_combine_scatter.set_xlim([0, 1])
    ax_combine_scatter.set_ylim([0, 1])
    ax_combine_scatter.spines[['top', 'right']].set_visible(False)
    ax_combine_scatter.set_aspect('equal', adjustable='box')
    ax_combine_scatter.set_xlabel('neuron predicted by %tail')
    ax_combine_scatter.set_ylabel('neuron predicting %tail')

    return fig, fig_response, fig_predict, fig_scatter