# functions to preprocess and help process photostimulation data 

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import caiman as cm
from PIL import Image
from scipy.signal import find_peaks 

from bruker_images import read_xml_to_str, read_xml_to_root
from utilities import arrutils, statutils
from utilities.roiutils import create_circular_mask
from utilities.coordutils import rotate_transform_coors, closest_coordinates

def make_population_avg_evoked_trace_plots(special_cells_list, frame_window, subplot = None, title = '', ylim = [-0.03, 0.03]):
    '''
    Population average response plot for photostim responses

    special_cells_array = list of array n cells x n traces x n frames for the frame window around each photostim event
    frame_window = list of frames that are around the photostim events to average over
    subplot = axis to plot on
    title = title of the plot
    ylim = y axis limits
    show_error = boolean to show error

    '''
    
    if subplot is not None:
        ax_n = subplot
        ax_n.set_title(title)
        ax_n.set_ylim(ylim[0], ylim[1])
    else:
        ax_n = plt
        plt.figure(figsize = (8, 8))
        plt.title(title)
        plt.ylim(ylim[0], ylim[1])
    
    # change the list of cell arrays into a mean array for each cell
    special_cells_array = np.zeros(shape = (len(special_cells_list), np.diff(frame_window)[0]))
    error_array = np.zeros(shape = (len(special_cells_list), np.diff(frame_window)[0])) # this was make a non-normalized error calculation...
    for m, l in enumerate(special_cells_list):
        special_cells_array[m] = np.nanmean(l, axis = 0)
        error_array[m] = np.std(l, axis = 0) / np.sqrt(l.shape[0])

    data_mean = np.nanmean(special_cells_array, axis = 0)
    data_base = np.nanmean(special_cells_array[:, :-frame_window[0]], axis = 1)
    plot_data = data_mean - np.nanmean(data_base)

    # figure out how to calculate error...

    ax_n.plot(plot_data, color = 'k')
    ax_n.axhline(0, color = 'black', alpha = 0.2, linestyle = '--')
    ax_n.axvline(x = -frame_window[0],  color = 'red', alpha = 0.4)
    ax_n.text(0.95, 0.95, f'n = {len(special_cells_array)}', transform=ax_n.transAxes, fontsize=18,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

