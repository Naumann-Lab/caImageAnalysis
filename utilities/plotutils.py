# functions to preprocess and help process photostimulation data 

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import caiman as cm
from PIL import Image
from scipy.signal import find_peaks 

from bruker_images import read_xml_to_str, read_xml_to_root
from utilities import arrutils, statutils
from utilities.roiutils import create_circular_mask
from utilities.coordutils import rotate_transform_coors, closest_coordinates


def convert_frame_to_sec(frame_lst, framerate):
    return [x / framerate for x in frame_lst]

def get_color_from_normval(value, vmin = -1, vmax = 1, cmap='coolwarm'):
    norm = plt.Normalize(vmin=vmin, vmax=vmax, clip=False)
    cmap = plt.get_cmap(cmap)

    return cmap(norm(value))

def clip_and_map_colors(values, vmin=-2, vmax=2, cmap_name='coolwarm'):
    """
    Clips the values to the range [vmin, vmax], normalizes them, maps them to colors using the specified colormap,
    and optionally visualizes the results.

    Parameters:
    - values (array-like): The array of values to be processed.
    - vmin (float): The minimum value for clipping and normalization.
    - vmax (float): The maximum value for clipping and normalization.
    - cmap_name (str): The name of the colormap to use.
    - visualize (bool): Whether to visualize the results with a scatter plot and colorbar.

    Returns:
    - clipped_values (numpy.ndarray): The clipped values.
    - colors (numpy.ndarray): The corresponding RGBA colors.
    """
    # Convert the input values to a numpy array
    values_array = np.array(values)
    
    # Clip the values between vmin and vmax
    clipped_values = np.clip(values_array, vmin, vmax)
    
    # Create a Normalize instance with the specified range
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    
    # Choose a colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Create a ScalarMappable instance and map the normalized values to colors
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mappable.to_rgba(clipped_values)

    return colors

def build_cmap_blue_to_red():
    cmap_colors = ['slateblue', '#F5F5F5', 'crimson']
    npoints = 500
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_cmap', cmap_colors, N=npoints)
    return cmap

def add_scalebar(ax, bar_pixel_length, label, location=(0.1, 0.1), bar_thickness=3, color='black'):
    """
    Adds a scale bar to a plot.
    
    :param ax: The axis on which to add the scale bar.
    :param size_in_data_units: The length of the scale bar in data units.
    :param label: The label to display above or beside the scale bar.
    :param location: The (x, y) location for the scale bar, as a fraction of the axes.
    :param bar_thickness: Thickness of the scale bar in pixels.
    :param color: Color of the scale bar.
    """
    # Get the x and y limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate the position for the scale bar
    x_pos = xlim[0] + location[0] * (xlim[1] - xlim[0])
    y_pos = ylim[0] + location[1] * (ylim[1] - ylim[0])
    
    # Draw the scale bar
    ax.hlines(y_pos, x_pos, x_pos + bar_pixel_length, colors=color, linewidth=bar_thickness)
    
    # Add the label
    ax.text(x_pos + bar_pixel_length / 2, y_pos, label, ha='center', va='bottom', color=color, fontsize=10)

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

