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

def quick_plotting_merged(img1, img2, brightness_factor = 10):
    fig, ax = plt.subplots(1, 3, figsize = (10, 10))
    ax[0].imshow(img1,cmap = 'gray', vmax = np.percentile(img1, 99))
    ax[0].set_title('Reference')
    ax[1].imshow(img2, cmap = 'gray', vmax = np.percentile(img2, 99))
    ax[1].set_title('Comparison')

    merged_img = np.zeros((img1.shape[0], img1.shape[1], 3))
    merged_img[:, :, 0] = img1 # reference is red
    merged_img[:, :, 1] = img2 # comparison is green
    # merged_img = merged_img / np.max(merged_img)
    merged_img = np.clip(merged_img + brightness_factor, 0, 255).astype(np.uint8)
    ax[2].imshow(merged_img, vmax = np.percentile(merged_img, 99))
    ax[2].set_title('Merged')

    return ax

def quick_plotting_image_pairs(img1, img2):
    fig, ax = plt.subplots(1, 2, figsize = (10, 10))
    ax[0].imshow(img1,cmap = 'gray', vmax = np.percentile(img1, 99))
    ax[0].set_title('image 1')
    ax[1].imshow(img2, cmap = 'gray', vmax = np.percentile(img2, 99))
    ax[1].set_title('image 2')

    return ax

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

    # Use TwoSlopeNorm to center at 0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    
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


def make_color_list_from_cmap(num_colors, colormap = plt.cm.viridis):
    '''
    Create a specific list of colors from a colormap, evenly spaced across the colormap given the number of total colors
    '''

    specific_ind_per_number = np.linspace(0, 1, num_colors)
    colors = [colormap(x) for x in specific_ind_per_number]

    return colors

def interpolate_colors(start_hex, end_hex, n):
    # Convert hex to RGB
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#%02x%02x%02x' % rgb
    
    start_rgb = np.array(hex_to_rgb(start_hex))
    end_rgb = np.array(hex_to_rgb(end_hex))
    
    colors = [
        rgb_to_hex(tuple((start_rgb + (end_rgb - start_rgb) * i / (n - 1)).astype(int)))
        for i in range(n)
    ]
    
    return colors


