import sys
import os.path
import SimpleITK as sitk
import re

import json
import os
import tifffile

import numpy as np
import pandas as pd

from pathlib import Path
from tifffile import imread, imwrite

def norm(im):
    im = im.astype(np.float32)
    return (im - im.min()) / (im.max() - im.min() + 1e-8)


def give_overlay_img(R, G):
    R = norm(R)
    G = norm(G)
    overlay = np.zeros((*R.shape, 3), dtype=np.float32)
    overlay[..., 0] = R   # red = template
    overlay[..., 1] = G # green = registered
    return overlay


#FUNCTIONS HELPFUL FOR POINT TRANSFORMATIONS (FINDING NEURONS LOCATIONS)
def write_transformix_points(points, filename):
    """
    Intermediate function to save a intermediate file during transforming point position according to a trans formix file
    :param points:
    :param filename:
    :return:
    """
    with open(filename, 'w') as f:
        f.write('point\n')
        f.write(f'{len(points)}\n')
        for i, p in enumerate(points):
            f.write(f'{p[0]} {p[1]}\n')

def read_transformix_output_points(filepath):
    """
    Imtermediate function used to read the transformed point file: Parsing Transformix outputpoints.txt

    Returns
    -------
    points : (N, 2) numpy array
        Transformed [x, y] coordinates (OutputPoint)
    """

    output_points = []

    pattern = re.compile(
        r"OutputPoint\s*=\s*\[\s*([-\d\.eE]+)\s+([-\d\.eE]+)\s*\]"
    )

    with open(filepath, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                x = float(match.group(1))
                y = float(match.group(2))
                output_points.append([x, y])

    return np.array(output_points)

def transform_points_with_transformix(points, out_path, img):
    """
    Transform the points with given transformix
    points: Nx2 numpy array of [x, y]
    transform_folder: folder containing your four transform_pmap_*.txt
    transformix_exe: path to Transformix executable
    Returns: Nx2 array of transformed points
    """

    # Make temporary folder for Transformix outputs

    # 1) Save points file
    points_file = os.path.join(out_path, "points.pts")
    write_transformix_points(points, points_file)

    # load saved parameter maps
    transform_files = [
        "transform_pmap_0_0.txt",  # affine
        "transform_pmap_0_1.txt",  # BSpline
        "transform_pmap_1_0.txt",  # affine
        "transform_pmap_1_1.txt"  # BSpline
    ]
    transform_files = [out_path + '//' + f for f in transform_files]

    new_sitk = sitk.GetImageFromArray(img.astype(np.float32))
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(new_sitk)
    transformixImageFilter.SetTransformParameterMap([sitk.ReadParameterFile(f) for f in transform_files])
    transformixImageFilter.SetOutputDirectory(out_path)
    transformixImageFilter.SetFixedPointSetFileName(points_file)
    transformixImageFilter.Execute()

    output_points = read_transformix_output_points(out_path + '//outputpoints.txt')

    return output_points

def read_cellpos(plane_dir, plane):
    # gather cell location data based on the original suite2p results @ chatgpt
    stat = np.load(plane_dir + 'stat.npy', allow_pickle=True)
    avg_xpos = np.array([np.mean(cell['ypix']) for cell in stat])
    avg_ypos = np.array([np.mean(cell['xpix']) for cell in stat])
    avg_zpos = np.array([plane] * len(avg_xpos))
    pos_plane = pd.DataFrame(data = np.array([avg_xpos, avg_ypos, avg_zpos]).T,
                                           columns = ['xpos', 'ypos', 'zpos'])
    return pos_plane

def downsample_points(df, orig_shape, imgsize, downsample, rot):
    """
    Transform point coordinates to match the image transformation pipeline.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing original coordinates.
    orig_shape : tuple
        (height, width) of the original image.
    imgsize : int
        Size used in sitkalignment.embed_image (assumes square canvas).
    downsample : int
        Downsampling factor used in img[::downsample, ::downsample]
    """

    h, w = orig_shape
    x = df['xpos'].to_numpy().astype(float)
    y = df['ypos'].to_numpy().astype(float)
    # 1. flipud
    y = h - 1 - y

    # 3. embed_image (center padding)
    pad_x = (imgsize - w) / 2
    pad_y = (imgsize - h) / 2

    x = x + pad_x
    y = y + pad_y

    # 3. rotate (same center as embedded image)
    if rot != 0:
        theta = np.deg2rad(rot)

        cx = (imgsize - 1) / 2
        cy = (imgsize - 1) / 2

        x0 = x - cx
        y0 = y - cy

        x_rot = x0 * np.cos(theta) - y0 * np.sin(theta)
        y_rot = x0 * np.sin(theta) + y0 * np.cos(theta)

        x = x_rot + cx
        y = y_rot + cy

    # 4. downsample
    x = x / downsample
    y = y / downsample

    df_out = df.copy()
    df_out['xpos'] = x
    df_out['ypos'] = y

    return df_out

def upsample_points(df, orig_shape, imgsize, downsample):
    """
    Reverse the downsample_points transformation.

    Converts coordinates from the processed image space
    back into the original image coordinate space.
    """

    h, w = orig_shape

    x = df['xpos'].to_numpy().astype(float)
    y = df['ypos'].to_numpy().astype(float)

    # 1. undo downsample
    x = x * downsample
    y = y * downsample

    # 2. remove embed_image padding
    midpt = imgsize // 2
    pad_y = midpt - h // 2
    pad_x = midpt - w // 2

    x = x - pad_x
    y = y - pad_y

    df_out = df.copy()
    df_out['xpos'] = x
    df_out['ypos'] = y

    return df_out

def return_point_regions(atlas_point_up, best_zs, regions_path = "C://Data//MapZBrainATLAS//mapZebrain__regions__v1.0//"):
    """
    Find the regions from the atlas from region path in the best z plane
    :param atlas_point_up: The up-transformed (reversed transfomed) point location in the atlas
    :param best_zs:
    :param regions_path:
    :return:
    """
    # convert dataframe coords to integer indices
    z = best_zs[atlas_point_up["zpos"][0]]
    y = atlas_point_up["ypos"].to_numpy().astype(int)
    x = atlas_point_up["xpos"].to_numpy().astype(int)
    #gather all regions
    region_files = [f for f in os.scandir(regions_path)]

    n_neurons = len(atlas_point_up)
    n_regions = len(region_files)

    # boolean matrix: neurons × regions
    membership = np.zeros((n_neurons, n_regions), dtype=bool)

    region_names = []

    for i, r in enumerate(region_files):
        region = tifffile.imread(r.path)

        inside = region[z, y, x] == 255
        membership[:, i] = inside

        region_names.append(r.name.split(".")[0])
    membership_df = pd.DataFrame(membership, columns=region_names)

    atlas_point_up["regions"] = membership_df.apply(
        lambda row: list(row.index[row]), axis=1)
    return atlas_point_up

