import numpy as np
import math
from pathlib import Path
import os
import pandas as pd

# utility functions for working with coordinates

def closest_coordinates(target_x, target_y, coordinates):
    '''
    Target x and y are what you want to find a match for in the list of coordinates.
    '''
    min_distance = float('inf')
    closest_coord = None

    for n, coord in enumerate(coordinates):
        x, y = coord
        distance = math.sqrt((target_x - x)**2 + (target_y - y)**2)
        if distance < min_distance:
            min_distance = distance
            closest_coord = coord
            closest_cell_id = n
            
    return closest_coord, closest_cell_id

def closest_coordinates_1to1(source_coords,
                            target_coords,
                            xy_offset=(0.0, 0.0),
                            max_distance=None):
    """
    Perform unbiased 1-to-1 matching between two sets of coordinates
    by minimizing total Euclidean distance (Hungarian algorithm).

    Parameters
    ----------
    source_coords : array-like, shape (N, 2)
        Reference coordinates (e.g. OMR cells)
    target_coords : array-like, shape (M, 2)
        Coordinates to be shifted and matched (e.g. stim cells)
    xy_offset : tuple (dx, dy)
        Offset applied to target_coords BEFORE matching.
        Convention:
            target_aligned = target_coords + (dx, dy)
        If dx = -4, target is shifted left by 4 pixels.
    max_distance : float or None
        Optional distance cutoff (pixels). Matches beyond this are discarded.

    Returns
    -------
    matches : dict
        source_index -> target_index
    distances : dict
        source_index -> distance after offset correction
    aligned_target_coords : ndarray, shape (M, 2)
        Offset-corrected target coordinates
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment

    source_coords = np.asarray(source_coords, dtype=float)
    target_coords = np.asarray(target_coords, dtype=float)

    dx, dy = xy_offset
    aligned_target_coords = target_coords + np.array([dx, dy])

    if source_coords.size == 0 or target_coords.size == 0:
        return {}, {}, aligned_target_coords

    # Distance matrix using offset-corrected stim coords
    D = cdist(source_coords, aligned_target_coords)

    src_idx, tgt_idx = linear_sum_assignment(D)

    matches = {}
    distances = {}

    for s, t in zip(src_idx, tgt_idx):
        d = D[s, t]
        if max_distance is None or d <= max_distance:
            matches[s] = t
            distances[s] = d

    return matches, distances, aligned_target_coords

def match_cell_ids(cell_arr1, stats_dict1, cell_arr2, stats_dict2,
                   distance_threshold_um=10, overlap_threshold=0.01,
                   um_to_px = 0.6, xy_offset=(0, 0)):
    """
    Match cell ids based on overlap of cells, ensuring a strict 1-to-1 match.
    If no good match exists, the result is NaN.
    Only based on location of cells

    cell_arr1: list of cell ids from dataset 1, this is the array of cell ids that you want to match to
    stats_dict1: suite2p stats dictionary from dataset 1
    cell_arr2: list of cell ids from dataset 2
    stats_dict2: suite2p stats dictionary from dataset 2
    distance_threshold_um: maximum distance in um for a match to be considered valid
    overlap_threshold: minimum overlap ratio for a match to be considered valid
    um_to_px: conversion factor from um to pixels for distance threshold calculation
    xy_offset: (dx, dy) offset to apply to dataset 2 in pixels, defualt is (0,0), applied to the 2nd dataset

    returns 
    matched_cell_ids: a DICTIONARY of each cell id from cell_arr1 (key) with the corresponding cell id from cell_arr2
    """
    from scipy.optimize import linear_sum_assignment
    dx, dy = xy_offset # pixel offset in x and y
    distance_threshold = distance_threshold_um / um_to_px # Convert um to pixels

    # Step 1: Compute overlap matrix
    overlap_matrix = np.zeros((len(cell_arr1), len(cell_arr2)))
    center_distances = np.zeros((len(cell_arr1), len(cell_arr2)))
    
    centers1 = {c: (np.mean(stats_dict1[c]['xpix']), np.mean(stats_dict1[c]['ypix'])) for c in cell_arr1}
    # Note: Apply offset to dataset 2 centers
    centers2 = {c: (np.mean(stats_dict2[c]['xpix']) + dx, np.mean(stats_dict2[c]['ypix']) + dy) for c in cell_arr2}
    
    for i, c1 in enumerate(cell_arr1):
        xpix_1, ypix_1 = stats_dict1[c1]['xpix'], stats_dict1[c1]['ypix']
        center1 = centers1[c1]
        
        for j, c2 in enumerate(cell_arr2):
            # xpix_2, ypix_2 = stats_dict2[c2]['xpix'], stats_dict2[c2]['ypix']
            xpix_2_shifted = stats_dict2[c2]['xpix'] + dx
            ypix_2_shifted = stats_dict2[c2]['ypix'] + dy
            center2 = centers2[c2]
            
            # Compute overlap
            overlap_size, overlap_ratio = get_overlap_between_neurons(xpix_1, ypix_1, xpix_2_shifted, ypix_2_shifted)
            overlap_matrix[i, j] = overlap_ratio if overlap_ratio >= overlap_threshold else 0
            
            # Compute Euclidean distance between centers
            center_distances[i, j] = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    # Step 2: Solve the assignment problem (maximize overlap)
    row_ind, col_ind = linear_sum_assignment(overlap_matrix - center_distances * 0.001, maximize=True)

    # Step 3: Store matches in a dictionary
    matched_cell_ids = {cell_id: np.nan for cell_id in cell_arr1}  # Default to NaN

    used_final_cells = set()
    for i, j in zip(row_ind, col_ind):
        if overlap_matrix[i, j] > 0 and center_distances[i, j] <= distance_threshold:
            dataset1_cell_id = cell_arr1[i]
            dataset2_cell_id = cell_arr2[j]

            if dataset2_cell_id not in used_final_cells:
                matched_cell_ids[dataset1_cell_id] = dataset2_cell_id
                used_final_cells.add(dataset2_cell_id)
        else:  
            # Try closest match if overlap/distance is insufficient
            target_center = centers1[cell_arr1[i]]
            closest_coord, closest_cell_id = closest_coordinates(target_center[0], target_center[1], list(centers2.values()))

            if math.sqrt((target_center[0] - closest_coord[0])**2 + (target_center[1] - closest_coord[1])**2) <= distance_threshold:
                dataset2_cell_id = cell_arr2[closest_cell_id] if closest_cell_id not in used_final_cells else None
            else:
                dataset2_cell_id = None

            if dataset2_cell_id is not None:
                matched_cell_ids[cell_arr1[i]] = dataset2_cell_id
                used_final_cells.add(dataset2_cell_id)
    
    return matched_cell_ids

def get_overlap_between_neurons(xpix1, ypix1, xpix2, ypix2):
    """
    Compute the overlap ratio between two neurons using pixel coordinates.
    From Jacob
    using boundary of entire neuron, rather than just the center of mass
    xpix is assumed to be list of x coordinates, ypix is y coordinates
    returns a float which is the ratio of overlap between the neurons
    and an int which is the number of pixels shared by the neurons
    """
    coords1 = set(zip(xpix1, ypix1))
    coords2 = set(zip(xpix2, ypix2))
    overlap = coords1 & coords2
    overlap_size = len(overlap)
    unique_pixels = len(coords1 | coords2)
    overlap_ratio = overlap_size / unique_pixels if unique_pixels > 0 else 0
    
    return overlap_size, overlap_ratio

def find_distance_between_coordinates(coord1, coord2):
    """
    Find the distance between two coordinates.

    Parameters:
    - coord1: Tuple (x1, y1).
    - coord2: Tuple (x2, y2).

    Returns:
    - Distance between the two coordinates.
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def determine_nearby_cells_xy(target_coord, coordinates, ums_per_px, radius_um=10):
    '''
    Determine the nearby cells from a target cell in XY

    target_coord = center coordinate
    coordinates = list of coordinates on the same XY plane to find nearby neighbors
    ums_to_px = scaling factor of ums to pixels
    radius_um = radius of the circle around the target coordinate that you are looking for closest neighbors

    returns
    nearby_coords_index = index of the coordinates list that are the nearby cells
    nearby_coords_list = the coordinates of the nearby cells
    '''
    import math
    radius_px = radius_um / ums_per_px

    nearby_coords_index = []
    nearby_coords_list = []

    for idx, point in enumerate(coordinates):
        dist = math.hypot(point[0] - target_coord[0], point[1] - target_coord[1])
        if dist <= radius_px:
            nearby_coords_index.append(idx)
            nearby_coords_list.append(point)

    return nearby_coords_index, nearby_coords_list

def rotate_transform_coors(coordinates, angle_degrees, translation=(0, 0)):
    """
    Rotate and transform 2D coordinates.

    Parameters:
    - coordinates: List of (x, y) coordinates.
    - angle_degrees: Rotation angle in degrees.
    - translation: Tuple (tx, ty) for translation (default is (0, 0)).

    Returns:
    - List of transformed (x', y') coordinates.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix
    rotation_matrix = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                                [np.sin(angle_radians), np.cos(angle_radians)]])

    # Apply rotation
    rotated_coordinates = np.dot(rotation_matrix, np.array(coordinates).T).T

    # Apply translation
    translated_coordinates = rotated_coordinates + np.array(translation)

    return translated_coordinates.tolist()

# automating sidedness for cells across the FOV

def return_midline_coords_per_plane_dict(stimpath):
    '''
    Gather midline coordinates for each plane, stored in dictionary
    stimpath = folder to the photostim dataset

    Returns:
    midline_dict = dictionary with plane str as keys, items are the midline coordinates for that plane

    '''
    midline_dict = {}
    with os.scandir(stimpath.joinpath('output_folders')) as entries:
        for entry in entries:
            midline_npy = Path(entry.path).joinpath('rois\midline.npy')
            plane = Path(entry.path).name

            midline = np.load(midline_npy)
            x = midline[:, 0]
            y = midline[:, 1]
            m, b = np.polyfit(y, x, 1)
            y_full = np.arange(0, 512)
            x_fit = m * y_full + b
            x_fit_int = np.rint(x_fit).astype(int)
            midline_coords = np.column_stack((x_fit_int, y_full))

            midline_dict[plane] = midline_coords

    return midline_dict

# for one cell
def determine_sideness_of_cell(cell_coordinates, x_midline):
    if cell_coordinates[0] < x_midline: # cell on left hemisphere
        side = 'L'
    if cell_coordinates[0] >= x_midline:
        side = 'R'
    return side

# for an array
def cell_side_of_midline(midline_coords, cell_coord):
    """
    midline_coords: list of (x, y) tuples or Nx2 numpy array
    cell_coord: (x, y) tuple for the cell position
    """
    x_mid = midline_coords[:, 0]
    y_mid = midline_coords[:, 1]

    # Find the closest midline y to the cell y
    idx = np.argmin(np.abs(y_mid - cell_coord[1]))
    x_at_same_y = x_mid[idx]

    # Compare x positions
    if cell_coord[0] < x_at_same_y:
        return "L"
    if cell_coord[0] >= x_at_same_y:
        return "R"



def match_omr_to_stim_coords(
    omr_indices,
    omr_coords,
    stim_indices,
    stim_coords,
    stimmed_indices=None,
    xy_offset=(0, 0),
    max_distance=np.inf,
    tile_size=100
):
    """
    Match OMR cells to stim cells (coordinates) with optional bias toward stimmed cells.

    Parameters
    ----------
    omr_indices : list/array
        Original indices of OMR cells.
    omr_coords : Nx2 array
        OMR cell coordinates.
    stim_indices : list/array
        Original indices of stim cells.
    stim_coords : Mx2 array
        Stim cell coordinates.
    stimmed_indices : list/array, optional
        Indices of stimmed cells to prioritize.
    xy_offset : tuple
        Optional XY offset to apply to stim coordinates.
    max_distance : float
        Maximum allowed distance for matching.
    tile_size : int
        Tile size in pixels for local matching.

    Returns
    -------
    omr_to_stim_matches_dict : dict
        Dictionary mapping OMR indices → stim indices (original IDs).
    aligned_stim_coords : np.ndarray
        Transformed stim coordinates after optional offset.
    """

    from scipy.spatial import cKDTree

    stim_coords = np.array(stim_coords) + np.array(xy_offset)
    omr_coords = np.array(omr_coords)
    stimmed_indices = stimmed_indices if stimmed_indices is not None else []

    # ---------------- Stage 1: priority matching for stimmed cells ----------------
    stim_matches = []
    claimed_omr_ids = set()

    if len(stimmed_indices) > 0:
        stimmed_coords = stim_coords[stimmed_indices]
        for i, stim in zip(stimmed_indices, stimmed_coords):
            # determine tile
            x0 = max(0, int(stim[0] // tile_size) * tile_size)
            y0 = max(0, int(stim[1] // tile_size) * tile_size)
            x1, y1 = x0 + tile_size, y0 + tile_size

            in_tile = [j for j, c in enumerate(omr_coords)
                       if x0 <= c[0] < x1 and y0 <= c[1] < y1]

            if len(in_tile) > 0:
                tile_coords = np.array([omr_coords[j] for j in in_tile])
                distances = np.linalg.norm(tile_coords - stim, axis=1)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                if min_dist <= max_distance:
                    idx = in_tile[min_idx]
                else:
                    # fallback to global nearest neighbor
                    distances_global = np.linalg.norm(omr_coords - stim, axis=1)
                    idx = np.argmin(distances_global)
            else:
                # fallback to global nearest neighbor
                distances_global = np.linalg.norm(omr_coords - stim, axis=1)
                idx = np.argmin(distances_global)

            stim_matches.append((omr_indices[idx], stim_indices[i]))
            claimed_omr_ids.add(omr_indices[idx])

    # ---------------- Optional: compute offset from stim matches ----------------
    if stim_matches:
        matched_omr_coords = np.array([omr_coords[omr_indices.tolist().index(omr_idx)] for omr_idx, _ in stim_matches])
        matched_stim_coords = np.array([stim_coords[stim_indices.tolist().index(stim_idx)] for _, stim_idx in stim_matches])
        offset = np.mean(matched_omr_coords - matched_stim_coords, axis=0)
        aligned_stim_coords = stim_coords + offset
    else:
        aligned_stim_coords = stim_coords

    # ---------------- Stage 2: global one-to-one matching ----------------
    remaining_omr_indices = [i for i in range(len(omr_coords)) if omr_indices[i] not in claimed_omr_ids]
    remaining_coords = np.array([omr_coords[i] for i in remaining_omr_indices])
    tree = cKDTree(remaining_coords)

    claimed_stim_indices = set(stimmed_indices)  # already assigned in Stage 1
    global_matches = {}

    for stim_idx, stim_coord in zip(stim_indices, aligned_stim_coords):
        if stim_idx in claimed_stim_indices:
            continue  # skip stim cells already matched
        if len(remaining_coords) == 0:
            break
        dist, idx = tree.query(stim_coord)
        if dist > max_distance:
            continue
        omr_idx = omr_indices[remaining_omr_indices[idx]]
        if omr_idx in claimed_omr_ids:
            continue

        global_matches[omr_idx] = stim_idx
        claimed_omr_ids.add(omr_idx)
        claimed_stim_indices.add(stim_idx)

        # remove matched OMR cell
        remaining_coords = np.delete(remaining_coords, idx, axis=0)
        remaining_omr_indices.pop(idx)
        tree = cKDTree(remaining_coords) if len(remaining_coords) > 0 else None

    # ---------------- Combine mappings ----------------
    omr_to_stim_matches_dict = {omr_idx: stim_idx for omr_idx, stim_idx in stim_matches}
    omr_to_stim_matches_dict.update(global_matches)

    return omr_to_stim_matches_dict, aligned_stim_coords