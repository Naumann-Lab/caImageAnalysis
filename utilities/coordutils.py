import numpy as np
import math

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

def determine_sideness_of_cell(cell_coordinates, x_midline):
    if cell_coordinates[0] < x_midline: # cell on left hemisphere
        side = 'L'
    if cell_coordinates[0] >= x_midline:
        side = 'R'
    return side
