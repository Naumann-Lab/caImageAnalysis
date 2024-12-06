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

def match_cell_ids(cell_arr1, stats_dict1, cell_arr2, stats_dict2):
    '''
    Match cell ids based on overlap of cells, 1-to-1. If no match, then left with a nan
    cell_arr1: list of cell ids from dataset 1, this is the array of cell ids that you want to match to
    stats_dict1: suite2p stats dictionary from dataset 1
    cell_arr2: list of cell ids from dataset 2
    stats_dict2: suite2p stats dictionary from dataset 2

    returns an array of each cell id from cell_arr1 with the corresponding cell id from cell_arr2
    '''
    from scipy.optimize import linear_sum_assignment

    # Step 1: make overlap matrix
    overlap_matrix = np.zeros((len(cell_arr1), len(cell_arr2)))
    for b, c in enumerate(cell_arr1):
        xpix_1 = stats_dict1[c]['xpix']
        ypix_1 = stats_dict1[c]['ypix']
        for d in cell_arr2:
            xpix_2 = stats_dict2[d]['xpix']
            ypix_2 = stats_dict2[d]['ypix']
            overlap_size, overlap_ratio  = get_overlap_between_neurons(xpix_1, ypix_1, xpix_2, ypix_2, plot = False)
            overlap_matrix[b, d] = overlap_ratio

    # Step 2: Solve the assignment problem (1-to-1 matching)
    row_ind, col_ind = linear_sum_assignment(overlap_matrix, maximize = True)  # Hungarian Algorithm

    # Step 3: Construct matched results, assign matches where they exist, else remains nan
    # length of cell_arr1, fill each cell id with the corresponding cell id from dataset2
    matched_cell_ids = np.full(shape = (len(cell_arr1)), fill_value = np.nan) 
    for i, j in zip(row_ind, col_ind):
        dataset1_cell_id = int(i)
        dataset2_cell_id = int(j)
        matched_cell_ids[dataset1_cell_id] = dataset2_cell_id
    
    return matched_cell_ids

def get_overlap_between_neurons(xpix1, ypix1, xpix2, ypix2, plot=False):
    """
    From Jacob
    using boundary of entire neuron, rather than just the center of mass
    xpix is assumed to be list of x coordinates, ypix is y coordinates
    returns a float which is the ratio of overlap between the neurons
    and an int which is the number of pixels shared by the neurons
    """
    import matplotlib.pyplot as plt

    # find the ratio/pixels of overlap, use set function to do logical &, not
    coords1 = set(zip(xpix1, ypix1)); coords2 = set(zip(xpix2, ypix2))
    overlap = coords1 & coords2 # logical AND of coords
    overlap_size = len(overlap)
    unique_pixels = len(coords1 | coords2) #in coords 1 and not in coords2
    overlap_ratio = overlap_size / unique_pixels if unique_pixels > 0 else 0

    # if you want to plot them..
    if(plot):
        plt.figure(figsize=(4, 4), dpi=100)
        plt.scatter(xpix1, ypix1, c='blue', marker='o', label='Neuron 1', alpha=0.5)
        plt.scatter(xpix2, ypix2, c='red', marker='o', label='Neuron 2', alpha=0.5)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.gca().invert_yaxis()
        plt.legend()
        plt.grid(True)
        plt.title(overlap_ratio)
        plt.show()

    return overlap_size, overlap_ratio   


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

