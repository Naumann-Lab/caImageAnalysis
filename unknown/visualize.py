from ImageAnalysisCodes import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

try:
    import caiman as cm
except:
    print('no caiman available')

import pandas as pd
import scipy.stats

from colour import Color
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from cmath import rect, phase
from math import radians, degrees


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


simplified_eva_weightings = {

    'backward': [1, 0, 1],
    'left': [0, 0.25, 1],
    'medial_left': [0, 0.25, 1],
    'lateral_left': [0, 0.25, 1],
    'forward': [0, 1, 0],
    'right': [1, 0.25, 0],
    'medial_right': [1, 0.25, 0],
    'lateral_right': [1, 0.25, 0],

}


eva_weightings = {

    0 : [0, 1, 0],
    45 : [0.75, 1, 0],
    90 : [1, 0.25, 0],
    135 : [1, 0, 0.25],
    180 : [1, 0, 1],
    225 : [0.25, 0, 1],
    270 : [0, 0.25, 1],
    315 : [0, 0.75, 1],


    'BackwardRight': [1, 0, 0.25],
    'Backward': [1, 0, 1],
    'backward': [1, 0, 1],

    'BackwardLeft': [0.25, 0, 1],

    'Left': [0, 0.25, 1],
    'left': [0, 0.25, 1],

    'LL': [0, 0.25, 1],
    'xL': [0.5, 0.5, 1],
    'medial_left': [0.5, 0.5, 1],

    'Lx': [0.5, 0.75, 1],
    'lateral_left': [0.5, 0.75, 1],

    'ForwardLeft': [0, 0.75, 1],
    'Forward': [0, 1, 0],
    'forward': [0, 1, 0],

    'ForwardRight': [0.75, 1, 0],

    'Right': [1, 0.25, 0],
    'right': [1, 0.25, 0],

    'RR': [1, 0.25, 0],
    'xR': [1, 0.75, 0.5],
    'lateral_right': [1, 0.75, 0.5],

    'Rx': [1, 0.5, 0.5],
    'medial_right': [1, 0.5, 0.5],

}





def color_returner(val, theta, threshold=0.5):

    if theta < 0:
        theta += 360

    if val >= threshold:
        # Forward
        if theta >= 337.5 or theta <= 22.5:
            outputColor = [0, 1, 0]

        # Forward Right
        elif 22.5 < theta <= 67.5:
            outputColor = [0.75, 1, 0]

        # Right
        elif 67.5 < theta <= 112.5:
            outputColor = [1, 0.25, 0]

        # Backward Right
        elif 112.5 < theta <= 157.5:
            outputColor = [1, 0, 0.25]

        # Backward
        elif 157.5 < theta <= 202.5:
            outputColor = [1, 0, 1]

        # Backward Left
        elif 202.5 < theta <= 247.5:
            outputColor = [0.25, 0, 1]

        # Left
        elif 247.5 < theta <= 292.5:
            outputColor = [0, 0.25, 1]

        # Forward Left
        elif 292.5 < theta <= 337.5:
            outputColor = [0, 0.75, 1]

        # if somehow we make it to here just make it gray
        else:
            outputColor = [0.66, 0.66, 0.66]

    else:
        # if not above some minimum lets make it gray
        outputColor = [0.66, 0.66, 0.66]
    return outputColor


def color_span(stim_df, c1='blue', c2='red'):
    if 'stimulus' not in stim_df.columns:
        print('please map stimuli first')
        return

    stim_df = stim_df[stim_df.stimulus.notna()]
    n_colors = stim_df.stimulus.nunique()

    clrs = list(Color(c1).range_to(Color(c2), n_colors))
    clrs = [i.rgb for i in clrs]

    loc_dic = {}
    z = 0
    for i in stim_df.stimulus.unique():
        loc_dic[i] = clrs[z]
        z += 1

    stim_df.loc[:, 'color'] = stim_df.stimulus.map(loc_dic)
    return stim_df


def color_generator(good_loadings, c1=None, c2=None):
    """
    can provide custom colors to step or let it handle
    """
    if c1 and c2 is not None:
        clrs = list(Color(c1).range_to(Color(c2), good_loadings.highest_loading.nunique()))
        corr_colors = [i.rgb for i in clrs]
    else:
        corr_colors = sns.color_palette(n_colors=good_loadings.highest_loading.nunique())

    factor_choice_dic = {}
    n = 0
    for i in good_loadings.highest_loading.unique():
        factor_choice_dic[i] = n
        n += 1

    choices = []
    for nrn in good_loadings.index:
        neuron = good_loadings.loc[nrn]
        factor_choice = neuron.highest_loading
        choices.append(corr_colors[factor_choice_dic[factor_choice]])
        rvals = []

    return choices, corr_colors


def plot_cells(_path):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen)

    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    cells_fnd = goodLoadings.index

    for cell_num in cells_fnd:
        ypix = stats[iscell][cell_num]['ypix']
        xpix = stats[iscell][cell_num]['xpix']
        cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img < 0.9, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.mean(image, axis=0), cmap=mpl.cm.gray)
    ax.imshow(masked, cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1)
    return


def plot_factor(_path, factors=[0,1,2], variance=0.5):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return
    goodLoadings = utils.threshold_by_variance(eigen, variance)

    factor_strings = []
    if isinstance(factors, int):
        factors = [factors]

    for i in factors:
        factor_str = "FA" + str(i)
        factor_strings.append(factor_str)

    goodLoadings = goodLoadings[goodLoadings.highest_loading.isin(factor_strings)]

    colors, factor_colors = color_generator(goodLoadings)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.rot90(np.mean(image, axis=0),-1), cmap=mpl.cm.gray)
    ax.imshow(np.rot90(cell_img,-1), interpolation=None, alpha=0.7)

    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]

    ax.legend(c_lines, ['factor ' + str(i) for i in range(len(factor_colors))], loc='upper right', fontsize='x-large')
    return


def plot_factor_all(_path):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen)

    colors, factor_colors = color_generator(goodLoadings)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.mean(image, axis=0), cmap=mpl.cm.gray)
    ax.imshow(cell_img, interpolation=None, alpha=0.7)

    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]

    ax.legend(c_lines, ['factor ' + str(i) for i in range(len(factor_colors))], loc='upper right', fontsize='x-large')
    return


def full_plot(_path, save=None, age=None, variance=0.5):

    _paths = utils.pathSorter(_path)
    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return

    try:
        image = cm.load(_paths['image']['move_corrected'])
    except KeyError:
        print('please movement correct image first')
        return

    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return

    goodLoadings = utils.threshold_by_variance(eigen, variance)

    factors, loadings, x = eigen

    cell_img = np.zeros((ops['Ly'], ops['Lx']))
    cells_fnd = goodLoadings.index

    a = []
    for val in range(goodLoadings.highest_loading.nunique()):
        a.append(len(goodLoadings[goodLoadings.highest_loading == goodLoadings.highest_loading.unique()[val]]))

    for cell_num in cells_fnd:
        ypix = stats[iscell][cell_num]['ypix']
        xpix = stats[iscell][cell_num]['xpix']
        cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img < 0.9, cell_img)
    mean_img = np.mean(image, axis=0)

    colors, factor_colors = color_generator(goodLoadings)
    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in range(len(colors)):

        real_cell = goodLoadings.index[cell]
        ypix = stats[iscell][real_cell]['ypix']
        xpix = stats[iscell][real_cell]['xpix']

        for c in range(cell_img.shape[2]):
            cell_img[ypix, xpix, c] = colors[cell][c]


    c_lines = [Line2D([0], [0], color=factor_colors[i], lw=4) for i in range(len(factor_colors))]
    c_legends = ['factor ' + str(i) for i in range(len(factor_colors))]

    fig, ax = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True)

    ax[0].imshow(mean_img, cmap=mpl.cm.gray)
    ax[0].imshow(masked, cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1)

    ax[1].imshow(mean_img, cmap=mpl.cm.gray)
    ax[1].imshow(cell_img, interpolation=None, alpha=0.7)

    ax[1].legend(c_lines, c_legends, loc='upper right', fontsize='medium')
    plt.tight_layout()
    if save is not None:
        pre_sve_path = save
        ind = pre_sve_path.find('.')
        mid_string = "brain_pics" + "age_" + str(age)
        final_save_path = pre_sve_path[:ind] + mid_string + pre_sve_path[ind:]
        plt.savefig(final_save_path, dpi=300)

    plt.show()

    plt.figure(figsize=(18, 18))

    bars = plt.subplot2grid((5, 5), (0, 0), colspan=2, rowspan=3)
    bars.bar(x=np.arange(0, len(a), 1), height=a)
    bars.set_xticks(np.arange(0, len(a), 1))
    bars.set_ylabel('Neurons')
    bars.set_xlabel('Factors')
    bars.set_title('Neurons per Factor')

    a1 = plt.subplot2grid((5, 5), (0, 2), colspan=3, rowspan=1)
    a1.plot(factors[:, 0])
    a1.set_title('Factor 0')

    a2 = plt.subplot2grid((5, 5), (1, 2), colspan=3, rowspan=1)
    a2.plot(factors[:, 1])
    a2.set_title('Factor 1')

    a3 = plt.subplot2grid((5, 5), (2, 2), colspan=3, rowspan=1)
    a3.plot(factors[:, 2])
    a3.set_title('Factor 2')

    for j in [a1, a2, a3]:
        for i in range(len(stim_df)):
            j.axvspan(stim_df.iloc[i].start, stim_df.iloc[i].stop, color=stim_df.iloc[i].color, alpha=0.3,
                      label=stim_df.iloc[i].stimulus)

    plt.tight_layout()

    if save is not None:
        pre_sve_path = save
        ind = pre_sve_path.find('.')
        mid_string = "Factors" + "age_" + str(age)
        final_save_path = pre_sve_path[:ind] + mid_string + pre_sve_path[ind:]
        plt.savefig(final_save_path, dpi=300)


def plot_factor_top6(_path, factor=0):

    _paths = utils.pathSorter(_path)

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return
    try:
        eigen = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')
        return


    goodLoadings = utils.threshold_by_variance(eigen)

    factors, loadings, x = eigen

    factorplt = factors[:, factor]
    factor_str = "FA" + str(factor)
    top6 = goodLoadings.sort_values(by=factor_str, ascending=False).iloc[:6].index

    fig = plt.figure(1, figsize=(16, 16))

    AMAIN = plt.subplot2grid((4, 4), (0, 0), colspan=4)
    AMAIN.plot(factorplt, c='black')

    A1 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
    A1.plot(x[top6].iloc[:, 0], c='black')

    A2 = plt.subplot2grid((4,4), (1,2), colspan=2 )
    A2.plot(x[top6].iloc[:, 1], c='black')

    B1 = plt.subplot2grid((4,4), (2,0), colspan=2 )
    B1.plot(x[top6].iloc[:, 2], c='black')

    B2 = plt.subplot2grid((4,4), (2,2), colspan=2 )
    B2.plot(x[top6].iloc[:, 3], c='black')

    C1 = plt.subplot2grid((4,4), (3,0), colspan=2 )
    C1.plot(x[top6].iloc[:, 4], c='black')

    C2 = plt.subplot2grid((4,4), (3,2), colspan=2 )
    C2.plot(x[top6].iloc[:, 5], c='black')

    for j in [AMAIN, A1, A2, B1, B2, C1, C2]:
        for i in range(len(stim_df)):
            j.axvspan(stim_df.iloc[i].start, stim_df.iloc[i].stop, color=stim_df.iloc[i].color, alpha=0.3,
                      label=stim_df.iloc[i].stimulus)

    handles, labels = AMAIN.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    AMAIN.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=12)
    return


def plot_avg_stimulus(_path, stim=['Rx'], sig=0.7, offset=5, threshold=2.0):
    # choices must come from stimdic
    stimdic = {"B": [0, 0], "F": [0, 1], "RR": [0, 2], "LL": [0, 3],
               "RR": [1, 0], "Rx": [1, 1], "xR": [1, 2],
               "LL": [2, 0], "Lx": [2, 1], "xL": [2, 2],
               "D": [3, 0], "C": [3, 1]}

    _paths = utils.pathSorter(_path)

    try:
        factors, loadings, x = utils.load_eigen(_paths['output']['eigenvalues'])
    except KeyError:
        print('please run eigenValues first')

    try:
        stim_df = utils.load_stimuli(_paths['stimuli']['frame_aligned'])
    except KeyError:
        print('please align stimuli first')
        return

    try:
        ops, iscell, stats, f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    except KeyError:
        print('please run neuron extraction first')
        return
    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    selection = []
    for nrn in range(f_cells[iscell].shape[0]):
        s = plot_df.iloc[0].start
        e = plot_df.iloc[0].stop

        neuron = f_cells[iscell][nrn]

        if np.mean(neuron[s + offset:e + offset]) / np.mean(neuron[s - 2 * offset:s]) >= threshold:
            selection.append(nrn)
    neurons = []
    for i in selection:
        neurons.append((f_cells[iscell][i] - np.mean(f_cells[iscell][i][:15])) / np.mean(f_cells[iscell][i][:15]))

    neurons = np.array(neurons)

    neurons = gaussian_filter(neurons, sigma=sig)

    fig, ax = plt.subplots(4, 4, figsize=(16, 16), sharey=True)
    for i in range(stim_df.stimulus.nunique()):
        stims = [stim_df.stimulus.unique()[i]]
        plot_df = stim_df[stim_df.stimulus.isin(stims)]

        ind1 = stimdic[stims[0]][0]
        ind2 = stimdic[stims[0]][1]

        for ind in range(len(plot_df)):
            ax[ind1, ind2].plot(
                np.mean(neurons[:, plot_df.start.values[ind] - offset:plot_df.stop.values[ind] + offset], axis=0),
                label="trial" + str(ind))

        ax[ind1, ind2].set_title("stimulus: " + stims[0])
        ax[ind1, ind2].axvspan(offset,
                               offset + abs(plot_df.start.values[ind] - offset - plot_df.stop.values[ind] + offset),
                               color='red', alpha=0.2)

    plt.show()
    return


def plot_avg_stimulus_neurons(_path, stim=['Rx'], sig=0.7, offset=5, threshold=2.0):
    # choices must come from stimdic
    stimdic = {"B": [0, 0], "F": [0, 1], "RR": [0, 2], "LL": [0, 3],
               "RR": [1, 0], "Rx": [1, 1], "xR": [1, 2],
               "LL": [2, 0], "Lx": [2, 1], "xL": [2, 2],
               "D": [3, 0], "C": [3, 1]}

    paths = utils.pathSorter(_path)

    factors, loadings, x = utils.load_eigen(paths['output']['eigenvalues'])
    stim_df = utils.load_stimuli(paths['stimuli']['frame_aligned'])
    ops, iscell, stats, f_cells = utils.load_suite2p(paths['output']['suite2p'])

    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    plot_df = stim_df[stim_df.stimulus.isin(stim)]

    selection = []
    for nrn in range(f_cells[iscell].shape[0]):
        s = plot_df.iloc[0].start
        e = plot_df.iloc[0].stop

        neuron = f_cells[iscell][nrn]

        if np.mean(neuron[s + offset:e + offset]) / np.mean(neuron[s - 2 * offset:s]) >= threshold:
            selection.append(nrn)
    neurons = []
    for i in selection:
        neurons.append((f_cells[iscell][i] - np.mean(f_cells[iscell][i][:15])) / np.mean(f_cells[iscell][i][:15]))

    neurons = np.array(neurons)

    neurons = gaussian_filter(neurons, sigma=sig)

    fig, ax = plt.subplots(neurons.shape[0], figsize=(neurons.shape[0], neurons.shape[0]))

    for i in range(neurons.shape[0]):
        ax[i].plot(neurons[i])

        for x in range(len(plot_df)):
            ax[i].axvspan(plot_df.start.values[x] + offset, plot_df.stop.values[x] + offset, color='red', alpha=0.2)

    plt.show()
    return


def binocMap(frameAlignedStims):
    stims = utils.map_stimuli(frameAlignedStims)
    b = {'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'}
    stims.loc[:, 'stimulus_name'] = stims.stimulus.map(b)
    return stims


def pixelwise(base_path, stimOffset=5, local_bright=1.5):
    try:
        imgpath = utils.pathSorter(base_path)['image']['move_corrected']
    except:
        print('no move corrected image found, defaulting to raw')
        imgpath = utils.pathSorter(base_path)['image']['raw']

    frameAlignedStimsPath = utils.pathSorter(base_path)['stimuli']['frame_aligned']

    img = cm.load(imgpath)
    frameAlignedStims = pd.read_hdf(frameAlignedStimsPath)

    try:
        if 'stim_name' in frameAlignedStims.columns:
            frameAlignedStims.rename(columns={'stim_name' : 'stimulus_name'}, inplace=True)
        frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values
    except AttributeError:
        frameAlignedStims = binocMap(frameAlignedStims)
        frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values

    a = frameAlignedStims

    # a = frameAlignedStims[(frameAlignedStims.velocity != 0)]
    try:
        staticIndices = a[(a.velocity_0 == 0) & (a.velocity_1 == 0)].index
    except AttributeError:
        staticIndices = a[(a.velocity == 0)].index

    stim_df = utils.stimStartStop(a.drop(staticIndices))
    stim_df = stim_df[~stim_df.stimulus.isna()]

    statInds = []
    for i in frameAlignedStims.loc[staticIndices].img_stacks.values:
        for j in i:
            statInds.append(j)

    bg_image = np.nanmean(img[statInds], axis=0)

    frames, x, y = img.shape

    all_img = []
    for stim in stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique():
        _frames = utils.arrangedArrays(stim_df[stim_df.stimulus == stim], offset=stimOffset)

        try:
            stimImage = np.nanmean(img[_frames], axis=0) - bg_image
        except IndexError:
            stimImage = np.nanmean(img[-10:], axis=0) - bg_image

        stimImage[stimImage < 0] = 0

        rgb = np.zeros((3, x, y), 'float64')

        rgb[0, :, :] = stimImage * eva_weightings[stim][0]
        rgb[1, :, :] = stimImage * eva_weightings[stim][1]
        rgb[2, :, :] = stimImage * eva_weightings[stim][2]

        r = rgb[0, :, :]
        g = rgb[1, :, :]
        b = rgb[2, :, :]

        r = r - r.min()
        b = b - b.min()
        g = g - g.min()
        mymax = np.max(np.dstack((r, g, b)))
        all_img.append(np.dstack((r ** local_bright, g ** local_bright, b ** local_bright)))

    _all_img = []
    for img in all_img:
        _all_img.append(img / np.max(all_img))

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)

    return fin_img, _all_img


def weighted_mean_angle(degs, weights):
    _sums = []
    for d in range(len(degs)):
        _sums.append(weights[d]*rect(1, radians(degs[d])))
    return degrees(phase(sum(_sums)/np.sum(weights)))


def pixelwise2(base_path, stimOffset=5):
    try:
        imgpath = utils.pathSorter(base_path)['image']['move_corrected']
    except:
        print('no move corrected image found, defaulting to raw')
        imgpath = utils.pathSorter(base_path)['image']['raw']

    frameAlignedStimsPath = utils.pathSorter(base_path)['stimuli']['frame_aligned']

    img = cm.load(imgpath)
    frameAlignedStims = pd.read_hdf(frameAlignedStimsPath)

    frameAlignedStims = binocMap(frameAlignedStims)
    frameAlignedStims.loc[:, 'stimulus'] = frameAlignedStims.stimulus_name.values

    stim_df = utils.stimStartStop(frameAlignedStims)

    statInds = utils.arrangedArrays(stim_df[stim_df.stimulus.isna()])
    try:
        bg_image = np.nanmean(img[statInds], axis=0)
    except IndexError:
        bg_image = np.nanmean(img[statInds[:-15]], axis=0)

    frames, x, y = img.shape
    stim_df.stimulus = stim_df.stimulus.map({'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'})

    all_img = []
    for stim in stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique():
        _frames = utils.arrangedArrays(stim_df[stim_df.stimulus == stim], offset=stimOffset)

        try:
            stimImage = np.nanmean(img[_frames], axis=0) - bg_image
        except IndexError:
            stimImage = np.nanmean(img[_frames[:-5]], axis=0) - bg_image

        stimImage[stimImage < 0] = 0

        rgb = np.zeros((3, x, y), 'float64')

        rgb[0, :, :] = stimImage * eva_weightings[stim][0]
        rgb[1, :, :] = stimImage * eva_weightings[stim][1]
        rgb[2, :, :] = stimImage * eva_weightings[stim][2]

        r = rgb[0, :, :]
        g = rgb[1, :, :]
        b = rgb[2, :, :]

        r = r - r.min()
        b = b - b.min()
        g = g - g.min()
        mymax = np.max(np.dstack((r, g, b)))
        all_img.append(np.dstack((r ** 1.5, g ** 1.5, b ** 1.5)))

    _all_img = []
    for img in all_img:
        _all_img.append(img / np.max(all_img))

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)

    return fin_img, _all_img


def pixelwisevolumes(planePath):
    paths = utils.pathSorter(planePath)
    image = cm.load(paths['image']['raw'])

    frameAlignedStims = pd.read_hdf(paths['stimuli']['frame_aligned'])
    stim_df = utils.stimStartStop(frameAlignedStims)

    statics = []
    for q in range(len(stim_df)):
        data = stim_df.iloc[q]
        inds = np.arange(data.start, data.stop)
        statics.append(inds[:len(inds) // 2])
    staticIndices = np.concatenate(statics)
    bg_img = np.mean(image, axis=0)

    all_img = []
    frames, x, y = image.shape

    for stim in stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique():
        _frames = utils.last_n_arrayd(stim_df[stim_df.stimulus == stim])

        try:
            stimImage = np.mean(image[_frames], axis=0) - bg_img
        except IndexError:
            stimImage = np.mean(image[_frames[:-5]], axis=0) - bg_img

        stimImage[stimImage < 0] = 0

        rgb = np.zeros((3, x, y), 'float64')

        rgb[0, :, :] = stimImage * eva_weightings[stim][0]
        rgb[1, :, :] = stimImage * eva_weightings[stim][1]
        rgb[2, :, :] = stimImage * eva_weightings[stim][2]

        r = rgb[0, :, :]
        g = rgb[1, :, :]
        b = rgb[2, :, :]

        r = r - r.min()
        b = b - b.min()
        g = g - g.min()
        mymax = np.max(np.dstack((r, g, b)))
        all_img.append(np.dstack((r ** 1.5, g ** 1.5, b ** 1.5)))

    _all_img = []
    for img in all_img:
        _all_img.append(img / np.max(all_img))

    fin_img = np.sum(_all_img, axis=0)
    fin_img /= np.max(fin_img)
    return fin_img, _all_img, stim_df[stim_df.stimulus.isin(eva_weightings.keys())].stimulus.unique()

def spatial_neurons(base_path, x=[0,9999], y=[0,9999]):
    ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(base_path)['output']['suite2p'])
    i = 0
    xs = []
    ys = []
    nrns = np.arange(0, len(stats[iscell]))
    for i in range(len(stats[iscell])):
        ys.append(np.mean(stats[iscell][i]['ypix']))
        xs.append(np.mean(stats[iscell][i]['xpix']))

    locs = pd.DataFrame({'neuron' : nrns, 'x' : xs, 'y' : ys})
    return locs[(locs.x>=x[0])&(locs.x<=x[1])&((locs.y>=y[0]))&((locs.y<=y[1]))]


def stimResponses(cells, stims, threshold=0.5, offset=10):

    stimuli = []
    neurons = []
    finVals = []
    finVals_std = []

    for w in range(len(cells)):
        x = cells[w]

        for s in stims.stimulus.unique():
            a = stims[stims.stimulus == s]
            nrnRespones = []

            for n in range(len(a)):
                _i = np.arange(a.iloc[n].start + offset, a.iloc[n].stop + offset)
                try:
                    maxInd = np.nanargmax(x[_i])
                    nrnRespones.append(x[_i][maxInd])
                except:
                    nrnRespones.append(0)

            finVals.append(np.nanmean(nrnRespones))
            finVals_std.append(np.nanstd(nrnRespones))

            stimuli.append(s)
            neurons.append(w)

    celldf = pd.DataFrame({'neuron': neurons, 'stimulus': stimuli, 'val': finVals, 'error': finVals_std})

    aboveThreshold = celldf[celldf.val >= threshold]
    return aboveThreshold


def neuronColor(basePath, threshold=1.5, offset=5, offset2=5, figshape=(12,12), plot=False, returnTuneds=False, loc=None, z_scored=False, volume=False):
    if volume is not False:
        from pathlib import Path
        ops, iscell, stats, f_cells = utils.load_suite2p(Path(basePath).joinpath('suite2p/plane0'))
        stim_df, cells = utils.stim_cell_returner(basePath, s2p=[ops, iscell, stats, f_cells])

    else:
        ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(basePath)['output']['suite2p'])
        stim_df, cells = utils.stim_cell_returner(basePath)

    if z_scored:
        cells = scipy.stats.zscore(cells)

    if loc is not None:
        loc_cells = spatial_neurons(basePath, loc[0], loc[1])
    else:
        loc_cells = pd.DataFrame({'neuron' : np.arange(0,len(cells))})

    monocStims = ['Backward', 'Forward', 'BackwardLeft', 'ForwardRight', 'ForwardLeft', 'BackwardRight', 'Right',
                  'Left']
    if 'left' in stim_df.stimulus.unique():
        stim_df.stimulus = stim_df.stimulus.map({'left':'Left', 'right':'Right', 'forward':'Forward', 'backward':'Backward'})
    mstim_df = stim_df[stim_df.stimulus.isin(monocStims)]
    if len(mstim_df) == 0:
        stim_df.stimulus = stim_df.stimulus.map({'F': 'Forward', 'RR': 'Right', 'LL': 'Left', 'B': 'Backward'})
        mstim_df = stim_df[stim_df.stimulus.isin(monocStims)]

    stimuli = []
    neurons = []
    finVals = []
    finVals_std = []
    for w in loc_cells.neuron.unique():
        x = cells[w]

        for s in mstim_df.stimulus.unique():
            a = mstim_df[mstim_df.stimulus == s]
            nrnRespones = []

            for n in range(len(a)):
                _i = np.arange(a.iloc[n].start + offset, a.iloc[n].stop + offset2)
                try:
                    maxInd = np.nanargmax(x[_i])
                    nrnRespones.append(x[_i][maxInd])
                except:
                    nrnRespones.append(0)

            finVals.append(np.nanmean(nrnRespones))
            finVals_std.append(np.nanstd(nrnRespones))

            stimuli.append(s)
            neurons.append(w)


    celldf = pd.DataFrame({'neuron': neurons, 'stimulus': stimuli, 'val': finVals, 'error': finVals_std})

    tunedCells = celldf[celldf.val>=threshold]
    if returnTuneds:
        return tunedCells, celldf, iscell, stats, f_cells, mstim_df, stim_df

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in tunedCells.neuron.unique():
        ypix = stats[iscell][cell]['ypix']
        xpix = stats[iscell][cell]['xpix']

        for c in range(cell_img.shape[2]):

            _df = celldf[celldf.neuron == cell]
            _df.val = _df.val / _df.val.sum()
            for _s in _df.stimulus.unique():
                _val = _df[_df.stimulus == _s].val.values

                cell_img[ypix, xpix, c] += eva_weightings[_s][c] * _val
    if plot:
        plt.figure(figsize=figshape)
        plt.imshow(cell_img)

        _cmap = eva_weightings.values()

        fig, ax = plt.subplots(figsize=(figshape[0], 1))
        fig.subplots_adjust(bottom=0.5)

        cmap = mpl.colors.ListedColormap(_cmap)
        cmap.set_over('0.25')
        cmap.set_under('0.75')

        bounds = list(np.arange(cmap.N))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                        norm=norm,
                                        ticks=bounds,
                                        spacing='proportional',
                                        orientation='horizontal')

        cb2.ax.set_xticklabels(eva_weightings.keys())

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.show()
    return cell_img, ops, celldf


def topneuroncolor(basePath, threshold=0.5, offset=5, z_scored=False, return_stuff = False):
    ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(basePath)['output']['suite2p'])
    stim_df, cells = utils.stim_cell_returner(basePath)

    tunedCells, celldf, iscell, stats, f_cells, mstim_df, stim_df = neuronColor(basePath, threshold=threshold,
                                                                                          offset=offset, plot=False,
                                                                                          z_scored=z_scored,
                                                                                          returnTuneds=True)
    topdic = {}
    for nrn in tunedCells.neuron.unique():
        pref = tunedCells[tunedCells.neuron == nrn].sort_values(by='val', ascending=False).stimulus.values[0]
        topdic[nrn] = eva_weightings[pref]

    tunedCells.loc[:, 'color'] = tunedCells.neuron.map(topdic)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for cell in tunedCells.neuron.unique():
        ypix = stats[iscell][cell]['ypix']
        xpix = stats[iscell][cell]['xpix']

        for c in range(cell_img.shape[2]):

            _df = tunedCells[tunedCells.neuron == cell]
            color = _df.color.values[0]
            cell_img[ypix, xpix, c] = color[c]

    if return_stuff:
        return cell_img, tunedCells, cells
    else:
        return cell_img


def plot_cell(_path, plottedCells, rot=-1, invert=False, annotating=False, use_iscell=True, title=None, pretty=False,
              fillval=-1, legend=False):
    # plots all cells included in a list

    _paths = utils.pathSorter(_path)
    ops, iscell, stats, _f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    if use_iscell:
        stats = stats[iscell]
    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    cdict = {}
    z = 1
    for cell in plottedCells:
        ypix = stats[cell]['ypix']
        xpix = stats[cell]['xpix']
        if not pretty:
            cell_img[ypix, xpix] = 1
        if pretty:
            import cv2
            mean_y = int(np.mean(ypix))
            mean_x = int(np.mean(xpix))
            colorval = z
            cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, fillval)
            cdict[cell] = colorval
            z += 1

    masked = np.ma.masked_where(cell_img == 0, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(np.rot90(ops['refImg'], rot), cmap=mpl.cm.gray)
    ax.imshow(np.rot90(masked, rot), cmap=mpl.cm.gist_rainbow, interpolation=None, alpha=1, vmax=np.max(masked), vmin=0)
    if invert:
        ax.invert_xaxis()
    if title is not None:
        ax.set_title(title)
    if annotating:
        for cell in plottedCells:
            plt.annotate(cell, (cell_img.shape[0] - np.mean(stats[cell]['ypix']), np.mean(stats[cell]['xpix'])), color='white', weight='bold', size=12)
    if legend:
        used_colors = [clr[0] for clr in mpl.cm.gist_rainbow(list(cdict.items()))]
        legend_lines = [Line2D([0], [0], color=clr) for clr in used_colors]
        legend_labels = [str(label) for label in cdict.keys()]
        ax.legend(legend_lines, legend_labels, loc='right', bbox_to_anchor=[1.175, 0.5], title='cell identities')
    plt.show()


def pltcell(ops, stats, nrn, pretty=False):

    if pretty:
        import cv2
    plt.figure(figsize=(6,6))
    ref_img = ops['refImg']

    plt.imshow(ref_img, cmap='gray')
    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    if hasattr(nrn, '__iter__'):
        for _, n in enumerate(nrn):
            ypix = stats[n]['ypix']
            xpix = stats[n]['xpix']
            if pretty:
                mean_y = int(np.mean(ypix))
                mean_x = int(np.mean(xpix))
                colorval = _ + 1
                cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, -1)
            else:
                cell_img[ypix, xpix] = _ + 1
    else:
        ypix = stats[nrn]['ypix']
        xpix = stats[nrn]['xpix']

        if pretty:
            mean_y = int(np.mean(ypix))
            mean_x = int(np.mean(xpix))
            colorval = 1
            cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, -1)
        else:
            cell_img[ypix, xpix] = 1

    masked = np.ma.masked_where(cell_img < 0.9, cell_img)

    plt.imshow(masked, cmap='cool')
    plt.show()


def plot_factor_corr(_path, fullCorr, minThresh=0.2):
    _paths = utils.pathSorter(_path)
    ops, iscell, _stats, _f_cells = utils.load_suite2p(_paths['output']['suite2p'])
    stats = _stats[iscell]
    f_cells = _f_cells[iscell]

    cell_img = np.zeros((ops['Ly'], ops['Lx']))

    for nrn in range(len(fullCorr)):

        corrval = fullCorr[nrn]

        if abs(corrval) >= minThresh:
            ypix = stats[nrn]['ypix']
            xpix = stats[nrn]['xpix']
            cell_img[ypix, xpix] = corrval

    masked = np.ma.masked_where(cell_img == 0, cell_img)

    fig, ax = plt.subplots(figsize=(10, 10))
    #  a background image of an average of the entire stack
    ax.imshow(ops['refImg'], cmap=mpl.cm.gray)
    ax.imshow(masked, cmap=mpl.cm.bwr, interpolation=None, alpha=1)

    fig, ax = plt.subplots(figsize=(10, 1))
    fig.subplots_adjust(bottom=0.5)

    cmap = mpl.cm.bwr
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    cb1.set_label('Correlation')
    fig.show()


def active_passive_plot(basePath, stim='forward', sig=3, abr_cutoff=75):

    stim_df, cells = utils.stim_cell_returner(basePath)
    clrstims = utils.colormapStimuli(stim_df)

    imgpath = utils.pathSorter(basePath)['image']['move_corrected']
    img = cm.load(imgpath)

    tail_data, full_tail = utils.tail_alignment(basePath)

    times = tail_data['t'].values
    smoothedVals = scipy.ndimage.gaussian_filter(tail_data["/'TailLoc'/'TailCurvature'"].values, sigma=sig)

    rollingVals = pd.Series(full_tail["/'TailLoc'/'TailCurvature'"].values).rolling(1000).median()[tail_data.index]
    diffVals = smoothedVals - rollingVals

    aligned_stims = pd.read_hdf(utils.pathSorter(basePath)['stimuli']['frame_aligned'])

    try:
        staticIndices = aligned_stims[(aligned_stims.velocity_0 == 0) & (aligned_stims.velocity_1 == 0)].index
    except:
        staticIndices = aligned_stims[aligned_stims.velocity == 0].index

    statInds = []
    for j in aligned_stims[(aligned_stims.velocity_0 == 0) & (aligned_stims.velocity_1 == 0)].img_stacks:
        for i in j:
            statInds.append(i)

    movingStims = aligned_stims.drop(staticIndices)
    median_img = np.median(img[statInds], axis=0)

    b_stims = stim_df[stim_df.stimulus==stim]

    active_trials = []
    passive_trials = []


    for n in range(len(b_stims)):
        n_inds = tail_data[tail_data.frame.isin(np.arange(b_stims.iloc[n].start - 1, b_stims.iloc[n].stop))].index

        if np.max(diffVals[n_inds]) >= abr_cutoff:
            active_trials.append(n)
        else:
            passive_trials.append(n)

    active_inds = utils.arrangedArrays(b_stims.iloc[active_trials], offset=0)
    try:
        active_img = np.mean(img[active_inds], axis=0)
    except IndexError:
        print('no active trials')
        active_img = np.zeros(median_img.shape)

    passive_inds = utils.arrangedArrays(b_stims.iloc[passive_trials], offset=0)
    passive_img = np.mean(img[passive_inds], axis=0)

    print(f'{len(active_trials)} actives    {len(passive_trials)} passives')

    fig, ax = plt.subplots(1, 4, figsize=(16, 8))

    ax[0].imshow(median_img, cmap='gray', vmin=0, vmax=100)
    ax[0].set_title('median image')

    ax[1].imshow(active_img - median_img, cmap='gray', vmin=0, vmax=100)
    ax[1].set_title('active')

    ax[2].imshow(passive_img - median_img, cmap='gray', vmin=0, vmax=100)
    ax[2].set_title('passive')

    im = ax[3].imshow(active_img - passive_img, cmap='inferno', vmax=100)
    ax[3].set_title('difference')

    fig.colorbar(im, ax=ax[3], shrink=0.4)

    for a in ax:
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    fig.suptitle(f'brain responses to : {stim}')
    plt.tight_layout()
    plt.show()


def monocplot(planepath, stimchoice, colormap, start_offset=45, background_percentile=0.3, tuned_thresh=1.5, suite2p=False):
    from pathlib import Path

    if not suite2p:
        ops, iscell, stats, f_cells = utils.load_suite2p(Path(planepath).joinpath('suite2p/plane0'))
    else:
        ops, iscell, stats, f_cells = suite2p
    plane_num = f_cells.shape[1]
    stims = pd.read_hdf(utils.pathSorter(planepath)['stimuli']['frame_aligned'])

    stim_df = utils.stimStartStop(stims)
    stim_df = stim_df.loc[1:]
    stim_df.loc[:, "start"] += start_offset

    colorstims = stim_df[stim_df.stimulus.isin(stimchoice)]
    colorstims.loc[:, 'color'] = colorstims.stimulus.map(colormap)

    cell_percentiles = np.nanquantile(f_cells, background_percentile, axis=1, keepdims=True)
    fdff_cells = f_cells / cell_percentiles

    stimResponses = []

    for stim in stimchoice:
        s = stim_df[stim_df.stimulus.isin([stim])]

        stim_inds = np.clip(utils.arrangedArrays(s), a_min=0, a_max=plane_num - 1)
        stimResponse = np.mean(fdff_cells[:, stim_inds], axis=1)
        stimResponses.append(stimResponse)

    nrns = np.unique(np.array(sorted(np.where(np.array(stimResponses) >= tuned_thresh)[1])))
    chosen_neurons = fdff_cells[nrns, :]

    _stimchoice = []
    stimval = []

    for n in range(len(nrns)):
        celltrace = fdff_cells[nrns[n], :]

        vals = []
        for stim in stimchoice:
            s = stim_df[stim_df.stimulus.isin([stim])]
            stim_inds = np.clip(utils.arrangedArrays(s), a_min=0, a_max=plane_num-1)
            stimResponse = np.mean(celltrace[stim_inds])
            vals.append(stimResponse)
        _stimchoice.append(np.where(vals==np.max(vals))[0][0])
        stimval.append(np.max(vals))
    stimval /= np.max(stimval)

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))

    for n in range(len(nrns)):
        x = stats[nrns[n]]['xpix']
        y = stats[nrns[n]]['ypix']
        cell_img[x, y, :] = np.array(colormap[stimchoice[_stimchoice[n]]]) * stimval[n]

    masked = np.ma.masked_where(cell_img == 0, cell_img)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.imshow(ops['refImg'], cmap=mpl.cm.gray)
    ax.imshow(masked,  alpha=0.65)

    plt.show()
    return


def radial_nrn_plot(input_path, uangles, uvels, ufreqs, threshold=20, offset=12, save_path=None, palette=None, plot=True,):
    '''

    :param input_path: path to folder of data
    :param uangles: list of the unique angles in the data
    :param uvels: list of the unique velocities in the data
    :param ufreqs: list of the unique frequencies in the data
    :param threshold: activity threshold for neurons in the data to be included in plot (is avgd across trials)
    :param offset: frame offset for each stimuli
    :return:
    '''
    stims = utils.load_stimmed(input_path)
    stims = stims.iloc[:-1]

    paths = utils.pathSorter(input_path)
    ops, iscell, stats, fcells = utils.load_suite2p(paths['output']['suite2p'])
    # corr_img = cm.load(paths['image']['move_corrected'])

    fcells = fcells[iscell]

    stim_starts = stims.start_frame.values

    stim_responses = []
    for stim in stim_starts:
        bg_inds = np.arange(stim - 2 * offset, stim, dtype=np.int32)
        bg_resp = np.nanmean(fcells[:, bg_inds], axis=1)

        stim_inds = np.arange(stim + offset, stim + 2 * offset, dtype=np.int32)
        stim_resp = np.nanmean(fcells[:, stim_inds], axis=1)

        stim_diff = stim_resp - bg_resp

        stim_responses.append(stim_diff)

    stim_responses = np.array(stim_responses)

    stimDict = {}

    for ang in stims.angle.unique():
        stim_arr = stim_responses[stims[stims.angle == ang].index, :]
        nrn_responses = np.mean(stim_arr, axis=0)
        stimDict[ang] = nrn_responses

    for vels in stims.vel.unique():
        stim_arr = stim_responses[stims[stims.vel == vels].index, :]
        nrn_responses = np.mean(stim_arr, axis=0)
        stimDict[vels] = nrn_responses

    for freqs in stims.freq.unique():
        stim_arr = stim_responses[stims[stims.freq == freqs].index, :]
        nrn_responses = np.mean(stim_arr, axis=0)
        stimDict[freqs] = nrn_responses
    df = pd.DataFrame(stimDict)

    a = [np.array(df[df[angle]>=threshold].index) for angle in uangles]
    nrns = np.unique(sorted(np.concatenate(a)))
    vels = df.loc[nrns, uvels].idxmax(axis=1)
    dirs = df.loc[nrns, uangles].idxmax(axis=1)
    freqs = df.loc[nrns, ufreqs].idxmax(axis=1)
    maxdf = pd.DataFrame({'neuron': nrns, 'direction': dirs, 'velocity': vels, 'frequency': freqs})
    maxdf.direction = [radians(x) for x in maxdf.direction]
    if plot:
        # seaborn style
        # g = sns.FacetGrid(maxdf, height=8, subplot_kws=dict(projection='polar'), despine=False,)
        # g.map_dataframe(sns.swarmplot, x='direction', y='velocity', hue='frequency', orient='h', size=7, palette='Set2', dodge=True)
        # ax = plt.gca()
        # ax.set_theta_zero_location('N')
        # ax.set_theta_direction(-1)
        # ax.invert_yaxis()
        # g.add_legend()
        sns.set_theme()

        n_colors = maxdf.frequency.nunique()
        # palette = sns.color_palette(n_colors=n_colors)
        if not palette:
            palette = ['#F72A74', '#D424CE', '#B134EB', '#6124D4']
        colordic = {}
        for n, freq in enumerate(sorted(maxdf.frequency.unique())):
            colordic[freq] = palette[n]
        maxdf.loc[:, 'color'] = maxdf.frequency.map(colordic)

        jitteryness = [1, 1] # X / Y
        # radial jitter
        maxdf.direction = maxdf.direction + [2.5 * jitteryness[0] * radians(x) for x in np.random.normal(0, 1, len(maxdf))]
        # value jitter
        maxdf.velocity = maxdf.velocity + np.random.normal(0, 1, len(maxdf)) / (400 * jitteryness[1])

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
        ax.scatter(maxdf.direction, maxdf.velocity, color=maxdf.color, alpha=0.8)

        # make the plot look logical
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, np.max(uvels)*1.5)

        # custom legend
        legend_lines = [Line2D([0], [0], color=clr, marker='o') for clr in palette]
        legend_labels = [str(label) for label in colordic.keys()]
        ax.legend(legend_lines, legend_labels, loc='right', bbox_to_anchor=[1.3, 0.5], title='frequency')

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')

        plt.show()
    return maxdf


def resp_plot(base_path, palette, offset=12, nrn=0):
    paths = utils.pathSorter(base_path)

    ops, iscell, stats, fcells = utils.load_suite2p(paths['output']['suite2p'])
    fcells = fcells[iscell]

    stims = utils.load_stimmed(base_path)
    stims = stims.iloc[:-1]

    bg_responses = np.mean(fcells[:, 0:50], axis=1)
    stim_starts = stims.start_frame.values

    stim_responses = []
    for stim in stim_starts:
        stim_inds = np.arange(stim + offset, stim + 2 * offset, dtype=np.int32)
        stim_resp = np.nanmean(fcells[:, stim_inds], axis=1)

        stim_diff = stim_resp - bg_responses

        stim_responses.append(stim_diff)

    stim_responses = np.array(stim_responses)


    d = stims.sort_values(by='angle')
    plt.scatter(np.arange(0,len(stim_responses[:, nrn])), stim_responses[:, nrn][d.index.values])
    for n, ang in enumerate(d.angle.values):
        plt.axvline(x=n, color=palette[ang], alpha=0.5)

    legend_lines = [Line2D([0], [0], color=palette[clr], marker='o') for clr in palette.keys()]
    legend_labels = [str(label) for label in palette.keys()]
    plt.legend(legend_lines, legend_labels, loc='right', bbox_to_anchor=[1.3, 0.5], title='direction')
    plt.show()


def response_hists(planePath, returns=False):
    import tifffile
    s2p_path = planePath.joinpath('suite2p\plane0')
    stimuli_path = utils.pathSorter(planePath)['stimuli']['frame_aligned']
    image_path = utils.pathSorter(planePath)['image']['raw']

    image = tifffile.imread(image_path)
    stimuli = pd.read_hdf(stimuli_path)
    ops, iscell, stats, f_cells = utils.load_suite2p(s2p_path)

    # put f_cells in 0-1 range
    norm_f_cells = []
    for i in range(f_cells.shape[0]):
        norm_f_cells.append(f_cells[i]/np.max(f_cells[i]))
    norm_f_cells = np.array(norm_f_cells)

    offset_val = 5
    ## stim start is [-5] of frame indicies
    ## stim end is [-1] + 5 of frame indicies
    try:
        stim_starts = [stimuli.img_stacks.values[i][-offset_val] for i in range(stimuli.img_stacks.values.shape[0])]
    except IndexError:
        stim_starts = [stimuli.img_stacks.values[i][-3] for i in range(stimuli.img_stacks.values.shape[0])]

    stim_stops = [stimuli.img_stacks.values[i][-1] + offset_val for i in range(stimuli.img_stacks.values.shape[0])]
    stim_arrs = [np.clip(np.arange(stim_starts[i], stim_stops[i]), a_min=0, a_max=image.shape[0]-1) for i in range(len(stim_starts))]
    bg_arrs = [np.clip(np.arange(stim_starts[i]-(2*offset_val), stim_starts[i]-(offset_val//2)), a_min=0, a_max=image.shape[0]-1) for i in range(len(stim_starts))]
    stimuli.loc[:, 'frames'] = stim_arrs
    stimuli.loc[:, 'bg_frames'] = bg_arrs

    cell_responses = {}
    background = np.mean(norm_f_cells[:, np.concatenate(stimuli.bg_frames.values)], axis=1)
    for stim in stimuli.stim_name.unique():
        stim_indices = np.concatenate(stimuli[stimuli.stim_name==stim].frames.values)
        cell_responses[stim] = np.clip(np.mean(norm_f_cells[:, stim_indices], axis=1) - background , a_min=0, a_max=1)

    response_df = pd.DataFrame(cell_responses)
    if returns:
        return response_df

    plt.figure(figsize=(10,10))
    for col in response_df.columns:
        bins = np.arange(0,1,1/100)
        hist, bins = np.histogram(response_df.loc[:,col], bins=bins)
        plt.plot(bins[1:], hist, label=col, c=eva_weightings[col])
        plt.legend()

def return_corr(data):
    lefties = data.loc[:, ['lateral_left', 'left', 'medial_left']].mean(axis=1)
    righties = data.loc[:, ['lateral_right', 'right', 'medial_right']].mean(axis=1)

    data.loc[:, 'lefties'] = lefties
    data.loc[:, 'righties'] = righties

    datum = data[['lefties', 'righties', 'forward', 'backward']]
    corr = datum.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    return corr, mask


def draw_corr_picture(corrs, neuron, neurons, ops, stats):
    from colour import Color

    c2 = '#F7343A'
    c_base = '#000000'

    clrs = list(Color(c_base).range_to(Color(c2), 100))
    clrs_pos = [i.rgb for i in clrs]

    cell_img = np.zeros((ops['Ly'], ops['Lx'], 3))
    for n, cor in enumerate(corrs[np.where(neurons==neuron)[0][0]]):
        ypix = stats[neurons[n]]['ypix']
        xpix = stats[neurons[n]]['xpix']
        choiceColor = clrs_pos[np.clip(int(50+cor*50), a_min=-99, a_max=99)]
        for _, c in enumerate(choiceColor):
            cell_img[ypix, xpix, _] = c
    return cell_img

def create_myDictionary(stimuli, theBestStimFdff):
    myDictionary = {}
    for stimmy in stimuli.stim_name.unique():
        stimInds = stimuli[stimuli.stim_name == stimmy].index.values

        stubby = theBestStimFdff[(theBestStimFdff.stimulus.isin(stimInds))]
        cellMean, cellStd = tolerant_mean(stubby.fdff)

        myDictionary[stimmy] = [cellMean, cellStd/np.sqrt(len(stubby))]
    return myDictionary

def graph_legend(stimuli, bbox=[1.2,0.5],lstyle=None):
    legend_lines = [Line2D([0], [0], color=clr, marker='o', linestyle=lstyle) for k,clr in eva_weightings.items() if k in stimuli.stim_name.unique()]
    legend_labels = [str(k) for k,clr in eva_weightings.items() if k in stimuli.stim_name.unique()]
    plt.legend(legend_lines, legend_labels, loc='right', bbox_to_anchor=bbox, title='stimulus');

def graph_label(spanner, stimuli):
    baseline_period = np.min([stimuli.start.values[i] - stimuli.stop.values[i-1] for i in range(len(stimuli)) if i > 0]) - 2 # use two frame min buffer

    for stim_n in spanner:
        plt.axvspan(stimuli.iloc[stim_n].img_stacks.values[0], stimuli.iloc[stim_n].img_stacks.values[-1], color = eva_weightings[stimuli.iloc[stim_n].stim_name], alpha=0.1)
        plt.axvspan(stimuli.iloc[stim_n].img_stacks.values[-6], stimuli.iloc[stim_n].img_stacks.values[-1], color = eva_weightings[stimuli.iloc[stim_n].stim_name], alpha=0.7)

    plt.axvspan(stimuli.iloc[stim_n+1].img_stacks.values[0], stimuli.iloc[stim_n+1].img_stacks.values[-1], color = eva_weightings[stimuli.iloc[stim_n+1].stim_name], alpha=0.1)
    plt.xlim(stimuli.iloc[spanner[0]].img_stacks.values[0], stimuli.iloc[spanner[-1]].img_stacks.values[-1]+(1.5*baseline_period));


def export_legend(legend, filename="legend.png"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def neuron_blast(stimuli, stim_fdff, chosen_neuron, highlights=[0,15,20], redlining=True):
    subplot_locations = {'lateral_left' : [0,3], 'lateral_right': [1,3], 'medial_left' : [0,2], 'medial_right' : [1,2],
                         'left' : [0,1], 'right' : [1,1], 'forward' : [0,0], 'backward' : [1,0]}

    if hasattr(chosen_neuron, '__iter__'):
        myDictionary = create_myDictionary(stimuli, stim_fdff[stim_fdff.neuron.isin(chosen_neuron)])
    else:
        myDictionary = create_myDictionary(stimuli, stim_fdff[stim_fdff.neuron==chosen_neuron])
    fig, ax = plt.subplots(2,4, figsize=(16,6))
    for k,v in subplot_locations.items():
        ax[v[0], v[1]].set_title(k)

        yvals, err = myDictionary[k]
        xvals = np.arange(0, len(yvals))
        ax[v[0], v[1]].errorbar(x=xvals, y=yvals, yerr=err, color='black')
        ax[v[0], v[1]].axvspan(highlights[0], highlights[1] , alpha=0.1, color=eva_weightings[k])
        ax[v[0], v[1]].axvspan(highlights[1], highlights[2], alpha=0.6, color=eva_weightings[k])
        ax[v[0], v[1]].set_ylim(-.5,1.0)
        if redlining:
            ax[v[0], v[1]].axhline(0, color='red')
    fig.tight_layout()
    plt.show()

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

def barcoding(stimuli, thresh_df, myLowThresh=0.05, plot=False, plot_external=False, return_a=False):
    from itertools import combinations, chain

    bool_df = pd.DataFrame(thresh_df >= myLowThresh)
    cols = bool_df.columns.values
    raw_groupings = [cols[np.where(bool_df.iloc[i] == 1)] for i in range(len(bool_df))]
    groupings_df = pd.DataFrame(raw_groupings).T
    groupings_df.columns = bool_df.index

    nrows = (2**stimuli.stim_name.nunique())-1
    ncols = len(cols)

    all_combinations = list(chain(*[list(combinations(cols, i)) for i in range(ncols+1)]))
    temp = list(groupings_df.T.values)
    new_list = [tuple(filter(None, temp[i])) for i in range(len(temp))]

    setNeuronMappings = list(set(new_list))
    indexNeuronMappings = [setNeuronMappings.index(i) for i in new_list]
    setmapNeuronMappings = [setNeuronMappings[i] for i in indexNeuronMappings]
    allmapNeuronMappings = [all_combinations.index(i) for i in setmapNeuronMappings]

    a = pd.DataFrame(indexNeuronMappings)
    a.rename(columns={0 : 'nrn_ind'}, inplace=True)
    a.loc[:, 'set']  = setmapNeuronMappings
    a.loc[:, 'fullcomb']  = allmapNeuronMappings
    a.loc[:, 'neuron'] = groupings_df.columns.values

    sorted_a = a.sort_values(by='nrn_ind')
    if return_a:
        return sorted_a
    if plot_external or plot:

        image = np.zeros([ncols, nrows, 3])
        u_stims = stimuli.stim_name.unique()
        for q in range(len(u_stims)):
            for n, v in enumerate(all_combinations[1:]):
                if u_stims[q] in v:
                    color = eva_weightings[u_stims[q]]
                    for b, c in enumerate(color):
                        image[q,n,b] = c

        xvals = np.arange(0, len(all_combinations))
        heights = np.zeros(len(all_combinations))
        valCounts = pd.value_counts(sorted_a['fullcomb'], sort=False)

        j = valCounts.values

        for k, i in enumerate(valCounts.index):
            heights[i] = j[k]
        if plot:
            plt.figure(figsize=(50,10))
            plt.imshow(image, extent=[-0.5,255.5, -8,0])
            plt.bar(x = xvals ,height = heights)
            plt.show()
        if plot_external:
            return xvals, heights, image

    return sorted_a


def hori_plot(stimuli, stim_fdff, chosen_neuron, highlights=[0, 15, 20]):
    subplot_locations = {'lateral_left': 7, 'lateral_right': 2, 'medial_left': 6, 'medial_right': 3,
                         'left': 5, 'right': 4, 'forward': 1, 'backward': 0}
    if hasattr(chosen_neuron, '__iter__'):
        myDictionary = create_myDictionary(stimuli, stim_fdff[stim_fdff.neuron.isin(chosen_neuron)])
    else:
        myDictionary = create_myDictionary(stimuli, stim_fdff[stim_fdff.neuron == chosen_neuron])

    fig, ax = plt.subplots(1, 8, figsize=(32, 4))
    for k, v in subplot_locations.items():
        # ax[v].set_title(k)

        yvals, err = myDictionary[k]
        xvals = np.arange(0, len(yvals))
        ax[v].errorbar(x=xvals, y=yvals, yerr=err, color='black', linewidth=4)
        ax[v].axvspan(highlights[1], highlights[2], alpha=0.6, color=eva_weightings[k])
        ax[v].set_ylim(-.5, 1.0)
        ax[v].set_xlim(13, 25)
        ax[v].axhline(0, color='red')
        ax[v].set_xticks([])
        ax[v].set_yticks([])

    fig.tight_layout()
    # plt.savefig(Path(base_save).joinpath(f'{5}_ex_right_mr_barcode.pdf'))
    plt.show()


def return_n_colors(n, cmap):
    c_arr = np.linspace(0,1,n)
    return [cmap(c) for c in c_arr]


class ImagingDataVolumetric:
    def __init__(self, data_path, run=False):
        self.datapath = data_path

        self.plane_list = list(utils.pathSorter(self.datapath)['image']['volume'].values())
        self.datum_dict = {i: {} for i in self.plane_list}

        self.load_stimuli()
        self.load_s2p()

        if run:
            self.create_threshdf()
            self.create_fdff()
            self.barcode()

    def load_s2p(self):
        for plane in self.plane_list:
            ops, iscell, stats, f_cells = utils.load_suite2p(utils.pathSorter(plane)['stimuli']['frame_aligned'].parents[0].joinpath('suite2p/plane0'))

            self.datum_dict[plane]['suite2p'] = {}
            self.datum_dict[plane]['suite2p']['ops'] = ops
            self.datum_dict[plane]['suite2p']['iscell'] = iscell
            self.datum_dict[plane]['suite2p']['stats'] = stats
            self.datum_dict[plane]['suite2p']['f_cells'] = f_cells

    def load_stimuli(self, trims=(-3, 7), clip=80):
        for plane in self.plane_list:
            plane_data_paths = utils.pathSorter(plane)
            stimuli = pd.read_hdf(plane_data_paths['stimuli']['frame_aligned'])

            stimulus_starts = [i[trims[0]] for i in stimuli.img_stacks.values]
            stimulus_stops = [i[trims[0]]+trims[1] for i in stimuli.img_stacks.values]
            stimuli.loc[:, 'start'] = stimulus_starts
            stimuli.loc[:, 'stop'] = stimulus_stops

            self.datum_dict[plane]['stimuli'] = stimuli.iloc[:clip]

    def create_threshdf(self, method='background_subtracted', used_threshold=0.1):

        for plane in self.plane_list:
            df, n_cells = utils.return_stim_response_dict(plane, response_type=method)

            temp = df[df >= used_threshold].isna().sum(axis=1) < self.datum_dict[plane]['stimuli'].stim_name.nunique()
            threshdf = df.loc[np.where(temp == 1)]

            self.datum_dict[plane]['threshdf'] = threshdf

    def create_fdff(self, offset=-4):
        for plane in self.plane_list:
            stim = self.datum_dict[plane]['stimuli']
            fcells = self.datum_dict[plane]['suite2p']['f_cells']
            stim_fdff = utils.new_fdff_all_stims_faster(stim, fcells, offset)

            self.datum_dict[plane]['stim_fdff'] = stim_fdff

    def barcode(self, lowThresh=0.05):
        for plane in self.plane_list:
            stim = self.datum_dict[plane]['stimuli']
            threshdf = self.datum_dict[plane]['threshdf']

            bcode = barcoding(stim, threshdf, myLowThresh=lowThresh)

            self.datum_dict[plane]['barcode'] = bcode

    def blast(self, plane, neurons, pretty=True, highlights=(0,15,20)):
        '''
        :param plane: just the z # of the plane is fine
        :param neurons: list or int of neurons to plot
        :param pretty: bool for opencv circles instead of raw xy coords
        :param highlights: highlighting of the celltraces for stims
        :return:
        '''

        if pretty:
            import cv2

        local_dict = self.datum_dict[self.plane_list[plane]]
        local_stats = local_dict['suite2p']['stats']
        local_ops = local_dict['suite2p']['ops']

        ref_img = local_ops['refImg']
        cell_img = np.zeros((local_ops['Ly'], local_ops['Lx']))

        if hasattr(neurons, '__iter__'):
            for _, n in enumerate(neurons):
                ypix = local_stats[n]['ypix']
                xpix = local_stats[n]['xpix']
                if pretty:
                    mean_y = int(np.mean(ypix))
                    mean_x = int(np.mean(xpix))
                    colorval = _ + 1
                    cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, -1)
                else:
                    cell_img[ypix, xpix] = _ + 1
        else:
            ypix = local_stats[neurons]['ypix']
            xpix = local_stats[neurons]['xpix']

            if pretty:
                mean_y = int(np.mean(ypix))
                mean_x = int(np.mean(xpix))
                colorval = 1
                cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, -1)
            else:
                cell_img[ypix, xpix] = 1

        masked = np.ma.masked_where(cell_img < 0.9, cell_img)

        local_fdff = local_dict['stim_fdff']
        local_stim = local_dict['stimuli']

        if hasattr(neurons, '__iter__'):
            myDictionary = create_myDictionary(local_stim, local_fdff[local_fdff.neuron.isin(neurons)])
        else:
            myDictionary = create_myDictionary(local_stim, local_fdff[local_fdff.neuron == neurons])



        subplot_locations = {'lateral_left': 7, 'lateral_right': 2, 'medial_left': 6, 'medial_right': 3,
                             'left': 5, 'right': 4, 'forward': 1, 'backward': 0}
        invDict = {v: k for k, v in subplot_locations.items()}


        fig = plt.figure(figsize=(6,12))

        imgspot = plt.subplot2grid((8, 8), (0, 0), rowspan=8, colspan=4)
        imgspot.imshow(ref_img, cmap='gray')
        imgspot.imshow(masked, cmap='cool')

        imgspot.set_xticks([])
        imgspot.set_yticks([])

        axes = [plt.subplot2grid((8,2), (n,1), colspan=1) for n in range(8)]
        for n, ax in enumerate(axes):

            ax_stim = invDict[n]
            yvals, err = myDictionary[ax_stim]
            xvals = np.arange(0, len(yvals))
            ax.errorbar(x=xvals, y=yvals, yerr=err, color='black', linewidth=4)
            ax.axvspan(highlights[1], highlights[2], alpha=0.6, color=eva_weightings[ax_stim])
            ax.set_ylim(-.5, 1.0)
            ax.set_xlim(13, 25)
            ax.axhline(0, color='red')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(ax_stim)

        fig.tight_layout()
        plt.show()

    def plot_nclasses(self, plane, top_n, bbox=(2,0.5), pretty=False, invert=True):

        if pretty:
            import cv2

        local_dict = self.datum_dict[self.plane_list[plane]]
        local_stats = local_dict['suite2p']['stats']
        local_ops = local_dict['suite2p']['ops']

        ref_img = local_ops['refImg']
        cell_img = np.zeros((local_ops['Ly'], local_ops['Lx']))

        bcode = local_dict['barcode']
        value_df = pd.DataFrame(bcode.set.value_counts())

        for x in range(top_n):
            neurons = bcode[bcode.set == value_df.index.values[x]].neuron.values

            for n in neurons:
                ypix = local_stats[n]['ypix']
                xpix = local_stats[n]['xpix']

                if not pretty:
                    cell_img[ypix, xpix] = x + 1
                else:
                    mean_y = int(np.mean(ypix))
                    mean_x = int(np.mean(xpix))
                    colorval = x + 1
                    cv2.circle(cell_img, (mean_x, mean_y), 3, colorval, -1)

        _cmap = ListedColormap(sns.color_palette("husl", top_n).as_hex())

        masked = np.ma.masked_where(cell_img < 0.9, cell_img)
        fig = plt.figure(figsize=(6,6))
        plt.imshow(ref_img, cmap='gray')
        plt.imshow(masked, cmap=_cmap)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

        used_colors = return_n_colors(top_n, _cmap)
        legend_lines = [Line2D([0], [0], color=clr, marker=' ', linestyle=None) for clr in used_colors]
        legend_labels = [k for k in value_df.index.values[:top_n]]
        plt.legend(legend_lines, legend_labels, loc='right', bbox_to_anchor=bbox, title='set')

        if invert:
            plt.gca().invert_xaxis()
        plt.show()
