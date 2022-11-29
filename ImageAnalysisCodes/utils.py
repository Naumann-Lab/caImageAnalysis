from tkinter import Tk
from tkinter import filedialog as fd
from pathlib import Path
from nptdms import TdmsFile
from datetime import datetime as dt
from datetime import date

from ImageAnalysisCodes.visualize import eva_weightings

import datetime
import shutil
import os
import copy

try:
    import caiman as cm
except:
    print('caiman not available')

import seaborn as sns
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def pathSorter(parentPath=None, key=None):
    """
    finds and returns specific paths to items from a home folder

    :param parentPath: central directory to data belonging to an experiment
    :param key: can request a specific path to be returned
    :return: returns requested path or all paths
    """

    # if a path is not provided, provides a popup window to select data source
    if parentPath is None:
        root = Tk()
        root.update()
        parentPath = fd.askdirectory(parent=root, title='Please Select Data Folder')
        root.destroy()

    _paths = {'stimuli': {}, 'image': {}, 'output': {}, 'etc': {}}

    volumetric=False
    with os.scandir(parentPath) as entries:
        for entry in entries:
            if entry.is_file():

                if entry.name.endswith('Notes.txt'):
                    _paths['etc']['notes'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('Params.json'):
                    _paths['etc']['moveCorrectionParams'] = Path(parentPath).joinpath(entry.name)

                if entry.name.endswith('.tif'):
                    _paths['image']['raw'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('Timestamps.tdms'):
                    _paths['image']['timestamps'] = Path(parentPath).joinpath(entry.name)
                if 'timestamps.txt' in entry.name:
                    _paths['image']['new_timestamps'] = Path(parentPath).joinpath(entry.name)

                if 'frametimes.txt' in entry.name:
                    _paths['image']['anne_timestamps'] = Path(parentPath).joinpath(entry.name)

                if 'frametimes.h5' in entry.name:
                    _paths['image']['new_timestamps_processed'] = Path(parentPath).joinpath(entry.name)


                if ('stims' in entry.name or entry.name.startswith('fish')) and entry.name.endswith('.txt'):
                    _paths['stimuli']['raw'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('.h5') and entry.name.startswith('fish'):
                    _paths['stimuli']['processed'] = Path(parentPath).joinpath(entry.name)
                if entry.name.endswith('aligned.h5'):
                    _paths['stimuli']['frame_aligned'] = Path(parentPath).joinpath(entry.name)

                if 'tail' in entry.name and entry.name.endswith('.tdms'):
                    _paths['etc']['tail'] = Path(parentPath).joinpath(entry.name)

                if 'log' in entry.name and entry.name.endswith('.txt'):
                    _paths['image']['log'] = Path(parentPath).joinpath(entry.name)

                if 'eigen' in entry.name:
                    _paths['output']['eigenvalues'] = Path(parentPath).joinpath(entry.name)
            else:
                if entry.name=='timing':
                    _paths['image']['new_timestamps'] = Path(parentPath).joinpath('timing/framesendtimes.txt')
                if entry.name == 'suite2p':
                    _paths['output']['suite2p'] = Path(parentPath).joinpath('suite2p/plane0')

            if 'planes' in entry.name:
                volumetric = True
                volumePaths = Path(parentPath).joinpath('planes')

                if not os.path.exists(volumePaths.joinpath('move_corrected')):
                    print('please move correct volume first')
                    return
                else:
                    volumePaths = volumePaths.joinpath('move_corrected')

    if volumetric:
        _paths['image']['volume'] = {}

        with os.scandir(volumePaths) as entries:
            for entry in entries:
                _paths['image']['volume'][entry.name] = volumePaths.joinpath(entry.name)



    moveCorrectedPath = Path(parentPath).joinpath('move_corrected')
    if moveCorrectedPath.exists():
        with os.scandir(moveCorrectedPath) as entries:
            for entry in entries:
                if entry.name.endswith('.tif'):
                    _paths['image']['move_corrected'] = Path(moveCorrectedPath).joinpath(entry.name)

    suite2pPath = Path(parentPath).joinpath('move_corrected').joinpath('suite2p')
    if suite2pPath.exists():
        _paths['output']['suite2p'] = {
            "iscell": Path(suite2pPath).joinpath('plane0/iscell.npy'),
            "stats": Path(suite2pPath).joinpath('plane0/stat.npy'),
            "ops": Path(suite2pPath).joinpath('plane0/ops.npy'),
            "f_cells": Path(suite2pPath).joinpath('plane0/F.npy'),
            "f_neuropil": Path(suite2pPath).joinpath('plane0/Fneu.npy'),
            "spikes": Path(suite2pPath).joinpath('plane0/spks.npy'),
            "data": Path(suite2pPath).joinpath('plane0/data.bin')
        }
    if key is not None:
        return _paths[key]
    else:
        return _paths


def load_suite2p(_paths):
    try:
        ops = np.load(_paths['ops'], allow_pickle=True).item()
        iscell = np.load(_paths['iscell'], allow_pickle=True)[:, 0].astype(bool)
        stats = np.load(_paths['stats'], allow_pickle=True)
        f_cells = np.load(_paths['f_cells'])
        return ops, iscell, stats, f_cells
    except KeyError:
        print('please run suite2p neuron extraction first')
    except TypeError:
        ops = np.load(_paths.joinpath('ops.npy'), allow_pickle=True).item()
        iscell = np.load(_paths.joinpath('iscell.npy'), allow_pickle=True)[:, 0].astype(bool)
        stats = np.load(_paths.joinpath('stat.npy'), allow_pickle=True)
        f_cells = np.load(_paths.joinpath('F.npy'))
        return ops, iscell, stats, f_cells


def load_eigen(eigen_path):
    return np.load(eigen_path, allow_pickle=True)['arr_0']


def load_stimuli(stimuli_path):
    # requires stimuli to be mapped onto df
    df = pd.read_hdf(stimuli_path)
    stim_df = stimStartStop(df)
    stim_df = stim_df[stim_df.stimulus.notna()]
    stim_df.reset_index()
    return colormapStimuli(stim_df)


def map_raw_stimuli(stims):
    monocDic = {
        0:'Forward',
        45:'ForwardRight',
        50:'ForwardRight',
        90:'Right',
        100:'Right',
        125:'BackwardRight',
        135:'BackwardRight',
        175:'Backward',
        180:'Backward',
        225:'BackwardLeft',
        270:'Left',
        275:'Left',
        315:'ForwardLeft',
        325:'ForwardLeft'
    }

    stims.loc[:, 'stimulus_name'] = '---'
    inds = stims.loc[(stims.stim_type == 's')].index
    vals = stims.loc[(stims.stim_type == 's')].angle.map(monocDic)
    stims.loc[inds, 'stimulus_name'] = vals

    angs = stims[(stims.stim_type == 'b')].angle.values
    ang_inds = stims[(stims.stim_type == 'b')].index
    rights = []
    lefts = []
    x = 0
    for a in angs:
        if a == [90, 90]:
            rights.append(x)
        elif a == [270, 270]:
            lefts.append(x)
        x += 1

    fullrights = []
    medrights = []
    latrights = []
    n = 0
    for v in stims.loc[ang_inds[rights]].velocity:
        if v[0] != 0 and v[1] != 0:
            fullrights.append(n)
        elif v[0] == 0 and v[1] != 0:
            latrights.append(n)
        elif v[0] != 0 and v[1] == 0:
            medrights.append(n)
        n += 1

    fulllefts = []
    medlefts = []
    latlefts = []
    n = 0
    for v in stims.loc[ang_inds[lefts]].velocity:
        if v[0] != 0 and v[1] != 0:
            fulllefts.append(n)
        elif v[0] == 0 and v[1] != 0:
            medlefts.append(n)
        elif v[0] != 0 and v[1] == 0:
            latlefts.append(n)
        n += 1

    for i in fullrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'RR'
    for i in medrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'Rx'
    for i in latrights:
        stims.loc[ang_inds[rights][i], 'stimulus_name'] = 'xR'

    for i in fulllefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'LL'
    for i in medlefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'xL'
    for i in latlefts:
        stims.loc[ang_inds[lefts][i], 'stimulus_name'] = 'Lx'

    return stims


def map_stimuli(stims):
    """
    Maps binocular stimuli

    Rx - medial right
    xR lateral right

    Lx lateral left
    xL medial left

    F forward
    B backward

    C converging
    D divering

    Easy to add different stims or replace this function

    :param stims:
    :return:
    """
    if 'stim_name' in stims.columns:
        stims.loc[:, 'stimulus'] = stims.stim_name.values
        return stims

    stims.loc[(stims['velocity_0'] == 0) & (stims['velocity_1'] != 0) & (stims['angle_1'] == 90), 'stimulus'] = 'Rx'
    stims.loc[(stims['velocity_0'] == 0) & (stims['velocity_1'] != 0) & (stims['angle_1'] == 270), 'stimulus'] = 'Lx'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] == 0) & (stims['angle_1'] == 90), 'stimulus'] = 'xR'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] == 0) & (stims['angle_1'] == 270), 'stimulus'] = 'xL'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 0) & (
                stims['angle_1'] == 0), 'stimulus'] = 'F'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 180) & (
                stims['angle_1'] == 180), 'stimulus'] = 'B'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 90) & (
                stims['angle_1'] == 90), 'stimulus'] = 'RR'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 270) & (
                stims['angle_1'] == 270), 'stimulus'] = 'LL'

    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 270) & (
                stims['angle_1'] == 90), 'stimulus'] = 'C'
    stims.loc[(stims['velocity_0'] != 0) & (stims['velocity_1'] != 0) & (stims['angle_0'] == 90) & (
                stims['angle_1'] == 270), 'stimulus'] = 'D'

    return stims


def colormapStimuli(stim_df):
    stims = stim_df.stimulus.unique()
    clrs = sns.color_palette(n_colors=len(stims))

    color_dict = {}

    x=0
    for stim in stims:
        try:
            color_dict[stim] = tuple(eva_weightings[stim])
        except KeyError:
            print('error on color: ', stim)
            color_dict[stim] = color_dict[x]
        x+=1

    try:
        stim_df.loc[:, 'color'] = stim_df.stimulus.map(color_dict)
    except NotImplementedError:
        stim_df.loc[:, 'color'] = stim_df.stimulus.astype(str).map(color_dict)
    except TypeError:
        color_dict = {}
        x = 0
        for stim in stims:
            color_dict[stim] = [clrs[x]]
            x += 1
        stim_df.loc[:, 'color'] = stim_df.stimulus.map(color_dict)

    return stim_df


def stimStartStop(stims):
    '''
    agnostic to what type of stimulus, it just needs to have stimuli labeled and include start/stop

    :param stims:
    :return:
    '''
    if not 'stimulus' in stims.columns:
        try:
            stims = map_stimuli(stims)
        except:
            print('error: please map stimuli on dataframe')
            return

    stimuli = []
    starts = []
    stops = []

    for row in range(len(stims)):
        data = stims.iloc[row]

        stimuli.append(data.stimulus)
        starts.append(data.img_stacks.values[0])
        stops.append(data.img_stacks.values[-1])

    stim_dict = {
        'stimulus': stimuli,
        'start': starts,
        'stop': stops
    }

    return pd.DataFrame(stim_dict)


def arrangedArrays(df, offset=5):
    '''
    only feed a dataframe with one stimulus and itll filter the repeats and return a single array image stacks

    '''

    if not 'start' in df.columns:
        print('error: use start stop format for stimuli')
        return
    a = []
    for repeat in range(len(df)):
        s = df.iloc[repeat].start + offset
        e = df.iloc[repeat].stop + offset
        a.append(np.arange(s, e))

    b = []
    for i in a:
        for j in i:
            b.append(j)
    return np.array(b)


def arrangedArrays2(df, offset=10):
    # like the other but it gives you an offset back and forward
    a = []
    for repeat in range(len(df)):
        s = df.iloc[repeat].start - offset
        e = df.iloc[repeat].stop + offset
        a.append(np.arange(s, e))

    b = []
    for i in a:
        for j in i:
            b.append(j)
    return np.array(b)


def neuronResponses(neuronArray, isCellArray, stimulusDf):

    if not 'start' in stimulusDf.columns:
        try:
            stimulusDf = stimStartStop(stimulusDf)
        except:
            print('error: use start stop format for stimuli')
            return

    nrns = np.where(isCellArray == 1)[0]
    neuronArray = neuronArray[isCellArray]
    neuron_responses = []

    for nrn_ind in range(len(neuronArray)):

        nrn_dict = {}
        nrn = neuronArray[nrn_ind]

        background_val = np.median(nrn)
        nrn_dict['bg'] = background_val

        for stim in sorted(stimulusDf.stimulus.unique()):
            stim_indices = arrangedArrays(stimulusDf[stimulusDf.stimulus == stim])
            stim_val = np.median(nrn[stim_indices])

            nrn_dict[stim] = stim_val

        neuron_responses.append(nrn_dict)
    neuron_df = pd.DataFrame(neuron_responses)
    neuron_df.loc[:, 'neuron'] = nrns
    return neuron_df


def neuronResponsesFdff(neuronArray, isCellArray, stimulusDf):

    if not 'start' in stimulusDf.columns:
        try:
            stimulusDf = stimStartStop(stimulusDf)
        except:
            print('error: use start stop format for stimuli')
            return

    nrns = np.where(isCellArray == 1)[0]
    neuronArray = neuronArray[isCellArray]
    neuron_responses = []

    for nrn_ind in range(len(neuronArray)):

        nrn_dict = {}
        nrn = neuronArray[nrn_ind]

        background_val = np.median(nrn)
        nrn_dict['bg'] = background_val

        for stim in sorted(stimulusDf.stimulus.unique()):
            stim_indices = arrangedArrays(stimulusDf[stimulusDf.stimulus == stim])
            stim_val = (np.median(nrn[stim_indices]) - background_val) / background_val

            nrn_dict[stim] = stim_val

        neuron_responses.append(nrn_dict)
    neuron_df = pd.DataFrame(neuron_responses)
    neuron_df.loc[:, 'neuron'] = nrns

    _df = neuron_df[neuron_df.bg > 0]
    _df.drop(columns='bg', inplace=True)

    return _df


def threshold_by_variance(eigen, variance=0.5):
    factors, loadings, x = eigen
    loading_mtrx = pd.DataFrame(loadings, columns=['FA{}'.format(i) for i in range(0, 10)], index=x.columns)
    loading_mtrx['highest_loading'] = loading_mtrx.idxmax(axis=1)
    b = loading_mtrx.drop(columns='highest_loading').T
    thresh = variance
    goods = []
    for i in b.columns:
        val = b[i].max()
        if val >= thresh:
            goods.append(i)
        else:
            pass
    good_cells = loading_mtrx.loc[goods]
    return good_cells


def pandas2hdf(_path):
    paths = pathSorter(_path)

    stimPath = paths['stimuli']['raw']

    with open(stimPath) as file:
        contents = file.read()

    # separate the text file into the different stimulus lines and withdraw the stimulus dictionaries
    parsed = contents.split('\n')
    fish_details = parsed[0]
    stimulus_details = parsed[1:]

    # some tricky text splitting
    times = [i[:i.find('{')] for i in stimulus_details]
    tex_freq = False
    if 'tex_freq' in stimulus_details[0]:
        tex_freq = True
        _stimulus_dicts = []
        tex_freqs = []
        for i in stimulus_details:
            _stimulus_dicts.append(i[i.find('{'):i.find('}') + 1])
            tex_freqs.append(i[i.find('freq: '):].split(' ')[-1])

        stimulus_dicts = [eval(i[i.find('{'):]) for i in _stimulus_dicts if 'stationary_end' not in i]
        freq_fixer = []
        for i in range(len(tex_freqs)):
            if tex_freqs[i] == '}':
                freq_fixer.append(tex_freqs[i - 1])
            else:
                freq_fixer.append(tex_freqs[i])
    elif 'texture_0' in stimulus_details[0]:
        stimulus_dicts = [eval(i[i.find('{'): int(i.find('texture_0')) - 1] + '}') for i in stimulus_details if
                          'stationary_end' not in i]
    else:
        stimulus_dicts = [eval(i[i.find('{'):]) for i in stimulus_details if 'stationary_end' not in i]

    # mostly a binocular gratings fix, need to stack the tuples into two separate columns
    for stim in range(len(stimulus_dicts)):
        for item in stimulus_dicts[stim].copy():
            try:
                if len(stimulus_dicts[stim][item]) > 1 and type(stimulus_dicts[stim][item]) is not str:
                    for i in range(len(stimulus_dicts[stim][item])):
                        name = item + '_' + str(i)
                        stimulus_dicts[stim][name] = stimulus_dicts[stim][item][i]
                    stimulus_dicts[stim].pop(item)
            except:
                pass

    stim_df = pd.DataFrame(stimulus_dicts)

    final_stims = stim_df
    if tex_freq:
        final_stims.loc[:, 'freq'] = freq_fixer

    # interpret the times and set up an array to measure elapsed times across experiment
    ntime_array = []
    for i in range(len((times))):
        ntime_array.append(dt.strptime(times[i].split(' ')[1], '%H:%M:%S.%f:'))
    time_array = []
    rawt_array = []
    for i in range(len(times)):
        try:
            # time_array.append((ntime_array[i + 1] - ntime_array[i]).total_seconds())
            rawt_array.append(str(ntime_array[i])[11:])
        except:
            pass

    final_stims.loc[:, 'raw_t'] = rawt_array

    # save a new file (don't overwrite an existing)
    fish_details = fish_details[:fish_details.rfind(' ')]

    val_offset = 0
    new_file = Path(_path).joinpath(f'{fish_details}_{val_offset}.h5')

    while os.path.exists(new_file):
        val_offset += 1
        new_file = Path(_path).joinpath(f'{fish_details}_{val_offset}.h5')


    final_stims.to_hdf(new_file, key='df')
    print('file saved:', new_file)
    return new_file


def anne_frametimes(_path):

    with open(_path) as file:
        contents = file.read()
    tstamps = contents.split('\n')[:-1]
    times = []
    for i in tstamps:
        times.append(dt.fromtimestamp(float(i.split(' ')[1])))
    return pd.DataFrame(times)


def pandas_stim_aligner(_path, anne_timestamps=False, mapstims=False):
    # requires a path to be sent in as a h5 stimuli path
    # requires an extra line or two into the pathSorter util for non-pandas stimuli
    # really just needs to be directed to a dataframe with a 'raw_t' column
    paths = pathSorter(_path)
    new_timestamps=False
    try:
        stimulusPath = paths['stimuli']['processed']
    except KeyError:
        print('trying to convert stim file')
        try:
            pandas2hdf(_path)
            paths = pathSorter(_path)
            stimulusPath = paths['stimuli']['processed']
        except:
            print('failed to process stimulus')
            return

    stims = pd.read_hdf(stimulusPath)
    stims['raw_t'] = stims['raw_t'].apply(lambda a: dt.strptime(a, '%H:%M:%S.%f').time())

    if 'duration_0' in stims:
        stims.loc[:, 'duration'] = stims.duration_0.values

    try:
        stims.duration.values
    except AttributeError:
        stims.loc[:, 'duration'] = stims.stim_time.values - stims.stationary_time.values

    if anne_timestamps:
        image_times = anne_frametimes(paths['image']['anne_timestamps'])
    else:
        try:
            frameTimePath = paths['image']['timestamps']
            image_times = image_timings(frameTimePath)

        except KeyError:
            print('no legacy timestamps found')
            try:
                image_times = pd.read_hdf(paths['image']['new_timestamps_processed']).rename({0:'times'}, axis=1).reset_index()
                image_times = image_times.drop(columns='index')
                new_timestamps = True
            except KeyError:
                try:
                    frameTimePath = paths['image']['new_timestamps']
                    image_times = raw_text_frametimes_to_df(frameTimePath)
                    image_times = image_times.rename({0:'times'}, axis=1).reset_index()
                    image_times.drop(columns='index', inplace=True)

                    new_timestamps = True
                except KeyError:
                    print('no updated timestamps found')
                    return

    # start_ind = image_times[image_times.times >= stims.loc[0].raw_t].index[0]
    if new_timestamps:
        diffs = []
        for i in range(len(image_times) - 2):
            diffs.append(dt.combine(date.today(), image_times.iloc[i + 1].values[0]) - dt.combine(date.today(), image_times.iloc[i].values[0]))
        imageHz = 1/np.mean(diffs).total_seconds()
        print('calculated imageHz at :', imageHz)
    else:
        try:
            notesPath = paths['etc']['notes']
            _notes = pd.read_csv(notesPath, sep=':', header=None)
            imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
        except KeyError:
            imageHz = 500
            print('defaulting to 2 Hz')

    try:
        end_ind_offset = (stims.iloc[-1].duration - stims.iloc[-1].stationary_time) // (imageHz / 1000)
    except AttributeError:
        _statval = max((stims.iloc[-1].duration - stims.iloc[-1].stationary_time_0), (stims.iloc[-1].duration - stims.iloc[-1].stationary_time_1))
        end_ind_offset = _statval // (imageHz / 1000)

    try:
        end_ind = image_times[image_times.times >= stims.iloc[-1].raw_t].index[0] + end_ind_offset
    except IndexError:
        end_ind = len(image_times) - 1

    imgs = []
    stim_num = len(stims)
    for i in range(stim_num):
        if i + 1 >= stim_num:
            break
        indices = image_times[
            (image_times.times >= stims.loc[i].raw_t) & (image_times.times <= stims.loc[i + 1].raw_t)].index
        imgs.append(indices)
    imgs.append(image_times[(image_times.times >= stims.loc[stim_num - 1].raw_t)].loc[:end_ind].index)
    try:
        stims.loc[:, 'img_stacks'] = imgs
    except ValueError:
        new_raw_t = []

        for i in stims.raw_t:
            new_raw_t.append(dt.strptime(i.strftime('%I:%M:%S.%f:'), '%I:%M:%S.%f:').time())
        stims.raw_t = new_raw_t

        stims = stims[stims.raw_t >= image_times.times.values[0]]

        imgs = []
        stim_num = len(stims)
        for i in range(stim_num):
            if i + 1 >= stim_num:
                break
            indices = image_times[
                (image_times.times >= stims.iloc[i].raw_t) & (image_times.times <= stims.iloc[i + 1].raw_t)].index
            imgs.append(indices)
        stims = stims.iloc[1:]
        stims.loc[:, 'img_stacks'] = imgs

    df_trimmer = []
    for row in range(len(stims)):
        if len(stims.img_stacks.values[row]) > 0:
            df_trimmer.append(True)
        else:
            df_trimmer.append(False)

    stims = stims[df_trimmer]

    if mapstims:
        stims = map_raw_stimuli(stims)

    stims.to_hdf(os.path.join(_path, 'frame_aligned.h5'), key='stimuli')
    return imageHz


def image_timings(frameTimePath):

    try:
        # legacy way
        time_data = TdmsFile(frameTimePath).object('2P_Frame', 'Time').data

        time_array = []
        date_array = []

        for i in range(len(time_data)):
            time_array.append(time_data[i].replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).time())
            date_array.append(time_data[i].replace(tzinfo=datetime.timezone.utc).astimezone(tz=None).date())

    except AttributeError:
        # updated
        time_data = TdmsFile(frameTimePath)['2P_Frame']['Time'][:]
        time_array = []
        date_array = []

        for i in range(len(time_data)):
            time_array.append(pd.to_datetime(time_data[i], utc=True).tz_convert('America/New_York').time())
            date_array.append(pd.to_datetime(time_data[i], utc=True).tz_convert('America/New_York').date())

    frame_dic = {"times": time_array, 'Date': date_array}
    imageTimeDf = pd.DataFrame(frame_dic)

    return imageTimeDf


def raw_text_frametimes_to_df(time_path):
    with open(time_path) as file:
        contents = file.read()
    parsed = contents.split('\n')

    times = []
    for line in range(len(parsed) - 1):
        times.append(dt.strptime(parsed[line], '%H:%M:%S.%f').time())
    return pd.DataFrame(times)


def raw_text_logfile_to_df(log_path, frametimes=None):
    with open(log_path) as file:
        contents = file.read()
    split = contents.split('\n')

    movesteps = []
    times = []
    for line in range(len(split)):
        if 'piezo' in split[line] and 'connected' not in split[line] and 'stopped' not in split[line]:
            t = split[line].split(' ')[0][:-1]
            z = split[line].split(' ')[6]
            try:
                if isinstance(eval(z), float):
                    times.append(dt.strptime(t, '%H:%M:%S.%f').time())
                    movesteps.append(z)
            except NameError:
                continue
    else:
        # last line is blank and likes to error out
        pass
    log_steps = pd.DataFrame({'times': times, 'steps': movesteps})

    if frametimes is not None:
        log_steps = log_aligner(log_steps, frametimes)
    else:
        pass
    return log_steps



def log_aligner(logsteps, frametimes):

    trimmed_logsteps = logsteps[(logsteps.times >= frametimes.iloc[0].values[0])&(logsteps.times <= frametimes.iloc[-1].values[0])]
    return trimmed_logsteps


def sequentialVolumes(frametimePath, imgPath, steps=5, leadingFrame=None):
    frametimes = raw_text_frametimes_to_df(frametimePath)
    img = cm.load(imgPath)

    if leadingFrame is not None:
        img = img[leadingFrame:]
        frametimes = frametimes.loc[leadingFrame:]

    step = img.shape[0] / steps

    # imgs = [[]] * steps
    frametime_all = [[]] * steps
    imgpaths = []
    frametime_paths = []

    for i in range(steps):

        root_path = Path(frametimePath).parents[0].joinpath('planes')

        try:
            os.mkdir(root_path)
        except FileExistsError:
            pass

        image = img[int(i*step) : int((i+1)*step)]
        frames = frametimes.loc[int(i*step) : int((i+1)*step)]

        # imgs[i] = image
        frametime_all[i] = frames

        imagePath = root_path.joinpath(f'{i}.tif')
        framePath = root_path.joinpath(f'{i}_frametimes.h5')

        imgpaths.append(imagePath)
        frametime_paths.append(framePath)

        image.save(imagePath)
        frames.to_hdf(framePath, 'frametime')
        print(f'finished frame: {i}')

    return [frametime_all], [imgpaths, frametime_paths]


def volumeSplitter(logPath, frametimePath, imgPath, leadingFrame=None, extraStep=False, intermediate_return=False):
    frametimes = raw_text_frametimes_to_df(frametimePath)
    logfile = raw_text_logfile_to_df(logPath, frametimes)
    img = cm.load(imgPath)

    if leadingFrame is not None:
        img = img[leadingFrame:]
        frametimes = frametimes.loc[leadingFrame:]

    if intermediate_return:
        return frametimes, logfile, img

    if extraStep:
        n_imgs = logfile.steps.nunique() - 1
    else:
        n_imgs = logfile.steps.nunique()

    imgs = [[]] * n_imgs
    frametime_all = [[]] * n_imgs

    imgpaths = []
    frametime_paths = []

    root_path = Path(logPath).parents[0].joinpath('planes')

    try:
        os.mkdir(root_path)
    except FileExistsError:
        print('')

    x=0
    for i in range(n_imgs):
        new_img = img[i::n_imgs]
        new_img_frametime = frametimes.iloc[1::n_imgs]

        imgs[i] = new_img
        frametime_all[i] = new_img_frametime

        new_img_path = root_path.joinpath(f'{x}.tif')
        new_framet_path = root_path.joinpath(f'{x}_frametimes.h5')
        imgpaths.append(new_img_path)
        frametime_paths.append(new_framet_path)

        new_img.save(new_img_path)
        new_img_frametime.to_hdf(new_framet_path, 'frametimes')

        print(f'saved {new_img_path}')
        print(f'saved {new_framet_path}')
        x += 1

    return [imgs, frametime_all], [imgpaths, frametime_paths]


def stackHzReturner(timeframe_df):
    t_vals = timeframe_df.loc[:, 0].values
    dt_offset = date.today()

    times = []
    for i in range(len(t_vals) - 1):
        times.append(dt.combine(dt_offset, t_vals[i + 1]) - dt.combine(dt_offset, t_vals[i]))

    return 1 / np.mean(times).total_seconds()


def returnFoverF(stim_df, good_cells):
    inds = []
    try:
        for arr in stim_df[(stim_df.velocity_0 == 0) & ((stim_df.velocity_1 == 0))].img_stacks.values:
            for j in arr[5:-5]:
                inds.append(j)
    except AttributeError:
        for arr in stim_df[(stim_df.velocity == 0) & ((stim_df.velocity == 0))].img_stacks.values:
            for j in arr[5:-5]:
                inds.append(j)

    fDff = []
    for i in range(good_cells.shape[0]):
        background = np.nanmean(good_cells[i, inds])
        ff = (good_cells[i, :] - background) / background
        fDff.append(ff)
    return np.array(fDff)


def volumeFrametimeMover(frametimePaths):
    basePath = frametimePaths[0].parents[0].joinpath('move_corrected')

    for i in range(len(frametimePaths)):
        locPath = basePath.joinpath(f'plane_{i}/{frametimePaths[i].name}')
        try:
            shutil.copy(frametimePaths[i], locPath)
        except:
            print(f'failed to move {locPath}')


def stimCopier(stimPath, planes=5):
    basePath = stimPath.parents[0].joinpath('planes/move_corrected')

    for i in range(planes):
        locPath = basePath.joinpath(f'plane_{i}/{stimPath.name}')
        shutil.copy(stimPath, locPath)

# probably dont use this
def lineOffset(image, offset_amount=1):
    N_img = image.copy()
    rows, cols = (N_img.shape[1], N_img.shape[2])

    for r in range(rows):
        if r < (rows - offset_amount) and (r + offset_amount) >= 0:
            if (r % 2) == 0:
                N_img[:, r, :] = N_img[:, r + offset_amount, :]

    return N_img

# expects image of XY dimensionality, shifts every row either 0/2/4... or 1/3/5...
def new_line_offset(img, row=0, shift=1):
    new_img = img.copy()
    new_img[row::2] = np.roll(new_img[row::2], axis=1, shift=shift)
    new_img = new_img[:,shift:]
    return new_img

def reduce_to_pi(ar):
    ar = ar * np.pi/180
    return (np.mod(ar + np.pi, np.pi * 2) - np.pi)*180/np.pi


def rad2deg(ar):
    return (ar*np.pi)/180

def last_n_arrayd(df, n=15):
    arrs = []
    for r in range(len(df)):
        stub = df.iloc[r]
        arrs.append(np.arange(stub.start, stub.stop)[-n:])
    return np.concatenate(arrs)

def nrnResponses(mstim_df, cells, allowNegs=False):
    nrn_vals = []
    stimuluses = []
    nrns = []

    for cell in range(len(cells)):
        for stimulus in mstim_df.stimulus.unique():
            nrns.append(cell)
            stimuluses.append(stimulus)
            nrn_vals.append(np.mean(cells[cell, arrangedArrays(mstim_df[mstim_df.stimulus.isin([stimulus])])]))
    neuron_df = pd.DataFrame({'neuron': nrns, 'stimulus': stimuluses, 'val': nrn_vals})

    if not allowNegs:
        neuron_df.loc[neuron_df.val < 0, 'val'] = 0

    return neuron_df


def stim_cell_returner(base_path, mapraw=False, fix_stims=False, s2p=False):

    # anything after 8/27 has corrected stims

    stimuli = pd.read_hdf(pathSorter(base_path)['stimuli']['frame_aligned'])

    if mapraw:
        try:
            stimuli = map_raw_stimuli(stimuli)
        except:
            pass
    try:
        stimuli.loc[:, 'stimulus'] = stimuli.stimulus_name.values
    except AttributeError:
        try:
            stimuli.loc[:, 'stimulus'] = stimuli.stim_name.values
        except AttributeError:
            tmp = map_stimuli(stimuli)
            stimuli.loc[:, 'stimulus_name'] = tmp.stimulus.values
            stimuli.loc[:, 'stimulus'] = stimuli.stimulus_name.values

    try:
        stimuli[(stimuli.velocity_0 == 0)].index
        binoc = True
    except:
        binoc=False

    if binoc:
        try:
            inds_1 = stimuli[(stimuli.velocity == 0)].index
            inds_2 = stimuli[(stimuli.velocity_0 == 0) & (stimuli.velocity_1 == 0)].index
            dropped_inds = sorted(np.concatenate([inds_1, inds_2]))
        except AttributeError:
            inds_2 = stimuli[(stimuli.velocity_0 == 0) & (stimuli.velocity_1 == 0)].index
            dropped_inds = sorted(np.concatenate([inds_2]))

    else:
        inds_1 = stimuli[(stimuli.velocity == 0)].index
        dropped_inds = sorted(np.concatenate([inds_1]))

    stim_df = stimStartStop(stimuli.drop(dropped_inds))

    if fix_stims:
        ## fix lateral medial mixup
        xrs = stim_df[stim_df.stimulus == 'xR'].index
        xls = stim_df[stim_df.stimulus == 'xL'].index
        rxs = stim_df[stim_df.stimulus == 'Rx'].index
        lxs = stim_df[stim_df.stimulus == 'Lx'].index

        stim_df.loc[xrs, 'stimulus'] = 'Rx'
        stim_df.loc[xls, 'stimulus'] = 'Lx'
        stim_df.loc[rxs, 'stimulus'] = 'xR'
        stim_df.loc[lxs, 'stimulus'] = 'xL'

    stim_df.stimulus = stim_df.stimulus.astype('category')

    if s2p is not False:
        ops, iscell, stats, f_cells = s2p
    else:
        ops, iscell, stats, f_cells = load_suite2p(pathSorter(base_path)['output']['suite2p'])

    cells = returnFoverF(stimuli, f_cells[iscell])

    return stim_df, cells


def returnStimCellAligned(stims_all, cells_all, leading_frames=50):

    first_stims = []
    last_stims = []

    start_offset = 0
    end_offset = 1

    sameStart = False
    sameFinish = False

    # have to find the same start and end stimulus.
    # potentially stim started before imaging or ended after imaging
    # this finds a same-start and same-stop to excise further along

    while not sameStart and not sameFinish:
        for data in stims_all:
            first_stims.append(data.iloc[start_offset].stimulus)
            last_stims.append(data.iloc[-end_offset].stimulus)

        # make a set of the grabbed stimuli, if only one item in set they are all the same, otherwise take a step
        if len(set(first_stims)) == 1:
            sameStart = True
        else:
            start_offset += 1
        if len(set(last_stims)) == 1:
            sameFinish = True
        else:
            end_offset += 1

    # grab the start and end of the stimulation shown
    rel_inds = []
    for data in stims_all:
        rel_inds.append([data.iloc[start_offset].start, data.iloc[-end_offset].stop])

    # this is supposed to fix the lengths to be equal
    # seems to sorta work? ends up being +/- 1
    all_lens = []
    for x in rel_inds:
        all_lens.append(x[1] - x[0])

    if True in (np.diff(all_lens) > 0):
        for j in np.where(np.diff(all_lens) > 0):
            if np.diff(all_lens)[j] > 0:
                rel_inds[int(j)][1] += 1
            elif np.diff(all_lens)[j] < 0:
                rel_inds[int(j)][1] -= 1

    # maintain original stimuli (was once concerned with mutability and downstream processing)
    # this arranges the stimuli from 0-end

    fin_stims = copy.deepcopy(stims_all)

    for n in range(len(stims_all)):
        fin_stims[n].loc[:, 'start'] = stims_all[n].start.values - rel_inds[n][0]
        fin_stims[n].loc[:, 'stop'] = stims_all[n].stop.values - rel_inds[n][0]

    first_l = fin_stims[0].stop.values[-1]

    # excises the relevant part from each cell for the new 0-end
    fin_cells = []
    for b in range(len(cells_all)):
        offset_l = fin_stims[b].stop.values[-1] - first_l
        fin_cells.append(cells_all[b][:, rel_inds[b][0] - leading_frames + offset_l : rel_inds[b][1]])

    return fin_stims, fin_cells


def corrNeurons(new_variables, ncells, corrThreshold=0.45):
    # takes the path of the eigens to compare to
    # as well as the array of cells
    fctrs = pd.DataFrame(new_variables)


    if ncells.shape[1] == fctrs.shape[0]:
        df = pd.DataFrame(ncells).T
    else:
        df = pd.DataFrame(ncells)

    x = df.copy()
    x.replace([np.inf, -np.inf], np.nan)


    factor_corrs = [[]] * fctrs.shape[1]
    for i in range(fctrs.shape[1]):
        corrs = []
        for j in range(x.shape[1]):
            corrs.append(np.corrcoef(fctrs.loc[:, i], x.loc[:, j])[0][1])

        factor_corrs[i] = corrs

    fullcorr = pd.DataFrame(factor_corrs)
    asdf = abs(fullcorr).max() >= corrThreshold
    cordf = fullcorr.loc[:, asdf]

    uniqueFactorsN = cordf.idxmax().nunique()
    factor_neurons = [[]] * uniqueFactorsN
    for q in range(uniqueFactorsN):
        factor_neurons[q] = cordf.idxmax()[cordf.idxmax() == q].index
    return factor_neurons, factor_corrs


def multiAlignment(dataPaths, n_components=10):

    from sklearn.decomposition import FactorAnalysis
    from sklearn.preprocessing import StandardScaler

    stims_all = []
    cells_all = []

    for p in dataPaths:
        s, c = stim_cell_returner(p)
        stims_all.append(s)
        cells_all.append(c)

    fin_stims, fin_cells = returnStimCellAligned(stims_all, cells_all)

    celldfs = []
    for arr in fin_cells:
        celldfs.append(pd.DataFrame(arr).T)
    allneurons = pd.concat(celldfs, axis=1).T.reset_index(drop=True).T

    x = allneurons.copy()
    x.replace([np.inf, -np.inf], np.nan)
    x.dropna(axis=1, inplace=True)

    X = StandardScaler().fit_transform(x)
    transformer = FactorAnalysis(n_components=n_components, rotation='varimax', max_iter=50000, iterated_power=10, tol=0.0001)
    X_transformed = transformer.fit_transform(X)

    return allneurons, fin_cells, fin_stims, X_transformed


def tail_reader(tail_data_path):
    tail_data = TdmsFile(tail_data_path)
    df = tail_data.as_dataframe()
    df = df[df["/'TailLoc'/'Time'"].notna()]
    df.loc[:, 't'] = df["/'TailLoc'/'Time'"].values - df["/'TailLoc'/'Time'"].values[0]
    t_arr = []
    for t in range(len(df.t.values)):
        t_arr.append(np.timedelta64(df.t.values[t], 'ms').astype(int))
    df['t'] = t_arr
    df["/'TailLoc'/'Time'"] = df["/'TailLoc'/'Time'"].dt.tz_convert('US/Eastern')

    tail_ts = []
    for i in range(len(df)):
        tail_ts.append(
            dt.strptime(str(df["/'TailLoc'/'Time'"].iloc[i]).split(' ')[1].split('-')[0],
                                 '%H:%M:%S.%f').time())
    df.loc[:, 'conv_t'] = tail_ts
    return df.iloc[1:]


def tail_alignment(basePath):

    paths = pathSorter(basePath)
    tailPath = paths['etc']['tail']
    timestampsPath = paths['image']['new_timestamps_processed']

    fullTailData = tail_reader(tailPath)
    imageTimes = pd.read_hdf(timestampsPath)

    trimmedTailData = fullTailData[(fullTailData.conv_t>=imageTimes.values[0][0])&(fullTailData.conv_t<=imageTimes.values[-1][0])]
    for frameN in range(len(imageTimes.values)):
        try:
            indices = trimmedTailData[(trimmedTailData.conv_t >= imageTimes.values[frameN][0]) & (
                        trimmedTailData.conv_t <= imageTimes.values[frameN + 1][0])].index
        except IndexError:
            pass
        trimmedTailData.loc[indices, 'frame'] = frameN
    return trimmedTailData, fullTailData


def load_stimmed(base_path):
    anneStimPath = Path(base_path).joinpath('stimmed.txt')

    with open(anneStimPath) as file:
        contents = file.read()

    frame_ns = []
    evas = []
    angles = []
    vels = []
    freqs = []
    contrasts = []
    ts = []

    for stim in contents.split('\n'):
        try:
            f, e, a, v, f1, c, t = np.float64(stim.split(' '))

            frame_ns.append(f)
            evas.append(e)
            angles.append(a)
            vels.append(v)
            freqs.append(f1)
            contrasts.append(c)
            ts.append(t)
        except ValueError:
            pass

    df = pd.DataFrame(
        {'start_frame': frame_ns, 'eva': evas, 'angle': angles, 'vel': vels, 'freq': freqs, 'contrast': contrasts,
         'time': ts})
    return df

def listed(gen_obj):
    return list(gen_obj)

def pretty(x, z=5):
    return np.convolve(x, np.ones(z)/z, mode='same')


def normcells(arr : np.array) -> np.array:
    mins = np.min(arr, axis=1)
    mins = np.expand_dims(mins, axis=1)

    zerod_cells = np.subtract(arr, mins)

    maxes = np.max(zerod_cells, axis=1)
    maxes = np.expand_dims(maxes, axis=1)
    return np.clip(np.nan_to_num(np.divide(zerod_cells, maxes)), a_min=-1, a_max=1)


def return_stim_responses(stimulus_df: pd.DataFrame, cell_array: np.array, background_response: np.array, offset: int = 12) -> np.array:
    '''

    :param stimulus_df: dataframe containing the stimulus information : needs start value column
    :param cell_array: fcells or equivalent
    :param background_response: background number per cell
    :param offset: grabs bin of offset:2x offset for stim response
    :return: array of stim x cells
    '''

    assert(background_response.shape[0] == cell_array.shape[0])

    stimulus_starts = stimulus_df.start_frame.values

    stimulus_responses = []
    for stim in stimulus_starts:
        stimulus_inds = np.arange(stim + offset, stim + 2 * offset, dtype=np.int32)
        stimulus_resp = np.nanmean(cell_array[:, stimulus_inds], axis=1)
        stimulus_diff = stimulus_resp - background_response
        stimulus_responses.append(stimulus_diff)

    return np.array(stimulus_responses)


def return_stim_response_dict(data_path: str, stim_start_offset: int = -3, stim_stop_offset: int = 7,
                              response_type: str = 'mean') -> pd.DataFrame:
    '''
    :param data_path: path of folder containing single experiment data objects
    :param stim_start_offset: frame offset to begin grabbing responses
    :param stim_stop_offset: frame offset relative to start
    :param response_type: mean, median, max
    :return: dataframe of response type of choice for every cell and array of norm'd cells corresponding
    '''
    accepted_response_types = ['mean', 'max', 'median', 'background_subtracted']
    assert(response_type in accepted_response_types), f'response type must be in {accepted_response_types}'

    stuff = pathSorter(data_path)
    stim = pd.read_hdf(stuff['stimuli']['frame_aligned'])
    ops, iscell, stats, f_cells = load_suite2p(stuff['stimuli']['frame_aligned'].parents[0].joinpath('suite2p/plane0'))
    norm_cells = normcells(f_cells)
    convolved_arrs = np.array([np.convolve(norm_cells[i], np.ones(100)/100, 'same') for i in range(norm_cells.shape[0])])

    stimulus_starts = [i[stim_start_offset] for i in stim.img_stacks.values]
    stimulus_stops = [i[stim_start_offset]+stim_stop_offset for i in stim.img_stacks.values]
    stim.loc[:, 'start'] = stimulus_starts
    stim.loc[:, 'stop'] = stimulus_stops

    responseDict = {}
    for stimChoice in stim.stim_name.unique():
        stub = stim[stim.stim_name==stimChoice]
        stimFrames = []

        for i in range(len(stub)):
            stimFrames.append(np.arange(stub.iloc[i].start, stub.iloc[i].stop))
        stimframes = np.unique(np.clip(np.concatenate(stimFrames), a_min=0, a_max=norm_cells.shape[1]-1))

        if response_type == 'mean':
            responseDict[stimChoice] = np.nanmean(norm_cells[:, stimframes], axis=1)
        elif response_type == 'max':
            responseDict[stimChoice] = np.nanmax(norm_cells[:, stimframes], axis=1)
        elif response_type == 'median':
            responseDict[stimChoice] = np.nanmedian(norm_cells[:, stimframes], axis=1)
        elif response_type == 'background_subtracted':
            responseDict[stimChoice] = np.nanmean(norm_cells[:, stimframes], axis=1) - np.nanmean(convolved_arrs[:, stimframes], axis=1)

    return pd.DataFrame(responseDict), norm_cells


def better_volume_splits(base_path, len_thresh=150):
    import sys
    sys.path.append(r'C:\Users\matt_analysis\Documents\GitHub')
    from celltrack import utils as other_utils

    import tifffile


    loc_paths = pathSorter(base_path)

    frametimes = other_utils.raw_text_frametimes_to_df(loc_paths['image']['new_timestamps'])
    log_steps = other_utils.raw_text_logfile_to_df(loc_paths['image']['log'])
    frametimes = other_utils.alignmentFramesSteps(frametimes, log_steps, time_offset=0.009)

    img = cm.load(loc_paths['image']['raw'])

    for n, s in enumerate(frametimes.step.unique()):
        imgInds = frametimes[frametimes.step==s].index
        new_fts = frametimes[frametimes.step==s].drop(columns='step')

        folderpath = loc_paths['image']['new_timestamps'].parents[0].joinpath('planes')
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        sub_img = img[imgInds]
        if len(sub_img) >= len_thresh:
            new_img_path = folderpath.joinpath(f'{n}.tif')
            tifffile.imsave(new_img_path, sub_img)
            print(f'saved {new_img_path}')

            new_framet_path = folderpath.joinpath(f'{n}_frametimes.h5')
            new_fts.to_hdf(new_framet_path, 'frametimes')
            print(f'saved {new_framet_path}')

def full_vol_wrapper(base_path):
    from ImageAnalysisCodes import core

    better_volume_splits(base_path)

    parentPath = Path(base_path).joinpath('planes')
    with os.scandir(parentPath) as entries:
        for entry in entries:
            if entry.name.endswith('.tif'):
                imgpath = parentPath.joinpath(entry.name)
                img = core.movement_correction(imgpath, volume=True)
    frametimepaths = []
    with os.scandir(parentPath) as entries:
        for entry in entries:
            if 'frametimes' in entry.name:
                fpath = parentPath.joinpath(entry.name)
                frametimepaths.append(fpath)

    volumeFrametimeMover(frametimepaths)
    stimCopier(pathSorter(parentPath.parents[0])['stimuli']['raw'], len(frametimepaths))

    for plane in pathSorter(parentPath.parents[0])['image']['volume']:
        planePath = pathSorter(parentPath.parents[0])['image']['volume'][plane]
        imagefreq = pandas_stim_aligner(planePath)

    core.neuron_extraction(parentPath.parents[0], imageFrequency=imagefreq)


def return_thresholded(base_path, threshold=0.3,  color_set = 'full', response_type='mean'):

    '''
    :param base_path: base path to volumetric data
    :param threshold: threshold for cutoff
    :param color_set: full or partial stims
    :return: df with neurons and responess
    '''
    accepted_color_sets = ['simplified', 'complex', 'full']

    assert(color_set in accepted_color_sets), f'color type must be one of {accepted_color_sets}'

    all_raw_paths = pathSorter(base_path)
    plane_list = list(all_raw_paths['image']['volume'].values())

    thresh_dfs = []
    for z in range(len(plane_list)):
        plane_path = plane_list[z]
        plane_data_paths = pathSorter(plane_path)

        stimuli = pd.read_hdf(plane_data_paths['stimuli']['frame_aligned'])

        stimulus_starts = [i[-3] for i in stimuli.img_stacks.values]
        stimulus_stops = [i[-3]+7 for i in stimuli.img_stacks.values]
        stimuli.loc[:, 'start'] = stimulus_starts
        stimuli.loc[:, 'stop'] = stimulus_stops

        df, n_cells = return_stim_response_dict(plane_path, response_type=response_type)
        rev_df = df.T
        simplfied_set = ['forward', 'backward', 'left', 'right']
        complex_set = ['forward', 'backward', 'medial_right', 'medial_left', 'lateral_left', 'lateral_right']

        maxVals = []
        keyVals = []
        neuronVals = []

        for i in range(len(df)):
            maxVals.append(rev_df.loc[:, i].max())
            keyVals.append(rev_df.loc[:, i].index[rev_df.loc[:, i].argmax()])
            neuronVals.append(i)

        peak_df = pd.DataFrame({'neuron' : neuronVals, 'stimulus' : keyVals, 'response' : maxVals})
        thresholded_df = peak_df[peak_df.response>=threshold]

        if color_set == 'simplified':
            thresh_df = thresholded_df[thresholded_df.stimulus.isin(simplfied_set)]
        elif color_set == 'complex':
            thresh_df = thresholded_df[thresholded_df.stimulus.isin(complex_set)]
        else:
            thresh_df = thresholded_df
        thresh_dfs.append(thresh_df)
    return thresh_dfs


def return_locationed(base_path, threshold=0.3):
    from ImageAnalysisCodes import visualize

    if not isinstance(Path(base_path), Path):
        base_path = Path(base_path)

    assert('volume' in pathSorter(base_path)['image']), "must run on volumetric data"
    all_raw_paths = pathSorter(base_path)
    plane_list = list(all_raw_paths['image']['volume'].values())
    neuron_sets = return_thresholded(base_path, threshold=threshold)

    pos_dfs = []
    for i, plane in enumerate(plane_list):
        ops, iscell, stats, f_cells = load_suite2p(plane.joinpath('suite2p/plane0'))
        neuron_set = neuron_sets[i]
        x_pos = [np.mean(stats[neuron]['xpix']) for neuron in neuron_set.index]
        y_pos = [np.mean(stats[neuron]['ypix']) for neuron in neuron_set.index]
        colors = [visualize.eva_weightings[neuron] for neuron in neuron_set.stimulus.values]

        pos_df = pd.DataFrame({'stimulus' : neuron_set.stimulus.values, 'x' : x_pos, 'y' : y_pos, 'color' : colors })
        pos_dfs.append(pos_df)
    return pos_dfs


def return_avg_response_arrs(stimChoice, n_cell, offset=10, pretty_length=100):
    # dataframe of stim limited to stimchoice
    good_arrs = []
    bg_arrs = []
    conv_array = np.convolve(n_cell, np.ones(pretty_length)/pretty_length, 'valid')

    for stim in range(len(stimChoice)-1):
        stub = stimChoice.iloc[stim]
        s = stub.start - offset
        e = stub.stop + offset
        plot_arr = n_cell[s:e]
        bg_array = conv_array[s:e]
        if len(bg_array) == len(plot_arr):
            bg_arrs.append(bg_array)
            good_arrs.append(plot_arr)
    return good_arrs, bg_arrs


def new_fdff(stimuli, f_cells, buffer=2):
    # returns a generic fdff for cells across entire stimulus set at once
    baseline_period = np.min([stimuli.start.values[i] - stimuli.stop.values[i-1] for i in range(len(stimuli)) if i > 0]) - buffer # use two frame min buffer
    fdff_cells = []
    for cell_n in range(f_cells.shape[0]):
        bg_val_n = np.nanmean([f_cells[cell_n][stimuli.iloc[stim_n].start-baseline_period : stimuli.iloc[stim_n].start-1] for stim_n in range(len(stimuli))])
        fdff_cell = (f_cells[cell_n] - [bg_val_n]*len(f_cells[cell_n]))/np.nanmax((f_cells[cell_n] - [bg_val_n]*len(f_cells[cell_n])))
        fdff_cells.append(fdff_cell)
    return np.array(fdff_cells)


def new_fdff_avg_all(stimuli, f_cells):
    from tqdm.auto import tqdm
    baseline_period = np.min([stimuli.start.values[i] - stimuli.stop.values[i-1] for i in range(len(stimuli)) if i > 0]) - 2 # use two frame min buffer

    all_arrs = []
    real_arr_inds = []
    all_dfs = []
    for cell_n in tqdm(range(f_cells.shape[0])):
        for stim_n in range(len(stimuli)-1):
            bg_arr = f_cells[cell_n][stimuli.iloc[stim_n].start-baseline_period : stimuli.iloc[stim_n].start-1]

            _f_max = np.max(f_cells[cell_n])
            pltarr = f_cells[cell_n][stimuli.iloc[stim_n].img_stacks.values[0]:stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period]
            bgarr = [np.mean(bg_arr)]*len(pltarr)

            fdiff = pltarr - bgarr
            # fmax = np.max(fdiff)
            fmax = _f_max - np.mean(bgarr)

            fdff = fdiff/fmax
            all_arrs.append(fdff)
            # print(stimuli.iloc[stim_n].img_stacks.values[0], stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period)
            real_arr_inds.append(np.arange(stimuli.iloc[stim_n].img_stacks.values[0],stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period ))
            # print((stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period) - (stimuli.iloc[stim_n+1].img_stacks.values[0]))

        f_Df = pd.DataFrame({'corr_index' : np.concatenate(real_arr_inds), 'fdff' : np.concatenate(all_arrs)})

        dupes = f_Df[f_Df.duplicated('corr_index')].corr_index.unique()
        vals = [np.mean(f_Df[f_Df.corr_index==i].fdff.values) for i in dupes]

        one_df = f_Df[~f_Df.corr_index.isin(dupes)]
        two_df = pd.DataFrame({"corr_index" : dupes, "fdff" : vals})

        df = pd.concat([one_df, two_df])
        df.sort_values(by='corr_index', inplace=True)
        all_dfs.append(df)
    for n,_df in enumerate(all_dfs):
        _df.loc[:, 'neuron'] = n
    fulldf = pd.concat(all_dfs)
    return fulldf


def new_fdff_all_stims(stimuli, f_cells):
    from tqdm.auto import tqdm
    baseline_period = np.min([stimuli.start.values[i] - stimuli.stop.values[i-1] for i in range(len(stimuli)) if i > 0]) # use two frame min buffer

    all_dataframes = []
    for cell_n in tqdm(range(f_cells.shape[0])):
        all_arrs = []
        all_stims = []
        real_arr_inds = []
        for stim_n in range(len(stimuli)-1):
            bg_arr = f_cells[cell_n][stimuli.iloc[stim_n].start-baseline_period : stimuli.iloc[stim_n].start-1]

            _f_max = np.nanmax(f_cells[cell_n])
            pltarr = f_cells[cell_n][stimuli.iloc[stim_n].img_stacks.values[0]:stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period]
            bgarr = [np.nanmean(bg_arr)]*len(pltarr)

            fdiff = pltarr - bgarr
            # fmax = np.max(fdiff)
            fmax = _f_max - np.nanmean(bgarr)

            fdff = fdiff/fmax

            all_arrs.append(fdff)
            all_stims.append(stim_n)
            real_arr_inds.append(np.arange(stimuli.iloc[stim_n].img_stacks.values[0],stimuli.iloc[stim_n].img_stacks.values[-1]+baseline_period ))

        f_Dfs = pd.DataFrame({'stimulus' : all_stims, 'fdff' : all_arrs, 'corr_index' : real_arr_inds})
        f_Dfs.loc[:, 'neuron'] = cell_n
        all_dataframes.append(f_Dfs)
    return pd.concat(all_dataframes)

def new_fdff_all_stims_faster(stimuli, f_cells, offset=2):
    from tqdm.auto import tqdm

    baseline_period = np.min([stimuli.start.values[i] - stimuli.stop.values[i-1] for i in range(len(stimuli)) if i > 0]) +offset # use two frame min buffer

    starts = stimuli.iloc[:-1].start.values
    all_img_stack_values = stimuli.iloc[:-1].img_stacks.values

    all_dataframes = []

    for cell in tqdm(range(f_cells.shape[0])):
        arr = f_cells[cell]

        all_arrs = []
        all_stims = []
        real_arr_inds = []

        for n, s in enumerate(starts):
            s = s - baseline_period
            e = s - 1

            if s >= e:
                s -= 3

            bg_arr = arr[s:e]
            sub_img_stack_values = all_img_stack_values[n]

            fmax = np.nanmax(arr)
            b1 = sub_img_stack_values[0]
            b2 = sub_img_stack_values[-1]+baseline_period
            pltarr = arr[b1 : b2]
            # bgarr = [np.nanmean(bg_arr)]*len(pltarr)
            bgarr = np.nanmean(bg_arr)
            fdiff = np.subtract(pltarr, bgarr)
            fmax -= bgarr

            all_arrs.append(np.divide(fdiff, fmax))
            all_stims.append(n)
            real_arr_inds.append(np.arange(b1, b2))

        f_Dfs = pd.DataFrame({'stimulus' : all_stims, 'fdff' : all_arrs, 'corr_index' : real_arr_inds})
        f_Dfs.loc[:, 'neuron'] = cell
        all_dataframes.append(f_Dfs)

    return pd.concat(all_dataframes)

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def make_predictions(stim_fdff, stimuli, split_frac=0.66, trimStim=False):
    subplot_locations = {'lateral_left': 7, 'lateral_right': 2, 'medial_left': 6, 'medial_right': 3,
                         'left': 5, 'right': 4, 'forward': 1, 'backward': 0}

    if trimStim:
        stimuli = stimuli[stimuli.stim_name.isin(['left', 'right', 'forward', 'backward'])]

    stimDict = {i: stimuli.loc[i].stim_name for i in stimuli.index}
    stim_fdff.loc[:, 'stim_name'] = stim_fdff.stimulus.map(stimDict)
    stim_fdff.loc[:, 'stim_code'] = stim_fdff.stim_name.map(subplot_locations)

    stimCodes = stimuli.stim_name.map(subplot_locations)[:-1]

    newDict = {}
    for stim in stimuli.index.values[:-1]:
        stub = stim_fdff[stim_fdff.stimulus == stim]
        maxVals = np.clip(np.nan_to_num([i.max() for i in stub.fdff.values]), a_min=-1, a_max=1)
        newDict[stim] = maxVals

    new_df = pd.DataFrame(newDict).T

    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    split = int(len(stimuli) * split_frac)
    x_tr = new_df.iloc[:split]
    y_tr = stimCodes[:split]

    x_test = new_df.iloc[split:]
    y_test = stimCodes[split:]

    LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000).fit(x_tr, y_tr)
    lr_score = round(LR.score(x_test, y_test), 4)

    SVM = svm.SVC(class_weight='balanced', gamma='scale').fit(x_tr, y_tr)
    svm_score = round(SVM.score(x_test, y_test), 4)

    RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_tr, y_tr)
    rf_score = round(RF.score(x_test, y_test), 4)

    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10), random_state=1, max_iter=1000).fit(x_tr, y_tr)
    nn_score = round(NN.score(x_test, y_test), 4)

    return pd.DataFrame(
        {"LogisticRegression": [lr_score], "SupportVectorMachine": [svm_score], "RandomForest": [rf_score],
         "PerceptronScore": [nn_score]})