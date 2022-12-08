try:
    import caiman as cm
    from caiman.motion_correction import MotionCorrect
except ModuleNotFoundError:
    print('caiman unavailable')

try:
    import suite2p
    from suite2p.run_s2p import run_s2p, default_ops
except ModuleNotFoundError:
    print('suite2p unavailable')

import os

from old_stuff.ImageAnalysisCodes import utils

import numpy as np
import pandas as pd

pd.set_option('mode.chained_assignment', None)

from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def movement_correction(_path, saveMoveCorrect=True, saveEls=False, volume=False):

    try:
        paths = utils.pathSorter(_path)

        imagePath = paths['image']['raw']
        try:
            paramsPath = paths['etc']['moveCorrectionParams']
        except KeyError:
            paramsPath = None
    except NotADirectoryError:
        imagePath = _path

    print('performing movement correction on: ', imagePath)
    # movement correction parameters
    try:
        if paramsPath is None:
            max_shifts = (3, 3)
            strides = (25, 25)
            overlaps = (15, 15)
            num_frames_split = 150
            max_deviation_rigid = 3
            pw_rigid = False
            shifts_opencv = True
            border_nan = 'copy'
            downsample_ratio = .2
        else:
            with open(paramsPath) as file:
                params = eval(file.read())
            try:
                max_shifts = params['max_shifts']
            except KeyError:
                print('defaulting max_shifts')
                max_shifts = (3, 3)
            try:
                strides = params['strides']
            except KeyError:
                print('defaulting strides')
                strides = (25, 25)
            try:
                overlaps = params['overlaps']
            except KeyError:
                print('defaulting overlaps')
                overlaps = (15, 15)
            try:
                num_frames_split = params['num_frames_split']
            except KeyError:
                print('defaulting num_frames_split')
                num_frames_split = 150
            try:
                max_deviation_rigid = params['max_deviation_rigid']
            except KeyError:
                print('defaulting max_deviation_rigid')
                max_deviation_rigid = 3
            try:
                pw_rigid = params['pw_rigid']
            except KeyError:
                print('defaulting pw_rigid')
                pw_rigid = False
            try:
                shifts_opencv = params['shifts_opencv']
            except KeyError:
                print('defaulting shifts_opencv')
                shifts_opencv = True
            try:
                border_nan = params['border_nan']
            except KeyError:
                print('defaulting border_nan')
                border_nan = 'copy'
            try:
                downsample_ratio = params['downsample_ratio']
            except KeyError:
                print('defaulting downsample_ratio')
                downsample_ratio = .2
    except UnboundLocalError:
        max_shifts = (3, 3)
        strides = (25, 25)
        overlaps = (15, 15)
        num_frames_split = 150
        max_deviation_rigid = 3
        pw_rigid = False
        shifts_opencv = True
        border_nan = 'copy'
        downsample_ratio = .2

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    try:
        dview.terminate()
    except:
        pass
    try:
        cm.stop_server()
    except:
        pass

    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=6, single_thread=False)

    mc = MotionCorrect([imagePath.as_posix()], dview=dview, max_shifts=max_shifts,
                       strides=strides, overlaps=overlaps,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=shifts_opencv, nonneg_movie=True,
                       border_nan=border_nan, is3D=False)

    mc.motion_correct(save_movie=True)
    m_rig = cm.load(mc.mmap_file)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)
    mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
    mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
    mc.motion_correct(save_movie=True, template=mc.total_template_rig)
    m_els = cm.load(mc.fname_tot_els)

    imagePathFolder = Path(imagePath).parents[0]
    if not saveEls:
        with os.scandir(imagePathFolder) as entries:
            for entry in entries:
                if entry.is_file():
                    if entry.name.endswith('.mmap'):
                        os.remove(entry)

    if saveMoveCorrect and not volume:
        savingFolder = imagePathFolder.joinpath('move_corrected')
        savingPath = savingFolder.joinpath('mve_' + Path(imagePath).name)
        try:
            os.mkdir(savingFolder)
        except FileExistsError:
            val_offset = 0
            while os.path.exists(savingPath):
                stem = savingPath.stem
                savingPath = savingFolder.joinpath(f"{stem}_{val_offset}.tif")
                val_offset += 1

        import tifffile
        outputimg = m_els.resize(1, 1)
        tifffile.imsave(savingPath, outputimg)
        print("saved at: ", savingPath)

    elif saveMoveCorrect and volume:

        savingFolder = imagePathFolder.joinpath('move_corrected')

        if not os.path.exists(savingFolder):
            os.mkdir(savingFolder)

        plane = Path(imagePath).stem
        savingFolderVolume = savingFolder.joinpath(f'plane_{plane}')

        if not os.path.exists(savingFolderVolume):
            os.mkdir(savingFolderVolume)

        import tifffile
        savingPath = savingFolderVolume.joinpath(f'plane_{plane}_mve.tif')
        outputimg = m_els.resize(1, 1)
        tifffile.imsave(savingPath, outputimg)

        print("saved at: ", savingPath)

    else:
        pass


    cm.stop_server(dview=dview)

    return m_els.resize(1, 1)


def neuron_extraction(_path, imageFrequency=None, s2p_ops=None, useRaw=False, input_tau=1.5):
    # normally use movement corrected image
    # this accepts the path to the notes file or an input image frequency
    paths = utils.pathSorter(_path)
    print('meow')
    try:
        imagePath = paths['image']['volume']
        volumes = True
    except KeyError:
        volumes = False
        pass
    if not volumes:
        try:
            try:
                imagePath = paths['image']['move_corrected']
            except KeyError:
                if not useRaw:
                    print('no movement corrected image found')
                    movement_correction(_path)
                    paths = utils.pathSorter(_path)
                    imagePath = paths['image']['move_corrected']
                else:
                    imagePath = paths['image']['raw']

            if s2p_ops is None:
                s2p_ops = {'data_path': [imagePath.parents[0].as_posix()], 'save_path0': imagePath.parents[0].as_posix(),
                           'tau': input_tau, 'pad_fft': True, 'force_refImg': False, 'do_bidiphase':True, "preclassify" : 0.15,
                          "smooth_sigma_time" : 0, "two_step_registration": True, "keep_movie_raw" : True, 'threshold_scaling' : 0.5, "allow_overlap" : True, 'high_pass' : 75, 'min_neuropil_pixels':100, 'max_iterations':50, 'sparse_mode' : False,
                           'block_size' : [50,50]}
            else:
                s2p_ops['data_path'] = [imagePath.parents[0].as_posix()]
                s2p_ops['save_path0'] = imagePath.parents[0].as_posix()
            try:
                notesPath = paths['etc']['notes']
                _notes = pd.read_csv(notesPath, sep=':', header=None)
                imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                s2p_ops['fs'] = 1000/imageHz

            except KeyError:
                if imageFrequency is not None:
                    s2p_ops['fs'] = imageFrequency
                else:
                    print('defaulting to 2 frames per second')
                    s2p_ops['fs'] = 2.0

            if imageFrequency is not None:
                s2p_ops['fs'] = imageFrequency

            ops = default_ops()
            db = {}
            for item in s2p_ops:
                ops[item] = s2p_ops[item]
            output_ops = run_s2p(ops=ops, db=db)

        except ValueError:
            # sometimes this angryboi needs to go twice
            print('rerunning')
            try:
                imagePath = paths['image']['move_corrected']
            except KeyError:
                if not useRaw:
                    print('no movement corrected image found')
                    movement_correction(_path)
                    paths = utils.pathSorter(_path)
                    imagePath = paths['image']['move_corrected']
                else:
                    imagePath = paths['image']['raw']


                if s2p_ops is None:
                    s2p_ops = {'data_path': [imagePath.parents[0].as_posix()],
                               'save_path0': imagePath.parents[0].as_posix(),
                               'tau': input_tau, 'pad_fft': True, 'force_refImg': True, 'do_bidiphase': True,
                               "preclassify": 0.45,
                               "smooth_sigma": 2, "smooth_sigma_time": 1, "two_step_registration": True,
                               "keep_movie_raw": True}
                else:
                    s2p_ops['data_path'] = [imagePath.parents[0].as_posix()]
                    s2p_ops['save_path0'] = imagePath.parents[0].as_posix()
                try:
                    notesPath = paths['etc']['notes']
                    _notes = pd.read_csv(notesPath, sep=':', header=None)
                    imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                    s2p_ops['fs'] = imageHz

                except KeyError:
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
                    else:
                        print('defaulting to 2 frames per second')
                        s2p_ops['fs'] = 2.0

                if imageFrequency is not None:
                    s2p_ops['fs'] = imageFrequency

                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]

                ops['do_registration'] = 0
                ops['force_refImg'] = 0
                ops['two_step_registration'] = 0
                output_ops = run_s2p(ops=ops, db=db)
    else:
        for p in imagePath:
            print(f'Running {p}')
            used_path = imagePath[p].as_posix()
            try:
                if s2p_ops is None:
                    s2p_ops = {'data_path': [used_path],
                               'save_path0': used_path,
                               'tau': 1.5, 'pad_fft': False, 'force_refImg': True, 'do_bidiphase':False, "preclassify" : 0.45,
                               "smooth_sigma" : 2, "smooth_sigma_time" : 1, "two_step_registration": True, "keep_movie_raw" : True, 'sparse_mode' : True}
                else:
                    s2p_ops['data_path'] = [used_path]
                    s2p_ops['save_path0'] = used_path
                try:
                    notesPath = paths['etc']['notes']
                    _notes = pd.read_csv(notesPath, sep=':', header=None)
                    imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                    s2p_ops['fs'] = imageHz

                except KeyError:
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
                    else:
                        print('defaulting to 2 frames per second')
                        s2p_ops['fs'] = 2.0

                if imageFrequency is not None:
                    s2p_ops['fs'] = imageFrequency

                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]
                output_ops = run_s2p(ops=ops, db=db)

            except ValueError:
                print(f'rerunning {p} because angry')
                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]
                ops['pad_fft'] = True
                ops["two_step_registration"] = True
                ops['force_refImg'] = True
                ops['sparse_mode'] = False
                ops['do_bidiphase'] = False
                output_ops = run_s2p(ops=ops, db=db)
                paths = utils.pathSorter(_path)
    """
    try:
        paths = utils.pathSorter(_path)

        try:
            imagePath = paths['image']['volume']
            volumes=True
        except KeyError:
            volumes=False
            pass

        if not volumes:
            try:
                imagePath = paths['image']['move_corrected']
            except KeyError:
                if not useRaw:
                    print('no movement corrected image found')
                    movement_correction(_path)
                    paths = utils.pathSorter(_path)
                    imagePath = paths['image']['move_corrected']
                else:
                    imagePath = paths['image']['raw']


                if s2p_ops is None:
                    s2p_ops = {'data_path': [imagePath.parents[0].as_posix()], 'save_path0': imagePath.parents[0].as_posix(),
                               'tau': 1.5, 'pad_fft': True, 'force_refImg': True}
                else:
                    s2p_ops['data_path'] = [imagePath.parents[0].as_posix()]
                    s2p_ops['save_path0'] = imagePath.parents[0].as_posix()
                try:
                    notesPath = paths['etc']['notes']
                    _notes = pd.read_csv(notesPath, sep=':', header=None)
                    imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                    s2p_ops['fs'] = imageHz

                except KeyError:
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
                    else:
                        print('defaulting to 2 frames per second')
                        s2p_ops['fs'] = 2.0

                if imageFrequency is not None:
                    s2p_ops['fs'] = imageFrequency

                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]
                output_ops = run_s2p(ops=ops, db=db)

        else:
            for p in imagePath:
                try:
                    print(f'Running {p}')
                    used_path = imagePath[p].as_posix()
    
                    if s2p_ops is None:
                        s2p_ops = {'data_path': [used_path],
                                   'save_path0': used_path,
                                   'tau': 1.5, 'pad_fft': True, 'force_refImg': True}
                    else:
                        s2p_ops['data_path'] = [used_path]
                        s2p_ops['save_path0'] = used_path
                    try:
                        notesPath = paths['etc']['notes']
                        _notes = pd.read_csv(notesPath, sep=':', header=None)
                        imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                        s2p_ops['fs'] = imageHz
    
                    except KeyError:
                        if imageFrequency is not None:
                            s2p_ops['fs'] = imageFrequency
                        else:
                            print('defaulting to 2 frames per second')
                            s2p_ops['fs'] = 2.0
    
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
    
                    ops = default_ops()
                    db = {}
                    for item in s2p_ops:
                        ops[item] = s2p_ops[item]
                    output_ops = run_s2p(ops=ops, db=db)

            except ValueError:
                # sometimes this angryboi needs to go twice

        
        try:
            imagePath = paths['image']['volume']
            volumes=True
        except KeyError:
            volumes=False
            pass

        if not volumes:
            try:
                imagePath = paths['image']['move_corrected']
            except KeyError:
                if not useRaw:
                    print('no movement corrected image found')
                    movement_correction(_path)
                    paths = utils.pathSorter(_path)
                    imagePath = paths['image']['move_corrected']
                else:
                    imagePath = paths['image']['raw']

                if s2p_ops is None:
                    s2p_ops = {'data_path': [imagePath.parents[0].as_posix()], 'save_path0': imagePath.parents[0].as_posix(),
                               'tau': 1.5, 'pad_fft': True, 'force_refImg': True}
                else:
                    s2p_ops['data_path'] = [imagePath.parents[0].as_posix()]
                    s2p_ops['save_path0'] = imagePath.parents[0].as_posix()
                try:
                    notesPath = paths['etc']['notes']
                    _notes = pd.read_csv(notesPath, sep=':', header=None)
                    imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                    s2p_ops['fs'] = imageHz

                except KeyError:
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
                    else:
                        print('defaulting to 2 frames per second')
                        s2p_ops['fs'] = 2.0

                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]
                output_ops = run_s2p(ops=ops, db=db)
        else:
            for p in imagePath:
                used_path = imagePath[p].as_posix()

                if s2p_ops is None:
                    s2p_ops = {'data_path': [used_path],
                               'save_path0': used_path,
                               'tau': 1.5, 'pad_fft': True, 'force_refImg': True}
                else:
                    s2p_ops['data_path'] = [used_path]
                    s2p_ops['save_path0'] = used_path
                try:
                    notesPath = paths['etc']['notes']
                    _notes = pd.read_csv(notesPath, sep=':', header=None)
                    imageHz = int(_notes[_notes[0] == 'timeperframe (ms)'][1].values[0])
                    s2p_ops['fs'] = imageHz

                except KeyError:
                    if imageFrequency is not None:
                        s2p_ops['fs'] = imageFrequency
                    else:
                        print('defaulting to 2 frames per second')
                        s2p_ops['fs'] = 2.0

                if imageFrequency is not None:
                    s2p_ops['fs'] = imageFrequency

                ops = default_ops()
                db = {}
                for item in s2p_ops:
                    ops[item] = s2p_ops[item]
                output_ops = run_s2p(ops=ops, db=db)
               """
    return


def eigenValues(_path, factors=10):

    from factor_analyzer import FactorAnalyzer

    save_path = Path(_path).joinpath('eigens.npz')

    paths = utils.pathSorter(_path)

    s2pPaths = paths['output']['suite2p']

    iscell = np.load(s2pPaths['iscell'], allow_pickle=True)[:, 0].astype(bool)
    f_cells = np.load(s2pPaths['f_cells'])

    df = pd.DataFrame(f_cells[iscell]).T
    x = df.copy()
    x.replace([np.inf, -np.inf], np.nan)
    x.dropna(inplace=True, axis=1)

    means = x.describe().loc['mean']

    x.drop(columns=means.index[np.where(means == 0)[0]], inplace=True)

    # normalize data -- shouldnt be by mean. Should probably Z-score this
    for col in x.columns:
        x.loc[:, col] /= np.mean(np.sort(x.loc[:, col].values)[:30])

    # fa = FactorAnalyzer()
    # fa.fit(x, factors)
    # ev, v = fa.get_eigenvalues()

    fa = FactorAnalyzer(factors, rotation='varimax')
    fa.fit(x)
    loads = fa.loadings_
    new_variables = fa.fit_transform(x)

    np.savez(save_path, [new_variables, loads, x])
    return

