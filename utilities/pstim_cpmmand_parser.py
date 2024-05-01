import os
from pathlib import Path
import pandas as pd
import numpy as np
from tifffile import imread, imwrite
from datetime import datetime as dt, timedelta
import xml.etree.ElementTree as ET
import json
import caiman as cm
from PIL import Image
import math
from scipy.signal import find_peaks

from bruker_images import read_xml_to_str, read_xml_to_root
from utilities import arrutils
from utilities.roiutils import create_circular_mask
from utilities.coordutils import rotate_transform_coors, closest_coordinates

import sys



def parse_command(commandIn):

        numPoints = commandIn['numPoints']
        complete_sweep_first = commandIn['complete_sweep_first']
        complete_iterations_first = commandIn['complete_iterations_first']
        laser_on_list = []
        laser_on_counter = 0

        procedure = commandIn['procedure']
        if procedure == 'galvo':
            convertedXs, convertedYs = [], []
            for i, point in enumerate(commandIn['points']):
                xP, yP = (point[0], point[1])
                convertedXs.append(xP)
                convertedYs.append(yP)

            command = "-MarkPoints"
            numPoints_iter = {'iterator': 0, 'count': commandIn['numPoints']}
            sweeps_iter = {'iterator': 0, 'count': commandIn['numSweeps']}
            iterations_iter = {'iterator': 0, 'count': commandIn['iterations'][0][0]}
            repetitions_iter = {'iterator': 0, 'count': commandIn['repetitions'][0][0]}

            if complete_sweep_first:
                first_iterator = numPoints_iter
                second_iterator = sweeps_iter
                third_iterator = iterations_iter
                fourth_iterator = repetitions_iter
            elif complete_iterations_first:
                first_iterator = sweeps_iter
                second_iterator = numPoints_iter
                third_iterator = iterations_iter
                fourth_iterator = repetitions_iter
            else:
                first_iterator = sweeps_iter
                second_iterator = iterations_iter
                third_iterator = numPoints_iter
                fourth_iterator = repetitions_iter

            for first_iterator['iterator'] in range(first_iterator['count']):
                repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                for second_iterator['iterator'] in range(second_iterator['count']):
                    repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                    iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                    for third_iterator['iterator'] in range(third_iterator['count']):
                        repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                        iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                        for fourth_iterator['iterator'] in range(fourth_iterator['count']):
                            photostimulation_instance = {'photostimulation_count' : laser_on_counter, 'photostimulation_type': 'galvo'}
                            laser_on_counter += 1
                            command += ' ' + str(convertedXs[numPoints_iter['iterator']])
                            command += ' ' + str(convertedYs[numPoints_iter['iterator']])

                            photostimulation_instance['x_coord'] = convertedXs[numPoints_iter['iterator']]
                            photostimulation_instance['y_coord'] = convertedYs[numPoints_iter['iterator']]

                            command += ' ' + str(commandIn['durations'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                            photostimulation_instance['duration'] = commandIn['durations'][sweeps_iter['iterator']][numPoints_iter['iterator']]

                            command += " 'Monaco 1035'"
                            photostimulation_instance['laser_name'] = 'Monaco 1035'

                            command += ' ' + str(commandIn['powers'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                            photostimulation_instance['laser_power_AU'] = commandIn['powers'][sweeps_iter['iterator']][numPoints_iter['iterator']]

                            command += ' True ' + str((commandIn['sizes'][sweeps_iter['iterator']][numPoints_iter['iterator']])) + ' ' + str(commandIn['revolutions'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                            photostimulation_instance['spiral_size'] = commandIn['sizes'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                            photostimulation_instance['spiral_revolutions'] = commandIn['revolutions'][sweeps_iter['iterator']][numPoints_iter['iterator']]

                            if (repetitions_iter['iterator'] == repetitions_iter['count'] - 1):
                                if (complete_sweep_first):
                                    if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                        if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                            if (numPoints_iter['iterator'] == numPoints_iter['count'] - 1):
                                                pass
                                            else:
                                                command += ' ' + str(commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                                                photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                                                photostimulation_instance['delay_type'] = 'interpoint'
                                        else:
                                            command += ' ' + str(commandIn['sweep_delay'])
                                            photostimulation_instance['delay'] = commandIn['sweep_delay']
                                            photostimulation_instance['delay_type'] = 'sweep'
                                    else:
                                        command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                        photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                        photostimulation_instance['delay_type'] = 'iteration'
                                elif (complete_iterations_first):
                                    if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                        if (numPoints_iter['iterator'] == numPoints_iter['count'] - 1):
                                            if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                                pass
                                            else:
                                                command += ' ' + str(commandIn['sweep_delay'])
                                                photostimulation_instance['delay'] = commandIn['sweep_delay']
                                                photostimulation_instance['delay_type'] = 'sweep'
                                        else:
                                            command += ' ' + str(commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                                            photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                                            photostimulation_instance['delay_type'] = 'interpoint'
                                    else:
                                        command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                        photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                        photostimulation_instance['delay_type'] = 'iteration'
                                else:
                                    if (numPoints_iter['iterator'] == numPoints_iter['count'] - 1):
                                        if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                            if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                                pass
                                            else:
                                                command += ' ' + str(commandIn['sweep_delay'])
                                                photostimulation_instance['delay'] = commandIn['sweep_delay']
                                                photostimulation_instance['delay_type'] = 'sweep'
                                        else:
                                            command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                            photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                            photostimulation_instance['delay_type'] = 'iteration'
                                    else:
                                        command += ' ' + str(commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                                        photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                            else:
                                command += ' ' + str(commandIn['repetition_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']])
                                photostimulation_instance['delay'] = commandIn['repetition_delays'][sweeps_iter['iterator']][numPoints_iter['iterator']]
                                photostimulation_instance['delay_type'] = 'repetition'
                            laser_on_list.append(photostimulation_instance)
        elif procedure in ['slm-2d', 'slm-3d']:
            convertedXs, convertedYs, convertedZs = [], [], []
            for i, point in enumerate(commandIn['points']):
                xP, yP = (point[0], point[1])
                convertedXs.append(xP)
                convertedYs.append(yP)
                if procedure == 'slm-3d':
                    try:
                        convertedZs.append(point[2])
                    except IndexError:
                        convertedZs.append(0)
            command = "-MarkAllPoints"
            numPoints_iter = {'iterator': 0, 'count': commandIn['numPoints']}
            sweeps_iter = {'iterator': 0, 'count': commandIn['numSweeps']}
            iterations_iter = {'iterator': 0, 'count': commandIn['iterations'][0][0]}
            repetitions_iter = {'iterator': 0, 'count': commandIn['repetitions'][0][0]}

            if complete_sweep_first:
                first_iterator = sweeps_iter
                second_iterator = iterations_iter
                third_iterator = repetitions_iter
                fourth_iterator = numPoints_iter
            elif complete_iterations_first:
                first_iterator = sweeps_iter
                second_iterator = iterations_iter
                third_iterator = repetitions_iter
                fourth_iterator = numPoints_iter
            else:
                first_iterator = sweeps_iter
                second_iterator = iterations_iter
                third_iterator = repetitions_iter
                fourth_iterator = numPoints_iter

            for first_iterator['iterator'] in range(first_iterator['count']):
                repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][0]
                iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                for second_iterator['iterator'] in range(second_iterator['count']):
                    repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][0]
                    iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                    for third_iterator['iterator'] in range(third_iterator['count']):
                        repetitions_iter['count'] = commandIn['repetitions'][sweeps_iter['iterator']][0]
                        iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                        photostimulation_instance = {'photostimulation_count': laser_on_counter, 'photostimulation_type': procedure}
                        command += ' ' + str(numPoints)
                        photostimulation_instance['x_coord'] = []
                        photostimulation_instance['y_coord'] = []
                        laser_on_counter += 1

                        for fourth_iterator['iterator'] in range(fourth_iterator['count']):
                            command += ' ' + str(convertedXs[numPoints_iter['iterator']])
                            command += ' ' + str(convertedYs[numPoints_iter['iterator']])
                            photostimulation_instance['x_coord'].append(convertedXs[numPoints_iter['iterator']])
                            photostimulation_instance['y_coord'].append(convertedYs[numPoints_iter['iterator']])
                            if procedure == 'slm-3d':
                                command += ' ' + str(convertedZs[numPoints_iter['iterator']])

                        command += ' ' + str(commandIn['durations'][sweeps_iter['iterator']][0])
                        photostimulation_instance['duration'] = commandIn['durations'][sweeps_iter['iterator']][0]

                        command += " 'Monaco 1035'"
                        photostimulation_instance['laser_name'] = 'Monaco 1035'

                        command += ' ' + str(commandIn['powers'][sweeps_iter['iterator']][0])
                        photostimulation_instance['laser_power_AU'] = commandIn['powers'][sweeps_iter['iterator']][0]

                        command += ' True ' + str((commandIn['sizes'][sweeps_iter['iterator']][0])) + ' ' + str(commandIn['revolutions'][sweeps_iter['iterator']][0])
                        photostimulation_instance['spiral_size'] = commandIn['sizes'][sweeps_iter['iterator']][0]
                        photostimulation_instance['spiral_revolutions'] = commandIn['revolutions'][sweeps_iter['iterator']][0]

                        if (repetitions_iter['iterator'] == repetitions_iter['count'] - 1):
                            if (complete_sweep_first):
                                if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                    if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                        pass
                                    else:
                                        command += ' ' + str(commandIn['sweep_delay'])
                                        photostimulation_instance['delay'] = commandIn['sweep_delay']
                                        photostimulation_instance['delay_type'] = 'sweep'
                                else:
                                    command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                    photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                    photostimulation_instance['delay_type'] = 'iteration'
                            elif (complete_iterations_first):
                                if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                    if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                        pass
                                    else:
                                        command += ' ' + str(commandIn['sweep_delay'])
                                        photostimulation_instance['delay'] = commandIn['sweep_delay']
                                        photostimulation_instance['delay_type'] = 'sweep'
                                else:
                                    command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                    photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                    photostimulation_instance['delay_type'] = 'iteration'
                            else:
                                if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                    if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                        pass
                                    else:
                                        command += ' ' + str(commandIn['sweep_delay'])
                                        photostimulation_instance['delay'] = commandIn['sweep_delay']
                                        photostimulation_instance['delay_type'] = 'sweep'
                                else:
                                    command += ' ' + str(commandIn['iteration_delays'][sweeps_iter['iterator']][0])
                                    photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                    photostimulation_instance['delay_type'] = 'iteration'
                        else:
                            command += ' ' + str(commandIn['repetition_delays'][sweeps_iter['iterator']][0])
                            photostimulation_instance['delay'] = commandIn['repetition_delays'][sweeps_iter['iterator']][0]
                            photostimulation_instance['delay_type'] = 'repetition'
                        laser_on_list.append(photostimulation_instance)
        return laser_on_list



if __name__ == '__main__':
    data = json.load(open("/media/gromit/124d7bfb-0e91-4cf0-8c38-dc3142188881/Binblows-Share/Tyler-Style Sweep/16 March/TSeries-03162024-1620-001/photostim_info.json"))
    summary_command = data["summary_command"]
    command_sent = data["command"]

    photostim_record = parse_command(summary_command)


    photostim_block_indices = []

    addNextRepetition = True
    for record in photostim_record:
        if record['delay_type'] == 'repetition':
            if(addNextRepetition):
                photostim_block_indices.append(record['photostimulation_count'])
                addNextRepetition = False
        else:
            addNextRepetition = True

    print(photostim_block_indices)