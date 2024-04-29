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





def parse_command(commandIn):
    numPoints = commandIn['numPoints']

    complete_sweep_first = commandIn['complete_sweep_first']
    complete_iterations_first = commandIn['complete_iterations_first']

    #this is an empty list, and similar to the Bruker XML, will be filled with entries, one for each time the laser is
    #powered on
    laser_on_list = []
    #this counter will keep track of the number of times the laser has been turned on
    laser_on_counter = 0

    match commandIn['procedure']:
        case 'galvo':

            convertedXs, convertedYs = [], []
            for i, point in enumerate(commandIn['points']):
                xP, yP = (point[0], point[1])
                convertedXs.append(xP)
                convertedYs.append(yP)

            command = "-MarkPoints"

            numPoints_iter = {'iterator': 0, 'count': commandIn['numPoints']}
            sweeps_iter = {'iterator': 0, 'count': commandIn['numSweeps']}

            # these can both change during a parameter sweep
            iterations_iter = {'iterator': 0, 'count': commandIn['iterations'][0][0]}
            repetitions_iter = {'iterator': 0, 'count': commandIn['repetitions'][0][0]}

            if (complete_sweep_first):
                first_iterator = numPoints_iter
                second_iterator = sweeps_iter
                third_iterator = iterations_iter
                fourth_iterator = repetitions_iter

            elif (complete_iterations_first):
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

                # this will help construc the proper loops for repetitions and iterations
                # worst case scenario we redundantly set it to its current value each loop
                repetitions_iter['count'] = commandIn['repetitions'][
                    sweeps_iter['iterator']][numPoints_iter['iterator']]
                iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                for second_iterator['iterator'] in range(second_iterator['count']):

                    # this will help construc the proper loops for repetitions and iterations
                    repetitions_iter['count'] = commandIn['repetitions'][
                        sweeps_iter['iterator']][numPoints_iter['iterator']]
                    iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                    for third_iterator['iterator'] in range(third_iterator['count']):

                        # this will help construc the proper loops for repetitions and iterations
                        repetitions_iter['count'] = commandIn['repetitions'][
                            sweeps_iter['iterator']][numPoints_iter['iterator']]
                        iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                        for fourth_iterator['iterator'] in range(fourth_iterator['count']):

                            #create a python dictionary for this photostimulation instance
                            #set the photostimulation instance counter to the correct instance of the laser being on
                            photostimulation_instance = {'photostimulation_count' : laser_on_counter, 'photostimulation_type': 'galvo'}
                            laser_on_counter += 1


                            # add X and Y coordinates
                            command += ' ' + str(convertedXs[numPoints_iter['iterator']])
                            command += ' ' + str(convertedYs[numPoints_iter['iterator']])

                            #set the x and y coordinates of the photostimulation instance

                            photostimulation_instance['x_coord'] = convertedXs[numPoints_iter['iterator']]
                            photostimulation_instance['y_coord'] = convertedYs[numPoints_iter['iterator']]

                            # add duration
                            command += ' ' + str(
                                commandIn['durations'][sweeps_iter['iterator']][numPoints_iter['iterator']])

                            #set the duration of the photostimulation instance
                            photostimulation_instance['duration'] = commandIn['durations'][sweeps_iter['iterator']][numPoints_iter['iterator']]

                            # add laser name
                            command += " 'Monaco 1035'"  # + 'num reps {}'.format(repetitions_iter['count']) + ' actual reps {} '.format(repetitions_iter['iterator'])

                            photostimulation_instance['laser_name'] = 'Monaco 1035'

                            # add power
                            command += ' ' + str(
                                commandIn['powers'][sweeps_iter['iterator']][numPoints_iter['iterator']])

                            photostimulation_instance['laser_power_AU'] = commandIn['powers'][sweeps_iter['iterator']][numPoints_iter['iterator']]

                            # add spiral size and revolutions
                            command += ' True ' + str((
                                commandIn['sizes'][sweeps_iter['iterator']][
                                    numPoints_iter['iterator']])) + ' ' + str(
                                commandIn['revolutions'][sweeps_iter['iterator']][numPoints_iter['iterator']])

                            photostimulation_instance['spiral_size']  = (
                                commandIn['sizes'][sweeps_iter['iterator']][
                                    numPoints_iter['iterator']])

                            photostimulation_instance['spiral_revolutions'] = commandIn['revolutions'][sweeps_iter['iterator']][numPoints_iter['iterator']]


                            # add delay before next repetition, point, iteration, etc

                            if (repetitions_iter['iterator'] == repetitions_iter['count'] - 1):
                                # interpoint delay or iteration delay depending on whether to move to a new point or do another iteration

                                if (complete_sweep_first):
                                    if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                        if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                            if (numPoints_iter['iterator'] == numPoints_iter['count'] - 1):
                                                #the points are the last thing iterated through so don't put delay at end
                                                pass
                                                photostimulation_instance['delay_type'] = None
                                            else:
                                                command += ' ' + str(
                                                    commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                        numPoints_iter['iterator']])

                                                photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                        numPoints_iter['iterator']]

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
                                                photostimulation_instance['delay_type'] = None
                                            else:
                                                command += ' ' + str(commandIn['sweep_delay'])

                                                photostimulation_instance['delay'] = commandIn['sweep_delay']
                                                photostimulation_instance['delay_type'] = 'sweep'

                                        else:
                                            command += ' ' + str(
                                                commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                    numPoints_iter['iterator']])

                                            photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                    numPoints_iter['iterator']]
                                            photostimulation_instance['delay_type'] = 'interpoint'


                                    else:
                                        command += ' ' + str(
                                            commandIn['iteration_delays'][sweeps_iter['iterator']][0])

                                        photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                        photostimulation_instance['delay_type'] = 'iteration'

                                else:
                                    if (numPoints_iter['iterator'] == numPoints_iter['count'] - 1):
                                        if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                            if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                                pass
                                                photostimulation_instance['delay_type'] = None
                                            else:
                                                command += ' ' + str(commandIn['sweep_delay'])

                                                photostimulation_instance['delay'] = commandIn['sweep_delay']
                                                photostimulation_instance['delay_type'] = 'sweep'
                                        else:
                                            command += ' ' + str(
                                                commandIn['iteration_delays'][sweeps_iter['iterator']][0])

                                            photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                            photostimulation_instance['delay_type'] = 'iteration'
                                    else:
                                        command += ' ' + str(
                                            commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                numPoints_iter['iterator']])

                                        photostimulation_instance['delay'] = commandIn['interpoint_delays'][sweeps_iter['iterator']][
                                                numPoints_iter['iterator']]


                            # otherwise this is just a regular repetition
                            else:
                                command += ' ' + str(commandIn['repetition_delays'][sweeps_iter['iterator']][
                                                         numPoints_iter['iterator']])

                                photostimulation_instance['delay'] = commandIn['repetition_delays'][sweeps_iter['iterator']][
                                                         numPoints_iter['iterator']]
                                photostimulation_instance['delay_type'] = 'repetition'

                            laser_on_list.append(photostimulation_instance)

        case 'slm-2d' | 'slm-3d':
            convertedXs, convertedYs, convertedZs = [], [], []
            for i, point in enumerate(commandIn['points']):
                xP, yP = (point[0], point[1])
                convertedXs.append(xP)
                convertedYs.append(yP)
                if (commandIn['procedure'] == 'slm-3d'):
                    try:
                        convertedZs.append(point[2])

                    # the user might want "3d" slm but if they specify only x and y we will give them 3d on one plane
                    except IndexError:
                        convertedZs.append(0)

            command = "-MarkAllPoints"

            numPoints_iter = {'iterator': 0, 'count': commandIn['numPoints']}
            sweeps_iter = {'iterator': 0, 'count': commandIn['numSweeps']}

            # these can both change during a parameter sweep
            iterations_iter = {'iterator': 0, 'count': commandIn['iterations'][0][0]}
            repetitions_iter = {'iterator': 0, 'count': commandIn['repetitions'][0][0]}

            if (complete_sweep_first):
                first_iterator = sweeps_iter
                second_iterator = iterations_iter
                third_iterator = repetitions_iter
                fourth_iterator = numPoints_iter

            elif (complete_iterations_first):
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

                # this will help construc the proper loops for repetitions and iterations
                # worst case scenario we redundantly set it to its current value each loop
                repetitions_iter['count'] = commandIn['repetitions'][
                    sweeps_iter['iterator']][0]
                iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                print(iterations_iter, repetitions_iter)

                for second_iterator['iterator'] in range(second_iterator['count']):

                    # this will help construc the proper loops for repetitions and iterations
                    repetitions_iter['count'] = commandIn['repetitions'][
                        sweeps_iter['iterator']][0]
                    iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                    for third_iterator['iterator'] in range(third_iterator['count']):
                        photostimulation_instance = {'photostimulation_count': laser_on_counter}

                        photostimulation_instance['photostimulation_type'] = commandIn['procedure']


                        # this will help construc the proper loops for repetitions and iterations
                        repetitions_iter['count'] = commandIn['repetitions'][
                            sweeps_iter['iterator']][0]
                        iterations_iter['count'] = commandIn['iterations'][sweeps_iter['iterator']][0]

                        command += ' ' + str(numPoints)

                        photostimulation_instance['x_coord'] = []
                        photostimulation_instance['y_coord'] = []
                        laser_on_counter += 1

                        # in the SLM case, the fourth iterator will always be num points
                        for fourth_iterator['iterator'] in range(fourth_iterator['count']):




                            # add X and Y coordinates
                            command += ' ' + str(convertedXs[numPoints_iter['iterator']])
                            command += ' ' + str(convertedYs[numPoints_iter['iterator']])


                            photostimulation_instance['x_coord'].append(convertedXs[numPoints_iter['iterator']])
                            photostimulation_instance['y_coord'].append(convertedYs[numPoints_iter['iterator']])

                            if commandIn['procedure'] == 'slm-3d':
                                command += ' ' + str(convertedZs[numPoints_iter['iterator']])

                        # add duration
                        command += ' ' + str(commandIn['durations'][sweeps_iter['iterator']][0])

                        photostimulation_instance['duration'] = commandIn['durations'][sweeps_iter['iterator']][0]

                        # add laser name
                        command += " 'Monaco 1035'"  # + 'num reps {}'.format(repetitions_iter['count']) + ' actual reps {} '.format(repetitions_iter['iterator'])
                        photostimulation_instance['laser_name'] = 'Monaco 1035'


                        # add power
                        command += ' ' + str(commandIn['powers'][sweeps_iter['iterator']][0])
                        photostimulation_instance['laser_power_AU'] = commandIn['powers'][sweeps_iter['iterator']][0]


                        # add spiral size and revolutions
                        command += ' True ' + str((
                            commandIn['sizes'][sweeps_iter['iterator']][0])) + ' ' + str(
                            commandIn['revolutions'][sweeps_iter['iterator']][0])

                        photostimulation_instance['spiral_size'] = commandIn['sizes'][sweeps_iter['iterator']][0]
                        photostimulation_instance['spiral_revolutions'] = commandIn['revolutions'][sweeps_iter['iterator']][0]


                        # add delay before next repetition, point, iteration, etc

                        if (repetitions_iter['iterator'] == repetitions_iter['count'] - 1):
                            # interpoint delay or iteration delay depending on whether to move to a new point or do another iteration

                            if (complete_sweep_first):
                                if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                    if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                        pass
                                        photostimulation_instance['delay_type'] = None
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
                                        photostimulation_instance['delay_type'] = None
                                    else:
                                        command += ' ' + str(commandIn['sweep_delay'])
                                        photostimulation_instance['delay'] = commandIn['sweep_delay']
                                        photostimulation_instance['delay_type'] = 'sweep'

                                else:
                                    command += ' ' + str(
                                        commandIn['iteration_delays'][sweeps_iter['iterator']][0])

                                    photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                    photostimulation_instance['delay_type'] = 'iteration'

                            else:
                                if (iterations_iter['iterator'] == iterations_iter['count'] - 1):
                                    if (sweeps_iter['iterator'] == sweeps_iter['count'] - 1):
                                        pass
                                        photostimulation_instance['delay_type'] = None
                                    else:
                                        command += ' ' + str(commandIn['sweep_delay'])
                                        photostimulation_instance['delay'] = commandIn['sweep_delay']
                                        photostimulation_instance['delay_type'] = 'sweep'
                                else:
                                    command += ' ' + str(
                                        commandIn['iteration_delays'][sweeps_iter['iterator']][0])

                                    photostimulation_instance['delay'] = commandIn['iteration_delays'][sweeps_iter['iterator']][0]
                                    photostimulation_instance['delay_type'] = 'iteration'


                        # otherwise this is just a regular repetition
                        else:
                            command += ' ' + str(
                                commandIn['repetition_delays'][sweeps_iter['iterator']][0])

                            photostimulation_instance['delay'] = commandIn['repetition_delays'][sweeps_iter['iterator']][0]
                            photostimulation_instance['delay_type'] = 'repetition'

                    laser_on_list.append(photostimulation_instance)

    return (laser_on_list)


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