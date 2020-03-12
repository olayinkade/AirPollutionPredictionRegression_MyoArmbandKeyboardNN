import collections
import math
from input_files_reader import \
    backward_orientation, backward_accelerometer, backward_gyro, backward_timestamps, backward_emg, \
    forward_accelerometer, forward_gyro, forward_orientation, forward_timestamps, forward_emg, \
    left_accelerometer, left_gyro, left_orientation, left_timestamps, left_emg, \
    right_accelerometer, right_gyro, right_orientation, right_timestamps, right_emg, \
    enter_accelerometer, enter_gyro, enter_orientation, enter_timestamps, enter_emg
import tensorflow as tf
import pandas as pd
import copy
import numpy as np
from typing import List


# extract the periods of time with no change in all three axes z, y and z
def extract_nochange_periods(timestamp, x, y, z):
    periods = []
    i = 0
    while i < len(timestamp) - 3:
        if x[i + 1] == x[i] and y[i + 1] == y[i] and z[i + 1] == z[i]:
            periods.append(timestamp[i])
            i += 3
        else:
            i += 1
    return periods


def create_ten_groups(peak: float, timestamps: List, accelerometer: dict) -> List:
    groups = []
    for i in range(len(timestamps) - 1):
        starting_position = accelerometer['timestamp'].index(timestamps[i])
        ending_position = accelerometer['timestamp'].index(timestamps[i + 1])
        while starting_position < ending_position:
            if accelerometer['y'][starting_position] < peak:
                groups.append((timestamps[i], timestamps[i + 1]))
                break
            starting_position += 1

    while len(groups) > 10:  # remove the first element until size is 10
        groups = groups[1:]
    while len(groups) < 10:
        groups.append(groups[len(groups) - 2])

    return groups


# split the data into 10 different groups based on orientation
def extract_data_to_ten_groups(orientation: dict, accelerometer: dict):
    orientation_nochange_timestamps = extract_nochange_periods(orientation['timestamp'], orientation['x'],
                                                               orientation['y'], orientation['z'])
    return create_ten_groups(sorted(accelerometer['y'])[30], orientation_nochange_timestamps, accelerometer)


# based on the calculated 10 groups, calculate the average length of the group based on the number of actual
# timestamps that exist in those ranges
def find_group_average_length_by_num_existing_timestamps(groups: List, timestamps: List) -> int:
    length = 0
    for i in range(len(groups)):
        for j in range(len(timestamps)):
            if float(groups[i][0]) <= float(timestamps[j]) < float(groups[i][1]):
                length += 1
    return int(length / len(groups))


# make sure that all groups have similar distribution and same size
def normalize_groups(groups: List, expected_group_len: int, timestamps: List, given_data: List) -> List:
    normalized_groups = []
    for i in range(len(groups)):
        curr_max, curr_max_index = 0.0, 0
        for j in range(len(timestamps)):
            if float(groups[i][0]) <= float(timestamps[j]) < float(groups[i][1]):
                if float(given_data[j]) > curr_max:
                    curr_max = float(given_data[j])
                    curr_max_index = j
        # make sure the groups have the same size
        normalized_groups.append((timestamps[curr_max_index - int(expected_group_len / 2)],
                                  timestamps[curr_max_index + int(expected_group_len / 2)]))
    return normalized_groups


# this function goes through all the data, makes sure that the necessary groups for processing later have been created.
# In other words, this function makes the data ready for the network.
# info_type being accelerometer, gyro or emg. These are the only supported data types.
# axis being x, y or z for gyro and accelerometer, and emg1, emg2,..., emg8 for emg.
def pre_process_data(info_type: str, axis: str) -> ():
    backward_groups = extract_data_to_ten_groups(backward_orientation, backward_accelerometer)
    forward_groups = extract_data_to_ten_groups(forward_orientation, forward_accelerometer)
    left_groups = extract_data_to_ten_groups(left_orientation, left_accelerometer)
    right_groups = extract_data_to_ten_groups(right_orientation, right_accelerometer)
    enter_groups = extract_data_to_ten_groups(enter_orientation, enter_accelerometer)

    mean_group_length_all_labels = \
        math.floor((find_group_average_length_by_num_existing_timestamps(backward_groups, backward_timestamps) +
                    find_group_average_length_by_num_existing_timestamps(forward_groups, forward_timestamps) +
                    find_group_average_length_by_num_existing_timestamps(left_groups, left_timestamps) +
                    find_group_average_length_by_num_existing_timestamps(right_groups, right_timestamps) +
                    find_group_average_length_by_num_existing_timestamps(enter_groups, enter_timestamps)) / 5)

    if info_type == 'gyro':
        backward_groups = normalize_groups(backward_groups, mean_group_length_all_labels,
                                           backward_timestamps, backward_gyro[axis])
        forward_groups = normalize_groups(forward_groups, mean_group_length_all_labels,
                                          forward_timestamps, forward_gyro[axis])
        left_groups = normalize_groups(left_groups, mean_group_length_all_labels, left_timestamps, left_gyro[axis])
        right_groups = normalize_groups(right_groups, mean_group_length_all_labels, right_timestamps, right_gyro[axis])
        enter_groups = normalize_groups(enter_groups, mean_group_length_all_labels, enter_timestamps, enter_gyro[axis])
    elif info_type == 'accelerometer':
        backward_groups = normalize_groups(backward_groups, mean_group_length_all_labels,
                                           backward_timestamps, backward_accelerometer[axis])
        forward_groups = normalize_groups(forward_groups, mean_group_length_all_labels,
                                          forward_timestamps, forward_accelerometer[axis])
        left_groups = normalize_groups(left_groups, mean_group_length_all_labels, left_timestamps,
                                       left_accelerometer[axis])
        right_groups = normalize_groups(right_groups, mean_group_length_all_labels, right_timestamps,
                                        right_accelerometer[axis])
        enter_groups = normalize_groups(enter_groups, mean_group_length_all_labels, enter_timestamps,
                                        enter_accelerometer[axis])
    elif info_type == 'emg':
        backward_groups = normalize_groups(backward_groups, mean_group_length_all_labels,
                                           backward_timestamps, backward_emg[axis])
        forward_groups = normalize_groups(forward_groups, mean_group_length_all_labels,
                                          forward_timestamps, forward_emg[axis])
        left_groups = normalize_groups(left_groups, mean_group_length_all_labels, left_timestamps, left_emg[axis])
        right_groups = normalize_groups(right_groups, mean_group_length_all_labels, right_timestamps, right_emg[axis])
        enter_groups = normalize_groups(enter_groups, mean_group_length_all_labels, enter_timestamps, enter_emg[axis])
    return backward_groups, forward_groups, left_groups, right_groups, enter_groups, mean_group_length_all_labels


# get the actual data in the right data set based on calculated start and end times of groups,
# and put them in pre-defined groups
def get_data_by_groups(groups: List[tuple], info_type: str, axis: str, given_data: dict, timestamps: List) -> List:
    actual_data = []
    if info_type == 'gyro':
        for i in range(len(groups)):
            actual_data.append(given_data[axis][timestamps.index(groups[i][0]): timestamps.index(groups[i][1])])
    elif info_type == 'accelerometer':
        for i in range(len(groups)):
            actual_data.append(given_data[axis][timestamps.index(groups[i][0]): timestamps.index(groups[i][1])])
    elif info_type == 'emg':
        given_data_timestamp = copy.deepcopy(given_data['timestamp'])
        for i in range(len(groups)):
            clostest_start_time = min(map(float, given_data_timestamp), key=lambda x: abs(x - float(groups[i][0])))
            clostest_end_time = min(map(float, given_data_timestamp), key=lambda x: abs(x - float(groups[i][1])))
            actual_data.append(given_data[axis][given_data['timestamp'].index(str(int(clostest_start_time))):
                                                  given_data['timestamp'].index(str(int(clostest_end_time)))])
    return actual_data


# makes sure that all groups have the same length
def standardize_groups_by_length(data: List, standard_length: int) -> List:
    standardized_groups = copy.deepcopy(data)
    for i in range(len(standardized_groups)):
        difference = standard_length - len(standardized_groups[i])
        if difference < 0:
            while len(standardized_groups[i]) != standard_length:
                standardized_groups[i].pop(-1)
        elif difference > 0:
            while len(standardized_groups[i]) != standard_length:
                standardized_groups[i].append('0.0')
    return standardized_groups


# given a type of the data that will be processed and a chosen axis, this will extract the data and return it to be
# processed by the neural network.
def process_single_axis(info_type: str, axis: str):
    result, data, labels = [], [], []  # labels being the expected/ground truth output
    backward_groups, forward_groups, left_groups, right_groups, enter_groups, mean_group_length_all_labels \
        = pre_process_data(info_type, axis)
    backward_chosen_data, forward_chosen_data = [], []
    left_chosen_data, right_chosen_data, enter_chosen_data = [], [], []
    print('Getting {} data of axis {}...'.format(info_type, axis))

    if info_type == 'gyro':
        backward_chosen_data = get_data_by_groups(backward_groups, info_type, axis, backward_gyro,
                                                  backward_timestamps)
        forward_chosen_data = get_data_by_groups(forward_groups, info_type, axis, forward_gyro, forward_timestamps)
        left_chosen_data = get_data_by_groups(left_groups, info_type, axis, left_gyro, left_timestamps)
        right_chosen_data = get_data_by_groups(right_groups, info_type, axis, right_gyro, right_timestamps)
        enter_chosen_data = get_data_by_groups(enter_groups, info_type, axis, enter_gyro, enter_timestamps)
    elif info_type == 'accelerometer':
        backward_chosen_data = get_data_by_groups(backward_groups, info_type, axis, backward_accelerometer,
                                                  backward_timestamps)
        forward_chosen_data = get_data_by_groups(forward_groups, info_type, axis, forward_accelerometer,
                                                 forward_timestamps)
        left_chosen_data = get_data_by_groups(left_groups, info_type, axis, left_accelerometer, left_timestamps)
        right_chosen_data = get_data_by_groups(right_groups, info_type, axis, right_accelerometer, right_timestamps)
        enter_chosen_data = get_data_by_groups(enter_groups, info_type, axis, enter_accelerometer, enter_timestamps)
    elif info_type == 'emg':
        backward_chosen_data = get_data_by_groups(backward_groups, info_type, axis, backward_emg, backward_timestamps)
        forward_chosen_data = get_data_by_groups(forward_groups, info_type, axis, forward_emg, forward_timestamps)
        left_chosen_data = get_data_by_groups(left_groups, info_type, axis, left_emg, left_timestamps)
        right_chosen_data = get_data_by_groups(right_groups, info_type, axis, right_emg, right_timestamps)
        enter_chosen_data = get_data_by_groups(enter_groups, info_type, axis, enter_emg, enter_timestamps)

    for i in range(len(backward_chosen_data)):
        data.append(forward_chosen_data[i])
        data.append(backward_chosen_data[i])
        data.append(left_chosen_data[i])
        data.append(right_chosen_data[i])
        data.append(enter_chosen_data[i])
        labels.append([1, 0, 0, 0, 0])
        labels.append([0, 1, 0, 0, 0])
        labels.append([0, 0, 1, 0, 0])
        labels.append([0, 0, 0, 1, 0])
        labels.append([0, 0, 0, 0, 1])

    data = standardize_groups_by_length(data, mean_group_length_all_labels)

    for i in range(len(data)):
        # first item of a pair is the data, second item is the correct label
        pair = [list(map(float, data[i])), labels[i]]
        result.append(tuple(pair))

    return result, labels


# With multiple axes, we will process all axes of the given info type. Multiple data types are acceptable as well.
def process_multi_axes(info_types: List[str]):
    print('Getting data from all three axes x, y and z...')
    result = []
    all_combinations = collections.defaultdict(list)
    for info_type in info_types:
        if info_type == 'accelerometer' or info_type == 'gyro':
            data_x = process_single_axis(info_type, 'x')[0]
            data_y = process_single_axis(info_type, 'y')[0]
            data_z, labels = process_single_axis(info_type, 'z')
            all_combinations[info_type] = []
            for i in range(len(data_x)):
                all_combinations[info_type].append(data_x[i][0] + data_y[i][0] + data_z[i][0])
            all_combinations['label'] = labels

        elif info_type == 'emg':
            data_1 = process_single_axis(info_type, 'emg1')[0]
            data_2 = process_single_axis(info_type, 'emg2')[0]
            data_3 = process_single_axis(info_type, 'emg3')[0]
            data_4 = process_single_axis(info_type, 'emg4')[0]
            data_5 = process_single_axis(info_type, 'emg5')[0]
            data_6 = process_single_axis(info_type, 'emg6')[0]
            data_7 = process_single_axis(info_type, 'emg7')[0]
            data_8, labels = process_single_axis(info_type, 'emg8')
            all_combinations[info_type] = []
            for i in range(len(data_1)):
                all_combinations[info_type].append(data_1[i][0] + data_2[i][0] + data_3[i][0] + data_4[i][0] +
                                                   data_5[i][0] + data_6[i][0] + data_7[i][0] + data_8[i][0])
            all_combinations['label'] = labels

    # group the data to prepare for the input layer
    combinations = []
    for i in range(len(all_combinations['label'])):
        curr_row = []
        for key in all_combinations.keys():
            if key != 'label':
                curr_row += all_combinations[key][i]
        combinations.append(curr_row)

    for i in range(len(combinations)):
        result.append([combinations[i], all_combinations['label'][i]])
    return result
