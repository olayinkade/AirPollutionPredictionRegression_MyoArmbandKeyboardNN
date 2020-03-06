import collections
import math

from CNN import \
    backward_orientation, backward_accelerometer, backward_gyro, backward_timestamps, \
    forward_accelerometer, forward_gyro, forward_orientation, forward_timestamps, \
    left_accelerometer, left_gyro, left_orientation, left_timestamps, \
    right_accelerometer, right_gyro, right_orientation, right_timestamps, \
    enter_accelerometer, enter_gyro, enter_orientation, enter_timestamps

import tensorflow as tf
import pandas as pd
import copy
import numpy as np
from typing import List


# backward_gyro = pd.read_csv('Myo Keyboard Data/Backward/gyro-1456704054.csv')
# # print(type(backward_gyro['x']))
# timestamp = backward_gyro['timestamp'].to_frame()
# times = []
# maxes = []
# x_col = copy.deepcopy(backward_gyro['x'])
# indices = [i for i in range(len(x_col))]
# # x_col.reindex(indices)
# num_rows = len(x_col)
# x_col = x_col.to_frame()
# x_col = x_col.drop([i for i in range(100)])
# x_col = x_col.drop([i for i in range(num_rows-90, num_rows)])
# x_col = np.array_split(x_col, 10)
#
# print("***" + str(type(x_col[0])))


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


# values are values of y axis
def get_highest_value(values):
    reverse_sorted = sorted(values)  # reversed because values are strings
    return reverse_sorted[30]


def create_ten_groups(peak, timestamps, accelerometer):
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


def extract_data_to_ten_groups(orientation: dict, accelerometer: dict):
    # orientation = read_data_into_dict_lists(orientation_file_path)
    # accelerometer = read_data_into_dict_lists(accelerometer_file_path)
    orientation_nochange_timestamps = extract_nochange_periods(orientation['timestamp'],
                                                               orientation['x'], orientation['y'], orientation['z'])
    highest = get_highest_value(accelerometer['y'])
    return create_ten_groups(highest, orientation_nochange_timestamps, accelerometer)


def find_existing_timestamp_group_average_lengths(groups, timestamps):
    length = 0
    for i in range(len(groups)):
        belongs_to_groups = False
        for j in range(len(timestamps)):
            if float(groups[i][0]) == float(timestamps[j]):
                belongs_to_groups = True
            if float(groups[i][1]) == float(timestamps[j]):
                belongs_to_groups = False
            if belongs_to_groups:
                length += 1
    return length / len(groups)


# data being accelerometer, gyro or orientation
def pre_process_data() -> ():  # type is the type of data
    # if type == 'gyro':
    backward_groups = extract_data_to_ten_groups(backward_orientation, backward_accelerometer)
    forward_groups = extract_data_to_ten_groups(forward_orientation, forward_accelerometer)
    left_groups = extract_data_to_ten_groups(left_orientation, left_accelerometer)
    right_groups = extract_data_to_ten_groups(right_orientation, right_accelerometer)
    enter_groups = extract_data_to_ten_groups(enter_orientation, enter_accelerometer)

    average_group_length_all_labels = \
        math.floor((find_existing_timestamp_group_average_lengths(backward_groups, backward_timestamps) +
                    find_existing_timestamp_group_average_lengths(forward_groups, forward_timestamps) +
                    find_existing_timestamp_group_average_lengths(left_groups, left_timestamps) +
                    find_existing_timestamp_group_average_lengths(right_groups, right_timestamps) +
                    find_existing_timestamp_group_average_lengths(enter_groups, enter_timestamps)) / 5)

    return backward_groups, forward_groups, left_groups, right_groups, enter_groups, average_group_length_all_labels


# def


# axis being x, y or z
def process_single_axis_gyro(axis: str):
    result, gyro_data, labels = [], [], []  # labels being the expected/ground truth output
    backward_groups, forward_groups, left_groups, right_groups, enter_groups, average_group_length_all_labels \
        = pre_process_data()



groups = extract_data_to_ten_groups(backward_orientation, backward_accelerometer)
print()
"""
for frame_num in range(0, len(x_col)):
    maxes.append(x_col[frame_num].max())
    i = x_col[frame_num].idxmax().x
    times.append(timestamp.iloc[i, :])
"""
# x_col = pd.Series(data=x_col, )
# the ten maxes for backward gyro are: 1456704056461410,
# for _ in range(10):
#     print(0)
#     maxes.append(x_col.max())
#     # print(x_col[x_col == maxes[-1]].index)
#     # print()
#     # i = x_col[x_col == maxes[-1]]
#     # i = i.index.values[0]
#     i = x_col.idxmax().x
#     times.append(timestamp.iloc[i, :])
#     neighbour_indices = [item for item in range(i-35, i+35)]
#     # print(neighbour_indices)
#     for row in neighbour_indices:
#         if (x_col.index == row).any():
#             x_col = x_col.drop([row])
#     # x_col = x_col[x_col != maxes[-1]]
#     # x_col.drop(labels=[])
# print(maxes)
# print(times)
# x_col

# print(tf.__version__)
