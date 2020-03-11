# from myo_keyboard_classification import read_data_into_dict_lists
import pandas as pd
import collections


def read_data_into_dict_lists(file_path: str):
    data = collections.defaultdict(list)
    raw_data = pd.read_csv(file_path, dtype=str)
    for column_name in raw_data:
        data[column_name] = raw_data[column_name].to_list()
    return data


''' READ IN ALL NECESSARY FILES '''

backward_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Backward/accelerometer-1456704054.csv')
backward_timestamps = backward_accelerometer['timestamp']
backward_gyro = read_data_into_dict_lists('Myo Keyboard Data/Backward/gyro-1456704054.csv')
backward_orientation = read_data_into_dict_lists('Myo Keyboard Data/Backward/orientation-1456704054.csv')

forward_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Forward/accelerometer-1456703940.csv')
forward_timestamps = forward_accelerometer['timestamp']
forward_gyro = read_data_into_dict_lists('Myo Keyboard Data/Forward/gyro-1456703940.csv')
forward_orientation = read_data_into_dict_lists('Myo Keyboard Data/Forward/orientation-1456703940.csv')

left_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Left/accelerometer-1456704106.csv')
left_timestamps = left_accelerometer['timestamp']
left_gyro = read_data_into_dict_lists('Myo Keyboard Data/Left/gyro-1456704106.csv')
left_orientation = read_data_into_dict_lists('Myo Keyboard Data/Left/orientation-1456704106.csv')

right_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Right/accelerometer-1456704146.csv')
right_timestamps = right_accelerometer['timestamp']
right_gyro = read_data_into_dict_lists('Myo Keyboard Data/Right/gyro-1456704146.csv')
right_orientation = read_data_into_dict_lists('Myo Keyboard Data/Right/orientation-1456704146.csv')

enter_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Enter/accelerometer-1456704184.csv')
enter_timestamps = enter_accelerometer['timestamp']
enter_gyro = read_data_into_dict_lists('Myo Keyboard Data/Enter/gyro-1456704184.csv')
enter_orientation = read_data_into_dict_lists('Myo Keyboard Data/Enter/orientation-1456704184.csv')
