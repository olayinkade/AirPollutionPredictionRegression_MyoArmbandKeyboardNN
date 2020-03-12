import pandas as pd
import collections
import matplotlib.pylab as plt


def read_data_into_dict_lists(file_path: str):
    data = collections.defaultdict(list)
    raw_data = pd.read_csv(file_path, dtype=str)
    for column_name in raw_data:
        data[column_name] = raw_data[column_name].to_list()
    return data


def compress_emg(emg: dict):
    compressed = {'timestamp': [], 'emg1': [], 'emg2': [], 'emg3': [], 'emg4': [],
                  'emg5': [], 'emg6': [], 'emg7': [], 'emg8': []}
    stop, i = False, 0
    while i < len(emg['timestamp'])-1:
        sum_emg1, sum_emg2, sum_emg3, sum_emg4, sum_emg5, sum_emg6, sum_emg7, sum_emg8 = \
            int(emg['emg1'][i]), int(emg['emg2'][i]), int(emg['emg3'][i]), int(emg['emg4'][i]), \
            int(emg['emg5'][i]), int(emg['emg6'][i]), int(emg['emg7'][i]), int(emg['emg8'][i])

        for j in range(i+1, i+5):
            if j >= len(emg['timestamp']):
                stop = True
                break

            if emg['timestamp'][j] == emg['timestamp'][i]:
                sum_emg1 += int(emg['emg1'][j])
                sum_emg2 += int(emg['emg2'][j])
                sum_emg3 += int(emg['emg3'][j])
                sum_emg4 += int(emg['emg4'][j])
                sum_emg5 += int(emg['emg5'][j])
                sum_emg6 += int(emg['emg6'][j])
                sum_emg7 += int(emg['emg7'][j])
                sum_emg8 += int(emg['emg8'][j])
            else:
                compressed['timestamp'].append(emg['timestamp'][i])
                compressed['emg1'].append(str(sum_emg1 / (j - i + 1)))
                compressed['emg2'].append(str(sum_emg2 / (j - i + 1)))
                compressed['emg3'].append(str(sum_emg3 / (j - i + 1)))
                compressed['emg4'].append(str(sum_emg4 / (j - i + 1)))
                compressed['emg5'].append(str(sum_emg5 / (j - i + 1)))
                compressed['emg6'].append(str(sum_emg6 / (j - i + 1)))
                compressed['emg7'].append(str(sum_emg7 / (j - i + 1)))
                compressed['emg8'].append(str(sum_emg8 / (j - i + 1)))
                i = j
                break
        if stop:
            break

    for key in compressed.keys():
        compressed[key] = compressed[key][::2]

    return compressed


''' READ IN ALL NECESSARY FILES '''

backward_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Backward/accelerometer-1456704054.csv')
backward_timestamps = backward_accelerometer['timestamp']
backward_gyro = read_data_into_dict_lists('Myo Keyboard Data/Backward/gyro-1456704054.csv')
backward_orientation = read_data_into_dict_lists('Myo Keyboard Data/Backward/orientation-1456704054.csv')
backward_emg = read_data_into_dict_lists('Myo Keyboard Data/Backward/emg-1456704054.csv')
backward_emg = compress_emg(backward_emg)

forward_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Forward/accelerometer-1456703940.csv')
forward_timestamps = forward_accelerometer['timestamp']
forward_gyro = read_data_into_dict_lists('Myo Keyboard Data/Forward/gyro-1456703940.csv')
forward_orientation = read_data_into_dict_lists('Myo Keyboard Data/Forward/orientation-1456703940.csv')
forward_emg = read_data_into_dict_lists('Myo Keyboard Data/Forward/emg-1456703940.csv')
forward_emg = compress_emg(forward_emg)

left_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Left/accelerometer-1456704106.csv')
left_timestamps = left_accelerometer['timestamp']
left_gyro = read_data_into_dict_lists('Myo Keyboard Data/Left/gyro-1456704106.csv')
left_orientation = read_data_into_dict_lists('Myo Keyboard Data/Left/orientation-1456704106.csv')
left_emg = read_data_into_dict_lists('Myo Keyboard Data/Left/emg-1456704106.csv')
left_emg = compress_emg(left_emg)

right_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Right/accelerometer-1456704146.csv')
right_timestamps = right_accelerometer['timestamp']
right_gyro = read_data_into_dict_lists('Myo Keyboard Data/Right/gyro-1456704146.csv')
right_orientation = read_data_into_dict_lists('Myo Keyboard Data/Right/orientation-1456704146.csv')
right_emg = read_data_into_dict_lists('Myo Keyboard Data/Right/emg-1456704146.csv')
right_emg = compress_emg(right_emg)

enter_accelerometer = read_data_into_dict_lists('Myo Keyboard Data/Enter/accelerometer-1456704184.csv')
enter_timestamps = enter_accelerometer['timestamp']
enter_gyro = read_data_into_dict_lists('Myo Keyboard Data/Enter/gyro-1456704184.csv')
enter_orientation = read_data_into_dict_lists('Myo Keyboard Data/Enter/orientation-1456704184.csv')
enter_emg = read_data_into_dict_lists('Myo Keyboard Data/Enter/emg-1456704184.csv')
enter_emg = compress_emg(enter_emg)
