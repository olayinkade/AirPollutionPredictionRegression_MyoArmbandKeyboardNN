import tensorflow as tf
import pandas as pd
import copy
import numpy as np

backward_gyro = pd.read_csv('Myo Keyboard Data/Backward/gyro-1456704054.csv')
# print(type(backward_gyro['x']))
timestamp = backward_gyro['timestamp'].to_frame()
times = []
maxes = []
x_col = copy.deepcopy(backward_gyro['x'])
indices = [i for i in range(len(x_col))]
# x_col.reindex(indices)
num_rows = len(x_col)
x_col = x_col.to_frame()
x_col = x_col.drop([i for i in range(100)])
x_col = x_col.drop([i for i in range(num_rows-90, num_rows)])
x_col = np.array_split(x_col, 10)

print("***" + str(type(x_col[0])))

for frame_num in range(0, len(x_col)):
    maxes.append(x_col[frame_num].max())
    i = x_col[frame_num].idxmax().x
    times.append(timestamp.iloc[i, :])

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
print(maxes)
print(times)
    # x_col

# print(tf.__version__)
