import numpy as np


def array_to_one_hot(arr, C):
    one_hot = np.zeros((arr.shape[0], C))

    for i in range(arr.shape[0]):

        idx = int(arr[i])
        one_hot[i, idx] = 1

    return one_hot
