import pyopencl as cl
import utils
import numpy as np
import math

FILE_PATH = './data/stoneFlakes_clusters.json'

if __name__ == '__main__':
    # matrix_dot_vector = np.zeros(4, np.float32)
    # print matrix_dot_vector.shape
    # exit()
    # Test Data preparation
    clusters = utils.load_from_file(FILE_PATH)['clusters']
    allData = []
    countData = 0

    for c in clusters:
        for member in c:
            allData.append(member)

    data_len = len(allData)
    matrix = np.zeros((data_len, data_len), np.float32)

    # count distance matrix
    range_data_len = range(data_len)
    for a_index in range_data_len:
        for b_index in range_data_len:
            inner_sum = 0.0
            for i in range(8):
                inner_sum += pow(allData[b_index][i] - allData[a_index][i], 2)
            matrix[a_index][b_index] = math.sqrt(inner_sum)

    # count inner cluster mean distance
    inner_clusters = []
    for i in range_data_len:
        offset = 0
        c_len = None
        for cluster in clusters:
            c_len = len(cluster)
            if i < (offset + c_len):
                break
            offset += c_len

        inner_sum = 0.0
        for j in range(c_len):
            inner_sum += matrix[i][offset + j]
        inner_clusters.append(inner_sum/data_len)

    print inner_clusters
