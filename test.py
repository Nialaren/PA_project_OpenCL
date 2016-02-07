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
    # inner_means = []
    # for i in range_data_len:
    #     offset = 0
    #     c_len = None
    #     for cluster in clusters:
    #         c_len = len(cluster)
    #         if i < (offset + c_len):
    #             break
    #         offset += c_len
    #
    #     inner_sum = 0.0
    #     for j in range(c_len):
    #         inner_sum += matrix[i][offset + j]
    #     inner_means.append(inner_sum/data_len)
    #
    # print inner_means
    # exit()


    # find all means
    inner_means = []
    other_means = []
    for i in range_data_len:
        offset = 0
        found = False
        c_len = None
        nearest_mean = -1
        for cluster in clusters:
            c_len = len(cluster)
            inner_sum = 0.0
            for j in range(c_len):
                inner_sum += matrix[i][offset + j]

            cluster_mean = inner_sum / data_len
            if not found and i < (offset + c_len):
                found = True
                inner_means.append(inner_sum/data_len)
            else:
                if nearest_mean == -1 or nearest_mean > cluster_mean:
                    nearest_mean = cluster_mean

            offset += c_len
        # attach nearest mean to other means
        other_means.append(nearest_mean)

    # print other_means
    # exit()

    # compute Silhouette
    silhouette_data = []
    for i in range_data_len:
        loop_silhouette = (other_means[i] - inner_means[i]) / max(other_means[i], inner_means[i])
        silhouette_data.append(loop_silhouette)

    print silhouette_data



