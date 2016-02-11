import utils
import numpy as np
import math
import time

FILE_PATH = './data/data_clean.json'
COUNT_RUN = 1

if __name__ == '__main__':
    clusters = utils.load_from_file(FILE_PATH)['clusters']
    allData = []
    countData = 0

    for c in clusters:
        countData += len(c)
        for member in c:
            allData.append(member)

    data = np.array(allData, np.float32)
    data_len = len(allData)

    # create empty matrix
    matrix = np.zeros((data_len, data_len), np.float32)

    print '\n~~~~~~ Silhouette algorithm - synchronous ~~~~~~\n'

    print '---------------------------------------------------------------------------------'
    print 'Computing ..'

    sum_time = 0
    for k in range(COUNT_RUN):
        start_time = time.time()

        # count distance matrix
        range_data_len = range(data_len)
        for a_index in range_data_len:
            for b_index in range_data_len:
                inner_sum = 0.0
                for i in range(8):
                    inner_sum += pow(allData[b_index][i] - allData[a_index][i], 2)
                matrix[a_index][b_index] = math.sqrt(inner_sum)

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

                cluster_mean = inner_sum / c_len
                if not found and i < (offset + c_len):
                    found = True
                    inner_means.append(cluster_mean)
                    # inner_means.append(offset)
                else:
                    if nearest_mean == -1 or nearest_mean > cluster_mean:
                        nearest_mean = cluster_mean

                offset += c_len
            # attach nearest mean to other means
            other_means.append(nearest_mean)

        # compute Silhouette
        silhouette_data = []
        for i in range_data_len:
            loop_silhouette = (other_means[i] - inner_means[i]) / max(other_means[i], inner_means[i])
            silhouette_data.append(loop_silhouette)

        run_time = time.time() - start_time
        sum_time += run_time

        print 'Silhouette Algorithm finished, run time: {0} seconds'.format(run_time)
        print 'result: {0}'.format(np.array(silhouette_data).sum()/data_len)

    print 'Average time: {0} seconds'. format(sum_time/COUNT_RUN)



