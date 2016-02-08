import utils
import numpy as np
import math
import time

FILE_PATH = './data/data_clean.json'
COUNT_RUN = 5

if __name__ == '__main__':
    # matrix_dot_vector = np.zeros(4, np.float32)
    # print matrix_dot_vector.shape
    # exit()
    # Test Data preparation
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
    matrix = np.zeros(data_len**2, np.float32)
    # matrix2 = np.zeros((data_len, data_len), np.float32)

    print '\n~~~~~~ Silhouette algorithm - synchronous ~~~~~~\n'

    print '---------------------------------------------------------------------------------'
    print 'Computing ..'

    sum_time = 0
    for k in range(COUNT_RUN):
        start_time = time.time()







        run_time = time.time() - start_time
        sum_time += run_time

        print 'Silhouette Algorithm finished, run time: {0} seconds'.format(run_time)
        print 'result: {0}'.format(result / data_len)

    print 'Average time: {0} seconds'. format(sum_time/COUNT_RUN)



