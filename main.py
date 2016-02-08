import pyopencl as cl
import utils
import numpy as np
import math
import time

FILE_PATH = './data/data_clean.json'

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
    matrix2 = np.zeros((data_len, data_len), np.float32)

    print '\n~~~~~~ Silhouette algorithm ~~~~~~\n'
    #choose platform
    platforms = cl.get_platforms()
    print 'Available platforms:'
    for p in platforms:
        print p

    try:
        platform_choose = int(raw_input("\nPlease choose platform (0 - first, 1 - second): "))
        # print platform_choose
    except ValueError:
        print 'Input is not a number'

    # Get platform - 0 is default which does not have any devices
    platform = cl.get_platforms()[platform_choose]

    print '---------------------------------------------------------------------------------'

    print 'Available devices:'
    devices = platform.get_devices()
    for d in devices:
        print d

    try:
        device_choose = int(raw_input("Please choose device: "))
        # print device_choose
    except ValueError:
        print 'Input is not a number'

    # print 'OpenCL v.: {0}'.format(PYOPENCL_COMPILER_OUTPUTplatform.version)
    # get device - there is only one - Intel processor
    device = platform.get_devices()[device_choose]

    print '---------------------------------------------------------------------------------'
    print 'Computing ..'

    # print device.extensions
    # exit()
    # print some info
    # utils.print_device_info(device)
    # create context
    context = cl.Context([device])

    # print str(len(allData[0]))
    # create program
    program = cl.Program(context, """
        #define size_n {size_n}
        #define size_v {size_v}
        __kernel void compute_distance_matrix(
        __global float *matrix,
        __global float data[][size_v]
        )
        {
          int gid = get_global_id(0);
          int a_index = (int)(gid/size_n);
          int b_index = (int)(gid - (size_n * a_index));
          int c;
          float sum = 0;

          for(int i=0; i < size_v; i++){
            sum += pown(data[b_index][i] - data[a_index][i], 2);
          }

          matrix[gid] = sqrt(sum);
        }
        """.replace('{size_n}', str(data_len))
                         .replace('{size_v}', str(len(allData[0])))
                         ).build()

    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)

    # Allocate device memory and move input data from the host to the device memory.
    mem_flags = cl.mem_flags
    matrix_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size=matrix.nbytes)
    data_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=data)
    # data_len_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=data_len_vec)
    # cl.enqueue_write_buffer(queue, matrix_buf, matrix)
    # matrix_dot_vector = numpy.zeros(4, numpy.float32)
    # destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, matrix_dot_vector.nbytes)

    parallelStart = time.time()

    # Associate the arguments to the kernel with kernel object.
    # Deploy the kernel for device execution.
    # program.compute_distance_matrix(queue, matrix.shape, (1,), matrix_buf)
    program.compute_distance_matrix(queue, matrix.shape, (1,), matrix_buf, data_buf)
    # result = np.empty_like(matrix)
    cl.enqueue_read_buffer(queue, matrix_buf, matrix).wait()

    parallelEnd = time.time()

    # Move the kernels output data to host memory.
    # cl.enqueue_copy(queue, matrix, matrix_buf).wait()

    # Release context, program, kernels and memory.
    # PyOpenCL performs this step for you, and therefore,
    # you don't need to worry about cleanup code

    syncStart = time.time()
    range_data_len = range(data_len)
    for a_index in range_data_len:
        for b_index in range_data_len:
            inner_sum = 0.0
            for i in range(8):
                inner_sum += pow(allData[b_index][i] - allData[a_index][i], 2)
            matrix2[a_index][b_index] = math.sqrt(inner_sum)
    syncEnd = time.time()

    parallelSpeed = (parallelEnd-parallelStart)
    print 'Parallel time start: {0}, end: {1}, result: {2}'.format(
        parallelStart,
        parallelEnd,
        parallelSpeed
    )
    syncSpeed = (syncEnd-syncStart)
    print 'Sync time start: {0}, end: {1}, result: {2}'.format(
        syncStart,
        syncEnd,
        syncSpeed
    )
    print 'Difference is: {0}'.format((syncSpeed-parallelSpeed))
    # print matrix
    print 'Singe element assert'
    print matrix[1]
    print matrix2[0][1]
    # print allData[1]
    # print flattened[0]
    # print(np.max(matrix))
    # print(np.min(matrix))
    # print len(matrix)
    # print 'All Data size {0}'.format(len(allData))



