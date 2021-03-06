import pyopencl as cl
import pyopencl.array as cl_array
import utils
import numpy as np
import math
import time

FILE_PATH = './data/data_clean.json'

COUNT_RUN = 5

mem_flags = cl.mem_flags

if __name__ == '__main__':
    # Test Data preparation
    clusters = utils.load_from_file(FILE_PATH)['clusters']

    allData = []
    clusterInfo = []
    countData = 0

    for c in clusters:
        countData += len(c)
        for member in c:
            allData.append(member)
        clusterInfo.append(len(c))

    data = np.array(allData, np.float32)
    data_len = len(allData)
    vec_size = len(data[0])
    clusterInfoBuff = np.array(clusterInfo, np.int32)

    # create empty matrix
    matrix = np.zeros(data_len**2, np.float32)
    matrix2 = np.zeros((data_len, data_len), np.float32)

    device = None
    platform = None

    print '\n~~~~~~ Silhouette algorithm ~~~~~~\n'
    # choose platform
    platforms = cl.get_platforms()
    print 'Available platforms:'
    for p in platforms:
        print p

    platform_choose = 0
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

    device_choose = 0
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

    # create context
    context = cl.Context([device])
    # Create a command queue for the target device.
    queue = cl.CommandQueue(context)

    # create program
    # ######### PART 1 - distance matrix ###########
    # create program - distance matrix

    sum_time = 0
    for k in range(COUNT_RUN):
        start_time = time.time()
        program = cl.Program(context,
                             open('./programs/distance_matrix.cl').read()
                             .replace('{size_v}', str(vec_size))
                             ).build()

        # Allocate device memory and move input data from the host to the device memory.
        matrix_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size=matrix.nbytes)
        data_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=data)

        # Deploy the kernel for device execution.
        compute_distance_matrix = program.compute_distance_matrix
        compute_distance_matrix.set_scalar_arg_dtypes([np.int32, None, None])
        compute_distance_matrix(queue, matrix.shape, (1,), data_len,  matrix_buf, data_buf)

        # Move the kernels output data to host memory.
        cl.enqueue_read_buffer(queue, matrix_buf, matrix).wait()

        out_data = np.zeros(data_len, np.float32)

        # #### PART 2 - Silhouette

        program2 = cl.Program(context,
                              open('./programs/silhouette_1.cl').read()
                              .replace('{size_n}', str(data_len))
                              .replace('{size_c}', str(len(clusters)))
                              ).build()

        matrix_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
        cluster_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=clusterInfoBuff)
        out_data_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size=out_data.nbytes)

        program2.compute_silhouettes(queue, out_data.shape, (1,), matrix_buf_2, cluster_info_buf, out_data_buf)

        cl.enqueue_read_buffer(queue, out_data_buf, out_data).wait()

        # print out_data

        # ###### PART 3. REDUCE
        # x_gpu = cl_array.to_device(queue, out_data.astype(np.float32))
        #
        # from pyopencl.reduction import ReductionKernel
        # reduction_kernel = ReductionKernel(
        #         context,
        #         dtype_out=np.float32,
        #         neutral="0",
        #         map_expr="x[i]",
        #         reduce_expr="a+b",
        #         arguments="__global float *x",
        #         preamble="#pragma OPENCL EXTENSION cl_khr_fp64: enable",
        #         name="reduction_kernel")
        #
        # result = reduction_kernel(x_gpu).get()

        result = out_data.sum()
        run_time = time.time() - start_time
        sum_time += run_time

        print 'Silhouette Algorithm finished, run time: {0} seconds'.format(run_time)
        print 'result: {0}'.format(result / data_len)

    print 'Average time: {0} seconds'. format(sum_time/COUNT_RUN)