import pyopencl as cl
import pyopencl.array as cl_array
import utils
import numpy as np
import math
import time

FILE_PATH = './data/stoneFlakes_clusters.json'
# FILE_PATH = './data/test_data.json'

if __name__ == '__main__':
    # matrix_dot_vector = np.zeros(4, np.float32)
    # print matrix_dot_vector.shape
    # exit()
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
    # print data_len
    # exit()
    clusterInfoBuff = np.array(clusterInfo, np.int32)

    # print clusterInfoBuff[0]
    # exit()

    # create empty matrix
    matrix = np.zeros(data_len**2, np.float32)
    matrix2 = np.zeros((data_len, data_len), np.float32)

    # Get platform - 0 is default which does not have any devices
    platform = cl.get_platforms()[1]
    # print 'OpenCL v.: {0}'.format(platform.version)
    # get device - there is only one - Intel processor
    device = platform.get_devices()[0]
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
          double sum = 0.0;

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

    # Associate the arguments to the kernel with kernel object.
    # Deploy the kernel for device execution.

    program.compute_distance_matrix(queue, matrix.shape, (1,), matrix_buf, data_buf)
    # Move the kernels output data to host memory.
    cl.enqueue_read_buffer(queue, matrix_buf, matrix).wait()

    # Release context, program, kernels and memory.
    # PyOpenCL performs this step for you, and therefore,
    # you don't need to worry about cleanup code

    out_data = np.zeros(data_len, np.float32)

    program2 = cl.Program(context, """
        #define size_n {size_n}
        #define cluster_size {size_c}
        __kernel void compute_silhouettes(
        __global float *matrix,
        __global int *clusters,
        __global float *data_out
        )
        {
          int i = 0;
          int gid = get_global_id(0);
          int row = gid * size_n;
          int cluster_id;
          int cluster_offset = 0;

          // Determine to which cluster data belongs
          for(;i<cluster_size; i++){
            if(gid < cluster_offset + clusters[i]){
              cluster_id = i;
              break;
            } else {
              cluster_offset += clusters[i];
            }
          }

          // Silhouette needed means
          float inner_cluster_mean = 0;
          float nearest_mean = -1;

          // loop dependencies - reset and reuse some variables
          i=0;
          cluster_offset = 0;
          float cluster_sum = 0;
          float cluster_mean = 0;


          // loop counting all cluster mean distances
          for(; i<cluster_size; i++){
            // loop summing cluster member distances
            for(int j=0; j < clusters[i]; j++){
              cluster_sum += matrix[row + cluster_offset + j];
            }
            // counting our mean for particular cluster
            cluster_mean = cluster_sum/clusters[i];
            cluster_sum = 0;
            // If processed cluster is our cluster we save it
            if(i == cluster_id){
              inner_cluster_mean = cluster_mean;
            } else {
            // Else we just check if processed cluster has lower mean (is closer)
              if(nearest_mean == -1 || nearest_mean > cluster_mean){
                nearest_mean = cluster_mean;
              }
            }
            // Always update offset
            cluster_offset += clusters[i];
          }


          // Output
          //data_out[gid] = (nearest_mean - inner_cluster_mean)/fmax(nearest_mean, inner_cluster_mean);
          data_out[gid] = inner_cluster_mean;
        }
        """.replace('{size_n}', str(data_len))
                          .replace('{size_c}', str(len(clusters)))
                          ).build()

    matrix_buf_2 = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=matrix)
    cluster_info_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, hostbuf=clusterInfoBuff)
    out_data_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, size=out_data.nbytes)

    program2.compute_silhouettes(queue, out_data.shape, (1,), matrix_buf_2, cluster_info_buf, out_data_buf)

    cl.enqueue_read_buffer(queue, out_data_buf, out_data).wait()

    print out_data

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
    #         name="reduction_kernel")
    #
    # result = reduction_kernel(x_gpu).get()
    #
    # print result


