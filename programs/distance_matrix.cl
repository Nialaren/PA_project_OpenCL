#define size_v {size_v}
__kernel void compute_distance_matrix(const int N, __global float *matrix, __global float data[][size_v])
{
  int gid = get_global_id(0);
  int a_index = (int)(gid/N);
  int b_index = (int)(gid - (N * a_index));
  float sum = 0.0;

  for(int i=0; i < size_v; i++){
    sum += pown(data[b_index][i] - data[a_index][i], 2);
  }

  matrix[gid] = sqrt(sum);
}