#define size_n {size_n}
#define cluster_size {size_c}
__kernel void compute_silhouettes(__global float *matrix, __global int *clusters, __global float *data_out)
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
	data_out[gid] = (nearest_mean - inner_cluster_mean)/fmax(nearest_mean, inner_cluster_mean);
}
