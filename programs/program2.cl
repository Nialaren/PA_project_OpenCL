#define size_n {size_n}
__kernel void compute_silhouettes(__global float *matrix, __global int *clusters, __global float *data_out, __local float* means)
{
	int i;
	int gid = get_global_id(0);
	int loc_id = get_local_id(0);
	int group_id = get_group_id(0);
	int cluster_num = get_local_size(0);
	int member_num = clusters[loc_id];

	
	int row = group_id * size_n;
	int reference_cluster_id = -1;
	int cluster_offset = 0;

	// Find Reference cluster id
	for(i=0;i<cluster_num; i++){
		if(reference_cluster_id == -1 && group_id < cluster_offset + clusters[i]){
			reference_cluster_id = i;
		} else {
			cluster_offset += clusters[i];
		}
	}
	

	// Silhouette needed means
	float cluster_sum = 0;

	// loop dependencies - reset and reuse some variables
	cluster_offset = 0;

	// find offset
	for(i=0; i<loc_id; i++){
		cluster_offset += clusters[i];
	}

	// sum
	for(i=0; i < member_num; i++){
		cluster_sum += matrix[row + cluster_offset + i];
	}
	// mean - save it to local memory
	float cluster_mean = cluster_sum/member_num;
	means[loc_id] = cluster_mean;
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// If processed cluster is reference cluster we continue
	if(loc_id == reference_cluster_id){
		
		float nearest_mean = -1, compare_mean;
		for(i=0; i<cluster_num; i++){
			if(i != loc_id){
				compare_mean = means[i];
				if(nearest_mean == -1 || nearest_mean > compare_mean){
					nearest_mean = compare_mean;
				}
			}
		}

		// Save silhouette
		data_out[group_id] = (nearest_mean - cluster_mean)/fmax(nearest_mean, cluster_mean);
		//data_out[group_id] = cluster_mean;
	}
}
