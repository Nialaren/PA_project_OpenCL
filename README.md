# Silhouette clustering coefficient with PyOpenCL
(PA I project PyOpenCL)

Project to Paralel Algorithms I course. Aim is to create parallel version of count of Silhouette clustering coefficient for some clustered Data

## Requirements
 - OpenCL driver based on your device architecture (AMD, Nvidia, Intel etc.)
 - Python 2.6+
 - PyOpenCL
 - Numpy
 - MingWPy
 - Mako
 
## File stucture
Project contains 3 exectutable standalone main files. 
 - `main.py` - contains non-parallel version of algorithm
 - `main_parallel.py` - first version of paralel computation (prove of concept)
 - `main_parallel2.py` - Second version with slight improvement
 
 Programs - File contains kernel programs used for parallel computation
  - `distance_matrix.cl`
  - `silhouette_1.cl` - first version
  - `silhouette_2.cl` - second version
  
Data - File contains sample data

## Run
Each file have at the top `FILE_PATH` constant which contains path to sample data
Sample data have to be in JSON file in following structure:
```JSON
{
  "clusters":[
    [[1,2], [4,5]],
    [[5,4], [55,43]],
  ]
}
```

Key clusters contains array of clusters. Each cluter contains array of individual members of cluster. Members are represented as arrays.

``python main_parallel.py``
