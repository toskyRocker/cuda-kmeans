Cuda K-Means algorithm for image clustering implemented in NVidia Cuda 
by Andrea Toscano, Università degli Studi di Milano (Informatica).

This little projects shows how to implement the K-Means clustering algorithm applied to an image in order to reduce its colours.
Some pre-processing is computed using python scripts in order to represent the image in a better way and to easily find suitable initial centroids for the algorithm.

In Cuda K-Means routine global memory and constant memory are involved.
Future work will include also the texture memory to contain the entire picture allowing a better performance of the algorithm.
