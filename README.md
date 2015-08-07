Brute-Force k-Nearest Neighbors Search on the GPU
=================================================

**bf-knn** implements a brute-force approach for finding k-nearest neighbors on
the GPU for many queries in parallel.
It takes advantage of recent advances in fundamental GPU computing primitives.
The squared Euclidean distances between queries and references are calculated by
a CUDA kernel modified from a matrix multiplication subroutine in the
[MAGMA](http://icl.cs.utk.edu/magma/) library.
The nearest neighbors selection is accomplished by a truncated merge sort built
on top of sorting and merging functions in the
[Modern GPU](http://nvlabs.github.io/moderngpu/) library.
Compared to state-of-the-art approaches, **bf-knn** is faster and handles larger
inputs.

Instructions to download **bf-knn** and compile the demo:
```
git clone git@github.com:NVlabs/moderngpu.git
git clone git@github.com:geomlab-ucd/bf-knn.git
cd bf-knn
nvcc -arch=sm_21 -I ../moderngpu/include/ bf_knn_device.cu bf_knn_host.cu demo.cc -o bf-knn-demo
```

[View License](LICENSE.txt)

