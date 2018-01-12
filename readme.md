# Benchmark Cython vs Numba

A work-in-progress assessment of Cython vs Numba with Numpy as a baseline for various array manipulation. So far

- Pairwise distances: a list of 3D coordinates as input, the list of all distances between any two points as output;
- Laplacian filter: a 512x512 image as input, compute the Laplacian, filter the resulting image with a threshold;

Conclusions so far.

- On MacOS Sierra with all packages installed with conda, with Westmere X5690
    - Pairwise distances: Cython and Numba are both about 5 times faster than Numpy
    - Laplacian: Numba is 2-3 times faster than Cython which is more than twice faster than Numpy

