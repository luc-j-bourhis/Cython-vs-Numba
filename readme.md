# Benchmark Cython vs Numba

A work-in-progress assessment of Cython vs Numba with Numpy as a baseline for various array manipulation. So far

- Pairwise distances (`pairwise_distances.py`): a list of 3D coordinates as input, the list of all distances between any two points as output;
- Laplacian filter (`laplacian.py`): a 512x512 image as input, compute the Laplacian, filter the resulting image with a threshold.

Running the benchmarks is as simple as running the scripts with python. There are some prerequisites of course: the easiest is to install them using [conda](https://conda.io/docs/) and the exported `environment.yml`:

    % conda env create -f environment.yml

This will create an environment named `numerics`: activate it before running the benchmarks.

Conclusions so far.

- On MacOS Sierra with all packages installed with conda, with Westmere X5690
    - Pairwise distances: Cython and Numba are both about 5 times faster than Numpy
    - Laplacian: Numba is 2-3 times faster than Cython which is more than twice faster than Numpy

