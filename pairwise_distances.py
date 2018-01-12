""" Computation of pairwise distances between 3D points

    Adapted from http://jakevdp.github.io/blog/2013/06/15/numba-vs-cython-take-2/
"""

import timeit

import numpy as np
X = np.random.random((1500, 3))

def pairwise_numpy(X):
  return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))

def pairwise_python(X):
  M = X.shape[0]
  N = X.shape[1]
  D = np.empty((M, M), dtype=np.float64)
  for i in range(M):
    for j in range(M):
      d = 0.0
      for k in range(N):
        tmp = X[i, k] - X[j, k]
        d += tmp * tmp
      D[i, j] = np.sqrt(d)
  return D

import numba
use_signature = False
if use_signature:
  pairwise_numba = numba.jit(pairwise_python)
  # Make sure numba compiles the function with the type or argument we are
  # going to use in the benchmark, so as to get a fair comparison with Cython.
  pairwise_numba(np.array([(1., 1., 1.)]*10))
else:
  # Alternative: specify the signature, as numba will infer
  # it at the call point and then compile just-in-time the appropriate variant.
  # However, for our simple-minded test, this would give numba an unfair
  # disadvantage compared to Cython which pre-compiled the variant
  # corresponding to the specified signature.
  pairwise_numba = numba.jit('float64[:,:](float64[:,:])')(pairwise_python)

import pyximport
pyximport.install()
from cy_pairwise_distance import pairwise_cython

def mytimeit(func, repeats=5):
  t = timeit.repeat(stmt="{}(X)".format(func),
                    setup="from __main__ import {}, X".format(func),
                    number=repeats)
  return sum(sorted(t)[:3])/3

base_is_python = False
if base_is_python:
  t0 = mytimeit("pairwise_python", repeats=1)
  print("Python: {:.2f} s".format(t0))
  print("Numpy: x {:.1f} speedup".format(t0/mytimeit("pairwise_numpy")))
else:
  t0 = mytimeit("pairwise_numpy")
  print("Numpy: {:.2f} s".format(t0))
print("Numba: x {:.1f} speedup".format(t0/mytimeit("pairwise_numba")))
print("Cython: x {:.1f} speedup".format(t0/mytimeit("pairwise_cython")))
