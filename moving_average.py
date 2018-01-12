#!/usr/bin/env python
"""
A moving average function using @guvectorize.
"""

import numpy as np

from numba import guvectorize

@guvectorize(['void(float64[:], intp[:], float64[:])'], '(n),()->(n)',
             cache=True)
def move_mean(a, window_arr, out):
  window_width = window_arr[0]
  asum = 0.0
  count = 0
  for i in range(window_width):
    asum += a[i]
    count += 1
    out[i] = asum / count
  for i in range(window_width, len(a)):
    asum += a[i] - a[i - window_width]
    out[i] = asum / count

n = 200000000
arr = np.arange(n, dtype=np.float64).reshape(2, n//2)
print(arr)
print(move_mean(arr, 3))
