""" Laplacian filtering of an image

    Adapted from Florian LE BOURDAIS' blog post at
    https://flothesof.github.io/optimizing-python-code-numpy-cython-pythran-numba.html
"""

import skimage.data
import skimage.color
import skimage.filters
import pylab

import timeit

import numpy as np

image0 = skimage.data.astronaut()
image = skimage.color.rgb2gray(image0)

def compare(left, right):
  """Compares two images, left and right."""
  fig, ax = pylab.subplots(1, 2, figsize=(10, 5))
  ax[0].imshow(left, cmap='gray')
  ax[1].imshow(right, cmap='gray')
  pylab.show()

def laplacian_skimage(image):
  laplacian = skimage.filters.laplace(image)
  thresh = np.abs(laplacian) > 0.05
  return thresh

def laplacian_numpy(image):
  laplacian = (image[:-2, 1:-1] + image[2:, 1:-1] +
               image[1:-1, :-2] + image[1:-1, 2:]
               - 4*image[1:-1, 1:-1])
  thresh = np.abs(laplacian) > 0.05
  return thresh

def laplacian_python(image):
  h = image.shape[0]
  w = image.shape[1]
  laplacian = np.empty((h-2, w-2), dtype=np.double)
  for i in range(1, h-1):
    for j in range(1, w-1):
      laplacian[i-1, j-1] = (image[i-1, j] + image[i+1, j] +
                             image[i, j-1] + image[i, j+1]
                             - 4*image[i, j])
  thresh = np.abs(laplacian) > 0.05
  return thresh

def laplacian_python_bis(image):
  h = image.shape[0]
  w = image.shape[1]
  laplacian = np.empty((h-2, w-2), dtype=np.double)
  for i in range(1, h-1):
    for j in range(1, w-1):
      l = (image[i-1, j] + image[i+1, j] +
               image[i, j-1] + image[i, j+1]
               - 4*image[i, j])
      laplacian[i-1, j-1] = 1 if l > 0.05 else 0
  return laplacian


import pyximport
pyximport.install()
from cy_laplacian import laplacian_cython, laplacian_cython_bis
#compare(left=image, right=laplacian_cython(image))

import numba
# Force compilation of functions by numba before we time
laplacian_numba = numba.jit(laplacian_python)
laplacian_numba(image)
laplacian_numba_bis = numba.jit(laplacian_python_bis)
laplacian_numba_bis(image)

def mytimeit(func, image):
  return timeit.timeit(
    stmt="{}(image)".format(func),
    setup="from __main__ import {}; ".format(func),
    globals=locals(),
    number=500)

print("Speedup with respect to Numpy")
t0 = mytimeit("laplacian_numpy", image)
print("Numpy: {:.2f} s".format(t0))
t = mytimeit("laplacian_skimage", image)
print("Skimage: x {:.1f}".format(t0/t))
t = mytimeit("laplacian_cython", image)
print("Cython: x {:.1f}".format(t0/t))
t = mytimeit("laplacian_numba", image)
print("Numba: x {:.1f}".format(t0/t))
t = mytimeit("laplacian_cython_bis", image)
print("Cython (bis): x {:.1f}".format(t0/t))
t = mytimeit("laplacian_numba_bis", image)
print("Numba (bis): x {:.1f}".format(t0/t))
