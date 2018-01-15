import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def laplacian_cython(double[:, ::1] image):
  cdef int h = image.shape[0]
  cdef int w = image.shape[1]
  cdef double[:, ::1] laplacian = np.empty((h-2, w-2), dtype=np.double)
  cdef int i,j # necessary here, as warned by Cython itself
  for i in range(1, h-1):
    for j in range(1, w-1):
      laplacian[i-1, j-1] = (image[i-1, j] + image[i+1, j] +
                             image[i, j-1] + image[i, j+1]
                             - 4*image[i, j])
  thresh = np.abs(laplacian) > 0.05
  return thresh

@cython.boundscheck(False)
@cython.wraparound(False)
def laplacian_cython_bis(double[:, :] image):
  cdef int h = image.shape[0]
  cdef int w = image.shape[1]
  cdef double[:, ::1] laplacian = np.empty((h-2, w-2), dtype=np.double)
  cdef int i,j # necessary here, as warned by Cython itself
  cdef double l
  for i in range(1, h-1):
    for j in range(1, w-1):
      l = (image[i-1, j] + image[i+1, j] +
           image[i, j-1] + image[i, j+1]
           - 4*image[i, j])
      laplacian[i-1, j-1] = 1 if l > 0.05 else 0
  return laplacian
