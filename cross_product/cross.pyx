cimport cython

import numpy as np

cimport numpy as np

cimport c_cross
	
def check_shape(expected, received):
	if expected != received:
		raise ValueError("Invalid shape (expected %s, received %s)." % (expected, received))

cdef remove_i(np.ndarray[float, ndim=2, mode="c"] x, int i):
	idx = range(x.shape[0])
	idx.remove(i)
	return x[idx,:]

cdef remove_j(np.ndarray[float, ndim=2, mode="c"] x, int j):
	idx = range(x.shape[1])
	idx.remove(j)
	return x[:,idx]

cdef remove_ij(np.ndarray[float, ndim=2, mode="c"] x, int i, int j):
	x = remove_i(x, i)
	x = remove_j(x, j)
	return x

def cross1():
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(1, dtype=np.float32)
	
	c_cross.cross1(&r0[0])
	
	return r0

def cross2(float[::1] v0):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(2, dtype=np.float32)
	
	check_shape((2,), (v0.shape[0],))
	
	c_cross.cross2(&v0[0], &r0[0])
	
	return r0

def cross3(float[::1] v0, float[::1] v1):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(3, dtype=np.float32)
	
	check_shape((3,), (v0.shape[0],))
	check_shape((3,), (v1.shape[0],))
	
	c_cross.cross3(&v0[0], &v1[0], &r0[0])
	
	return r0

def cross4(float[::1] v0, float[::1] v1, float[::1] v2):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(4, dtype=np.float32)
	
	check_shape((4,), (v0.shape[0],))
	check_shape((4,), (v1.shape[0],))
	check_shape((4,), (v2.shape[0],))
	
	c_cross.cross4(&v0[0], &v1[0], &v2[0], &r0[0])
	
	return r0

def cross5(float[::1] v0, float[::1] v1, float[::1] v2, float[::1] v3):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(5, dtype=np.float32)
	
	check_shape((5,), (v0.shape[0],))
	check_shape((5,), (v1.shape[0],))
	check_shape((5,), (v2.shape[0],))
	check_shape((5,), (v3.shape[0],))
	
	c_cross.cross5(&v0[0], &v1[0], &v2[0], &v3[0], &r0[0])
	
	return r0

def cross6(float[::1] v0, float[::1] v1, float[::1] v2, float[::1] v3, float[::1] v4):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(6, dtype=np.float32)
	
	check_shape((6,), (v0.shape[0],))
	check_shape((6,), (v1.shape[0],))
	check_shape((6,), (v2.shape[0],))
	check_shape((6,), (v3.shape[0],))
	check_shape((6,), (v4.shape[0],))
	
	c_cross.cross6(&v0[0], &v1[0], &v2[0], &v3[0], &v4[0], &r0[0])
	
	return r0

def cross7(float[::1] v0, float[::1] v1, float[::1] v2, float[::1] v3, float[::1] v4, float[::1] v5):
	cdef np.ndarray[float, ndim=1, mode="c"] r0 = np.empty(7, dtype=np.float32)
	
	check_shape((7,), (v0.shape[0],))
	check_shape((7,), (v1.shape[0],))
	check_shape((7,), (v2.shape[0],))
	check_shape((7,), (v3.shape[0],))
	check_shape((7,), (v4.shape[0],))
	check_shape((7,), (v5.shape[0],))
	
	c_cross.cross7(&v0[0], &v1[0], &v2[0], &v3[0], &v4[0], &v5[0], &r0[0])
	
	return r0

def crossn(np.ndarray[float, ndim=2, mode="c"] vn):
	cdef size_t n = vn.shape[1]
	return np.array([(-1.0)**(j+n+1) * np.linalg.det(remove_j(vn, j)) for j in range(n)], dtype=np.float32)

def cross(*args):
	cdef int n = len(args)
	if n==0:
		return cross1(*args)
	elif n==1:
		return cross2(*args)
	elif n==2:
		return cross3(*args)
	elif n==3:
		return cross4(*args)
	elif n==4:
		return cross5(*args)
	elif n==5:
		return cross6(*args)
	elif n==6:
		return cross7(*args)
	else:
		return crossn(np.array(args, dtype=np.float32))