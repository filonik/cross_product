import logging
import unittest

import numpy as np

import cross_product as cp

"""
Some basic tests
"""

logging.basicConfig(level=logging.DEBUG)

def remove_i(x, i):
    idx = range(x.shape[0])
    idx.remove(i)
    return x[idx,:]
    
def remove_j(x, j):
    idx = range(x.shape[1])
    idx.remove(j)
    return x[:,idx]
    
def remove_ij(x, i, j):
    x = remove_i(x, i)
    x = remove_j(x, j)
    return x

class TestCross(unittest.TestCase):
    def test_cross1(self):
        self.assertAlmostEqual(cp.cross(), np.array([-1.0], dtype=np.float32))
        
    def test_cross2(self):
        v = np.eye(2, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([-1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, +1.0], dtype=np.float32))

    def test_cross3(self):
        v = np.eye(3, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([+1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, +1.0], dtype=np.float32))

    def test_cross4(self):
        v = np.eye(4, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, +1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 3)), np.array([0.0, 0.0, 0.0, +1.0], dtype=np.float32))

    def test_cross5(self):
        v = np.eye(5, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([+1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, +1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 3)), np.array([0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 4)), np.array([0.0, 0.0, 0.0, 0.0, +1.0], dtype=np.float32))
        
    def test_cross6(self):
        v = np.eye(6, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 3)), np.array([0.0, 0.0, 0.0, +1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 4)), np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 5)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, +1.0], dtype=np.float32))
        
    def test_cross7(self):
        v = np.eye(7, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([+1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 3)), np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 4)), np.array([0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 5)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 6)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0], dtype=np.float32))
    
    def test_cross8(self):
        v = np.eye(8, dtype=np.float32)
        
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 0)), np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 1)), np.array([0.0, +1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 2)), np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 3)), np.array([0.0, 0.0, 0.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 4)), np.array([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 5)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, +1.0, 0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 6)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, 7)), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, +1.0], dtype=np.float32))
    
    def test_crossn(self):
        for n in range(8, 32):
            v = np.eye(n, dtype=np.float32)
            
            for i in range(n):
                np.testing.assert_array_almost_equal(cp.cross(*remove_i(v, i)), (-1.0)**(n+i+1) * v[i])

if __name__ == '__main__':
    unittest.main()