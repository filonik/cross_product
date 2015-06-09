# cross_product
General cross products for numpy. (Simple Cython project for experimenting with wrapping C libraries.)

### Usage

```python
import numpy as np
import cross_product as cp

v = np.eye(3, dtype=np.float32)

np.testing.assert_array_almost_equal(cp.cross(v[0], v[1]), v[2])
```