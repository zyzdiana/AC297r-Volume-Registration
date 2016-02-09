import numpy as np
import scipy.special
from cost_functions import cf_ssd,cf_L1,cf_L2
from mask import sphere_mask
import cPickle as pickle

X_inv =np.array([
( 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 9,-9,-9, 9, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-6, 6, 6,-6, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-2,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-6, 6, 6,-6, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-1,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 4,-4,-4, 4, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9,-9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3,-6,-3, 0, 0, 0, 0, 6,-6, 3,-3, 0, 0, 0, 0, 4, 2, 2, 1, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 3, 3, 0, 0, 0, 0,-4, 4,-2, 2, 0, 0, 0, 0,-2,-2,-1,-1, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4,-2, 4, 2, 0, 0, 0, 0,-3, 3,-3, 3, 0, 0, 0, 0,-2,-1,-2,-1, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4,-4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2,-2,-2, 0, 0, 0, 0, 2,-2, 2,-2, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0),
(-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 9,-9, 0, 0,-9, 9, 0, 0, 6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-6, 6, 0, 0, 6,-6, 0, 0,-3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0, 0, 0,-1, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9,-9, 0, 0,-9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 3, 0, 0,-6,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,-6, 0, 0, 3,-3, 0, 0, 4, 2, 0, 0, 2, 1, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3,-3, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 4, 0, 0,-2, 2, 0, 0,-2,-2, 0, 0,-1,-1, 0, 0),
( 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 9, 0,-9, 0,-9, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 3, 0,-6, 0,-3, 0, 6, 0,-6, 0, 3, 0,-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 2, 0, 2, 0, 1, 0),
(-27,27,27,-27,27,-27,-27,27,-18,-9,18, 9,18, 9,-18,-9,-18,18,-9, 9,18,-18, 9,-9,-18,18,18,-18,-9, 9, 9,-9,-12,-6,-6,-3,12, 6, 6, 3,-12,-6,12, 6,-6,-3, 6, 3,-12,12,-6, 6,-6, 6,-3, 3,-8,-4,-4,-2,-4,-2,-2,-1),
(18,-18,-18,18,-18,18,18,-18, 9, 9,-9,-9,-9,-9, 9, 9,12,-12, 6,-6,-12,12,-6, 6,12,-12,-12,12, 6,-6,-6, 6, 6, 6, 3, 3,-6,-6,-3,-3, 6, 6,-6,-6, 3, 3,-3,-3, 8,-8, 4,-4, 4,-4, 2,-2, 4, 4, 2, 2, 2, 2, 1, 1),
(-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 0,-3, 0, 3, 0, 3, 0,-4, 0, 4, 0,-2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-2, 0,-1, 0,-1, 0),
(18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6, 9,-9, 9,-9,-9, 9,-9, 9,12,-12,-12,12, 6,-6,-6, 6, 6, 3, 6, 3,-6,-3,-6,-3, 8, 4,-8,-4, 4, 2,-4,-2, 6,-6, 6,-6, 3,-3, 3,-3, 4, 2, 4, 2, 2, 1, 2, 1),
(-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-6, 6,-6, 6, 6,-6, 6,-6,-8, 8, 8,-8,-4, 4, 4,-4,-3,-3,-3,-3, 3, 3, 3, 3,-4,-4, 4, 4,-2,-2, 2, 2,-4, 4,-4, 4,-2, 2,-2, 2,-2,-2,-2,-2,-1,-1,-1,-1),
( 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
(-6, 6, 0, 0, 6,-6, 0, 0,-4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 4,-4, 0, 0,-4, 4, 0, 0, 2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-6, 6, 0, 0, 6,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4,-2, 0, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-3, 3, 0, 0,-3, 3, 0, 0,-2,-1, 0, 0,-2,-1, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4,-4, 0, 0,-4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0,-2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,-2, 0, 0, 2,-2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0),
(-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0,-6, 0, 6, 0, 6, 0,-6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-4, 0,-2, 0, 4, 0, 2, 0,-3, 0, 3, 0,-3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,-2, 0,-1, 0,-2, 0,-1, 0),
(18,-18,-18,18,-18,18,18,-18,12, 6,-12,-6,-12,-6,12, 6,12,-12, 6,-6,-12,12,-6, 6, 9,-9,-9, 9, 9,-9,-9, 9, 8, 4, 4, 2,-8,-4,-4,-2, 6, 3,-6,-3, 6, 3,-6,-3, 6,-6, 3,-3, 6,-6, 3,-3, 4, 2, 2, 1, 4, 2, 2, 1),
(-12,12,12,-12,12,-12,-12,12,-6,-6, 6, 6, 6, 6,-6,-6,-8, 8,-4, 4, 8,-8, 4,-4,-6, 6, 6,-6,-6, 6, 6,-6,-4,-4,-2,-2, 4, 4, 2, 2,-3,-3, 3, 3,-3,-3, 3, 3,-4, 4,-2, 2,-4, 4,-2, 2,-2,-2,-1,-1,-2,-2,-1,-1),
( 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
( 0, 0, 0, 0, 0, 0, 0, 0, 4, 0,-4, 0,-4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0,-2, 0,-2, 0, 2, 0,-2, 0, 2, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0),
(-12,12,12,-12,12,-12,-12,12,-8,-4, 8, 4, 8, 4,-8,-4,-6, 6,-6, 6, 6,-6, 6,-6,-6, 6, 6,-6,-6, 6, 6,-6,-4,-2,-4,-2, 4, 2, 4, 2,-4,-2, 4, 2,-4,-2, 4, 2,-3, 3,-3, 3,-3, 3,-3, 3,-2,-1,-2,-1,-2,-1,-2,-1),
( 8,-8,-8, 8,-8, 8, 8,-8, 4, 4,-4,-4,-4,-4, 4, 4, 4,-4, 4,-4,-4, 4,-4, 4, 4,-4,-4, 4, 4,-4,-4, 4, 2, 2, 2, 2,-2,-2,-2,-2, 2, 2,-2,-2, 2, 2,-2,-2, 2,-2, 2,-2, 2,-2, 2,-2, 1, 1, 1, 1, 1, 1, 1, 1)
])

def get_target_Y(z, y, x):
    Y = np.empty([z.shape[1],z.shape[0],64])
    Y[:,:,0] = 1.
    Y[:,:,1] = x
    Y[:,:,2] = x**2
    Y[:,:,3] = x**3
    Y[:,:,4] = y
    Y[:,:,5] = x*y
    Y[:,:,6] = x**2*y
    Y[:,:,7] = x**3*y
    Y[:,:,8] = y**2
    Y[:,:,9] = x*y**2
    Y[:,:,10] = x**2*y**2
    Y[:,:,11] = x**3*y**2
    Y[:,:,12] = y**3
    Y[:,:,13] = x*y**3
    Y[:,:,14] = x**2*y**3
    Y[:,:,15] = x**3*y**3
    

    Y[:,:,16] = z
    Y[:,:,17] = x*z
    Y[:,:,18] = x**2*z
    Y[:,:,19] = x**3*z
    Y[:,:,20] = y*z
    Y[:,:,21] = x*y*z
    Y[:,:,22] = x**2*y*z
    Y[:,:,23] = x**3*y*z
    Y[:,:,24] = y**2*z
    Y[:,:,25] = x*y**2*z
    Y[:,:,26] = x**2*y**2*z
    Y[:,:,27] = x**3*y**2*z
    Y[:,:,28] = y**3*z
    Y[:,:,29] = x*y**3*z
    Y[:,:,30] = x**2*y**3*z
    Y[:,:,31] = x**3*y**3*z
    
    Y[:,:,32] = z**2
    Y[:,:,33] = x*z**2
    Y[:,:,34] = x**2*z**2
    Y[:,:,35] = x**3*z**2
    Y[:,:,36] = y*z**2
    Y[:,:,37] = x*y*z**2
    Y[:,:,38] = x**2*y*z**2
    Y[:,:,39] = x**3*y*z**2
    Y[:,:,40] = y**2*z**2
    Y[:,:,41] = x*y**2*z**2
    Y[:,:,42] = x**2*y**2*z**2
    Y[:,:,43] = x**3*y**2*z**2
    Y[:,:,44] = y**3*z**2
    Y[:,:,45] = x*y**3*z**2
    Y[:,:,46] = x**2*y**3*z**2
    Y[:,:,47] = x**3*y**3*z**2
    
    Y[:,:,48] = z**3
    Y[:,:,49] = x*z**3
    Y[:,:,50] = x**2*z**3
    Y[:,:,51] = x**3*z**3
    Y[:,:,52] = y*z**3
    Y[:,:,53] = x*y*z**3
    Y[:,:,54] = x**2*y*z**3
    Y[:,:,55] = x**3*y*z**3
    Y[:,:,56] = y**2*z**3
    Y[:,:,57] = x*y**2*z**3
    Y[:,:,58] = x**2*y**2*z**3
    Y[:,:,59] = x**3*y**2*z**3
    Y[:,:,60] = y**3*z**3
    Y[:,:,61] = x*y**3*z**3
    Y[:,:,62] = x**2*y**3*z**3
    Y[:,:,63] = x**3*y**3*z**3
    return Y

def get_target_Y_1d(x, y, z):
    Y = np.zeros([1,64])
    Y[:,0] = 1.
    Y[:,1] = x
    Y[:,2] = x**2
    Y[:,3] = x**3
    Y[:,4] = y
    Y[:,5] = x*y
    Y[:,6] = x**2*y
    Y[:,7] = x**3*y
    Y[:,8] = y**2
    Y[:,9] = x*y**2
    Y[:,10] = x**2*y**2
    Y[:,11] = x**3*y**2
    Y[:,12] = y**3
    Y[:,13] = x*y**3
    Y[:,14] = x**2*y**3
    Y[:,15] = x**3*y**3
    

    Y[:,16] = z
    Y[:,17] = x*z
    Y[:,18] = x**2*z
    Y[:,19] = x**3*z
    Y[:,20] = y*z
    Y[:,21] = x*y*z
    Y[:,22] = x**2*y*z
    Y[:,23] = x**3*y*z
    Y[:,24] = y**2*z
    Y[:,25] = x*y**2*z
    Y[:,26] = x**2*y**2*z
    Y[:,27] = x**3*y**2*z
    Y[:,28] = y**3*z
    Y[:,29] = x*y**3*z
    Y[:,30] = x**2*y**3*z
    Y[:,31] = x**3*y**3*z
    
    Y[:,32] = z**2
    Y[:,33] = x*z**2
    Y[:,34] = x**2*z**2
    Y[:,35] = x**3*z**2
    Y[:,36] = y*z**2
    Y[:,37] = x*y*z**2
    Y[:,38] = x**2*y*z**2
    Y[:,39] = x**3*y*z**2
    Y[:,40] = y**2*z**2
    Y[:,41] = x*y**2*z**2
    Y[:,42] = x**2*y**2*z**2
    Y[:,43] = x**3*y**2*z**2
    Y[:,44] = y**3*z**2
    Y[:,45] = x*y**3*z**2
    Y[:,46] = x**2*y**3*z**2
    Y[:,47] = x**3*y**3*z**2
    
    Y[:,48] = z**3
    Y[:,49] = x*z**3
    Y[:,50] = x**2*z**3
    Y[:,51] = x**3*z**3
    Y[:,52] = y*z**3
    Y[:,53] = x*y*z**3
    Y[:,54] = x**2*y*z**3
    Y[:,55] = x**3*y*z**3
    Y[:,56] = y**2*z**3
    Y[:,57] = x*y**2*z**3
    Y[:,58] = x**2*y**2*z**3
    Y[:,59] = x**3*y**2*z**3
    Y[:,60] = y**3*z**3
    Y[:,61] = x*y**3*z**3
    Y[:,62] = x**2*y**3*z**3
    Y[:,63] = x**3*y**3*z**3
    return Y

def tricubic_derivatives(volume):
    shape = volume.shape
    dest = np.empty(shape)
    tricubic_derivative_dict = np.empty([shape[0]+30,shape[1]+30,shape[2]+30,64])
    for i in xrange(-15,shape[0]+15):
        for j in xrange(-15,shape[1]+15):
            for k in xrange(-15,shape[2]+15):
                # Take care of boundary conditions
                x1 = i
                y1 = j
                z1 = k
                x0 = x1 - 1
                x2 = x1 + 1
                x3 = x2 + 1
                y0 = y1 - 1
                y2 = y1 + 1
                y3 = y2 + 1
                z0 = z1 - 1
                z2 = z1 + 1
                z3 = z2 + 1

                # Wrap Around
                x0 = (x0 + volume.shape[1]) % volume.shape[1]
                x1 = (x1 + volume.shape[1]) % volume.shape[1]
                x2 = (x2 + volume.shape[1]) % volume.shape[1]
                x3 = (x3 + volume.shape[1]) % volume.shape[1]

                y0 = (y0 + volume.shape[0]) % volume.shape[0]
                y1 = (y1 + volume.shape[0]) % volume.shape[0]
                y2 = (y2 + volume.shape[0]) % volume.shape[0]
                y3 = (y3 + volume.shape[0]) % volume.shape[0]

                z0 = (z0 + volume.shape[2]) % volume.shape[2]
                z1 = (z1 + volume.shape[2]) % volume.shape[2] 
                z2 = (z2 + volume.shape[2]) % volume.shape[2]
                z3 = (z3 + volume.shape[2]) % volume.shape[2] 

                # Compute vector Y from known points
                Y = np.zeros([64,])
                # values of f(x,y,z) at each corner.
                Y[0] = volume[y1,x1,z1]
                Y[1] = volume[y1,x1,z2]
                Y[2] = volume[y1,x2,z1]
                Y[3] = volume[y1,x2,z2]
                Y[4] = volume[y2,x1,z1]
                Y[5] = volume[y2,x1,z2]
                Y[6] = volume[y2,x2,z1]
                Y[7] = volume[y2,x2,z2]

                # values of df/dx at each corner.
                Y[8] = ((volume[y1,x1,z2]-volume[y1,x1,z0])/2.)
                Y[9] = ((volume[y1,x1,z3]-volume[y1,x1,z1])/2.)
                Y[10] = ((volume[y1,x2,z2]-volume[y1,x2,z0])/2.)
                Y[11] = ((volume[y1,x2,z3]-volume[y1,x2,z1])/2.)
                Y[12] = ((volume[y2,x1,z2]-volume[y2,x1,z0])/2.)
                Y[13] = ((volume[y2,x1,z3]-volume[y2,x1,z1])/2.)
                Y[14] = ((volume[y2,x2,z2]-volume[y2,x2,z0])/2.)
                Y[15] = ((volume[y2,x2,z3]-volume[y2,x2,z1])/2.)

                # values of df/dy at each corner.
                Y[16] = ((volume[y1,x2,z1]-volume[y1,x0,z1])/2.)
                Y[17] = ((volume[y1,x2,z2]-volume[y1,x0,z2])/2.)
                Y[18] = ((volume[y1,x3,z1]-volume[y1,x1,z1])/2.)
                Y[19] = ((volume[y1,x3,z2]-volume[y1,x1,z2])/2.)
                Y[20] = ((volume[y2,x2,z1]-volume[y2,x0,z1])/2.)
                Y[21] = ((volume[y2,x2,z2]-volume[y2,x0,z2])/2.)
                Y[22] = ((volume[y2,x3,z1]-volume[y2,x1,z1])/2.)
                Y[23] = ((volume[y2,x3,z2]-volume[y2,x1,z2])/2.)

                # values of df/dz at each corner.
                Y[24] = ((volume[y2,x1,z1]-volume[y0,x1,z1])/2.)
                Y[25] = ((volume[y2,x1,z2]-volume[y0,x1,z2])/2.)
                Y[26] = ((volume[y2,x2,z1]-volume[y0,x2,z1])/2.)
                Y[27] = ((volume[y2,x2,z2]-volume[y0,x2,z2])/2.)
                Y[28] = ((volume[y3,x1,z1]-volume[y1,x1,z1])/2.)
                Y[29] = ((volume[y3,x1,z2]-volume[y1,x1,z2])/2.)
                Y[30] = ((volume[y3,x2,z1]-volume[y1,x2,z1])/2.)
                Y[31] = ((volume[y3,x2,z2]-volume[y1,x2,z2])/2.)

                # values of d2f/dxdy at each corner.
                Y[32] = ((volume[y1,x2,z2]-volume[y1,x0,z2]-volume[y1,x2,z0]+volume[y1,x0,z0])/4.)
                Y[33] = ((volume[y1,x2,z3]-volume[y1,x0,z3]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
                Y[34] = ((volume[y1,x3,z2]-volume[y1,x1,z2]-volume[y1,x3,z0]+volume[y1,x1,z0])/4.)
                Y[35] = ((volume[y1,x3,z3]-volume[y1,x1,z3]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
                Y[36] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y2,x2,z0]+volume[y2,x0,z0])/4.)
                Y[37] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y2,x2,z1]+volume[y2,x0,z1])/4.)
                Y[38] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y2,x3,z0]+volume[y2,x1,z0])/4.)
                Y[39] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y2,x3,z1]+volume[y2,x1,z1])/4.)

                # values of d2f/dxdz at each corner.
                Y[40] = ((volume[y2,x1,z2]-volume[y0,x1,z2]-volume[y2,x1,z0]+volume[y0,x1,z0])/4.)
                Y[41] = ((volume[y2,x1,z3]-volume[y0,x1,z3]-volume[y2,x1,z1]+volume[y0,x1,z1])/4.)
                Y[42] = ((volume[y2,x2,z2]-volume[y0,x2,z2]-volume[y2,x2,z0]+volume[y0,x2,z0])/4.)
                Y[43] = ((volume[y2,x2,z3]-volume[y0,x2,z3]-volume[y2,x2,z1]+volume[y0,x2,z1])/4.)
                Y[44] = ((volume[y3,x1,z2]-volume[y1,x1,z2]-volume[y3,x1,z0]+volume[y1,x1,z0])/4.)
                Y[45] = ((volume[y3,x1,z3]-volume[y1,x1,z3]-volume[y3,x1,z1]+volume[y1,x1,z1])/4.)
                Y[46] = ((volume[y3,x2,z2]-volume[y1,x2,z2]-volume[y3,x2,z0]+volume[y1,x2,z0])/4.)
                Y[47] = ((volume[y3,x2,z3]-volume[y1,x2,z3]-volume[y3,x2,z1]+volume[y1,x2,z1])/4.)

                # values of d2f/dydz at each corner.
                Y[48] = ((volume[y2,x2,z1]-volume[y2,x0,z1]-volume[y0,x2,z1]+volume[y0,x0,z1])/4.)
                Y[49] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y0,x2,z2]+volume[y0,x0,z2])/4.)
                Y[50] = ((volume[y2,x3,z1]-volume[y2,x1,z1]-volume[y0,x3,z1]+volume[y0,x1,z1])/4.)
                Y[51] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y0,x3,z2]+volume[y0,x1,z2])/4.)
                Y[52] = ((volume[y3,x2,z1]-volume[y3,x0,z1]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
                Y[53] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y1,x2,z2]+volume[y1,x0,z2])/4.)
                Y[54] = ((volume[y3,x3,z1]-volume[y3,x1,z1]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
                Y[55] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y1,x3,z2]+volume[y1,x1,z2])/4.)

                # values of d3f/dxdydz at each corner.
                Y[56] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y2,x2,z0]+volume[y2,x0,z0])
                          -(volume[y0,x2,z2]-volume[y0,x0,z2]-volume[y0,x2,z0]+volume[y0,x0,z0]))/8.
                Y[57] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y2,x2,z1]+volume[y2,x0,z1])
                          -(volume[y0,x2,z3]-volume[y0,x0,z3]-volume[y0,x2,z1]+volume[y0,x0,z1]))/8.
                Y[58] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y2,x3,z0]+volume[y2,x1,z0])
                          -(volume[y0,x3,z2]-volume[y0,x1,z2]-volume[y0,x3,z0]+volume[y0,x1,z0]))/8.
                Y[59] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y2,x3,z1]+volume[y2,x1,z1])
                          -(volume[y0,x3,z3]-volume[y0,x1,z3]-volume[y0,x3,z1]+volume[y0,x1,z1]))/8.

                Y[60] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y3,x2,z0]+volume[y3,x0,z0])
                          -(volume[y1,x2,z2]-volume[y1,x0,z2]-volume[y1,x2,z0]+volume[y1,x0,z0]))/8.
                Y[61] = ((volume[y3,x2,z3]-volume[y3,x0,z3]-volume[y3,x2,z1]+volume[y3,x0,z1])
                          -(volume[y1,x2,z3]-volume[y1,x0,z3]-volume[y1,x2,z1]+volume[y1,x0,z1]))/8.
                Y[62] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y3,x3,z0]+volume[y3,x1,z0])
                          -(volume[y1,x3,z2]-volume[y1,x1,z2]-volume[y1,x3,z0]+volume[y1,x1,z0]))/8.
                Y[63] = ((volume[y3,x3,z3]-volume[y3,x1,z3]-volume[y3,x3,z1]+volume[y3,x1,z1])
                          -(volume[y1,x3,z3]-volume[y1,x1,z3]-volume[y1,x3,z1]+volume[y1,x1,z1]))/8.

                #tricubic_derivative_dict[(j,i,k)] = np.dot(X_inv,Y)
                tricubic_derivative_dict[j+15,i+15,k+15,:] = np.dot(X_inv,Y)
    return tricubic_derivative_dict
'''
                # Clip Range
                x0 = np.clip(x0, 0, volume.shape[1]-1)
                x1 = np.clip(x1, 0, volume.shape[1]-1)
                x2 = np.clip(x2, 0, volume.shape[1]-1)
                x3 = np.clip(x3, 0, volume.shape[1]-1)
                y0 = np.clip(y0, 0, volume.shape[0]-1)
                y1 = np.clip(y1, 0, volume.shape[0]-1)
                y2 = np.clip(y2, 0, volume.shape[0]-1)
                y3 = np.clip(y3, 0, volume.shape[0]-1)
                z0 = np.clip(z0, 0, volume.shape[2]-1)
                z1 = np.clip(z1, 0, volume.shape[2]-1)
                z2 = np.clip(z2, 0, volume.shape[2]-1)
                z3 = np.clip(z3, 0, volume.shape[2]-1)
'''
# Tricubic interpolation
def tricubic_interp(shape, derivatives, x, y, z):
    '''
    shape: Shape of the Volume to be interpolated
    derivatives: precomputed derivatives for the volume
    x,y,z: point at which to be interpolated
    '''
    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    z1 = np.floor(z).astype(int)

    # load in precomputed first and second derivatives for this volume
    #Y = derivatives[(y1,x1,z1)]

    # Compute A
    A = derivatives[y1+15,x1+15,z1+15]
    # get vector Y from points that need to be interpolated
    target_Y = get_target_Y(y-y1, x-x1, z-z1)
    # compute result
    result = np.sum(target_Y * A, axis=2)
    return result

# Tricubic interpolation
def tricubic_interp_1d(shape, derivatives, x, y, z):
    '''
    shape: Shape of the Volume to be interpolated
    derivatives: precomputed derivatives for the volume
    x,y,z: point at which to be interpolated
    '''
    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    z1 = np.floor(z).astype(int)

    # load in precomputed first and second derivatives for this volume
    A = derivatives[(y1+15,x1+15,z1+15)]
    print A
    # Compute A
    #A = np.dot(X_inv,Y)
    # get vector Y from points that need to be interpolated
    target_Y = get_target_Y_1d(y-y1, x-x1, z-z1)
    # compute result
    print target_Y
    result = np.dot(target_Y, A)
    return result[0]


# Trilinear interplation
def trilinear_interp(volume, x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # find the closest grid of the target points
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1
    volume_shape = volume.shape
    # Clip
    x0 = x0.clip(0, volume_shape[0]-1)
    x1 = x1.clip(0, volume_shape[0]-1)
    y0 = y0.clip(0, volume_shape[1]-1)
    y1 = y1.clip(0, volume_shape[1]-1)
    z0 = z0.clip(0, volume_shape[2]-1)
    z1 = z1.clip(0, volume_shape[2]-1)    

    # define some coefficients
    xd = x-x0
    yd = y-y0
    zd = z-z0
    
    # set up for the bilinear interpolation
    C00 = volume[y0,x0,z0]*(1-xd) + volume[y0,x1,z0]*xd
    C10 = volume[y1,x0,z0]*(1-xd) + volume[y1,x1,z0]*xd
    
    C01 = volume[y0,x0,z1]*(1-xd) + volume[y0,x1,z1]*xd
    C11 = volume[y1,x0,z1]*(1-xd) + volume[y1,x1,z1]*xd
    
    C0 = C00*(1-yd) + C10*yd
    C1 = C01*(1-yd) + C11*yd
    
    C = C0*(1-zd) + C1*zd
    return C

def to_radian(theta):
    '''
    Convert theta from degrees to radians
    '''
    return theta*np.pi/180.

def rotation_matrix_zyx(gamma, beta, alpha):
    """
    Return the rotation matrix associated with counterclockwise rotation 
    about x axis by gamma degrees
    about y axis by beta degrees
    about z axis by alpha degrees
    """
    # convert degrees to radians
    gamma = to_radian(gamma)
    beta = to_radian(beta)
    alpha = to_radian(alpha)
    
    rz = np.array([[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]])
    ry = np.array([[np.cos(beta),0, np.sin(beta)],[0, 1, 0],[-np.sin(beta),0,np.cos(beta)]])
    rx = np.array([[1,0,0],[0,np.cos(gamma),-np.sin(gamma)],[0,np.sin(gamma),np.cos(gamma)]])
    return (rz.dot(ry)).dot(rx)


# Rotate the coordinates using rotation matrix
def rotate_coords_3d_matrix(x, y, z, gamma, beta, alpha, ox, oy, oz):
    '''
    Rotate arrays of coordinates x, y and z about the point (ox, oy, oz)
    about x axis by gamma degrees
    about y axis by beta degrees
    about z axis by alpha degrees
    '''
    R = rotation_matrix_zyx(gamma, beta, alpha)
    #tmp = np.vstack([a,np.zeros([1,3])])
    x, y, z = x - ox, y - oy, z - oz
    return ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox, 
            (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy, 
            (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz)

# Rotate the coordinates using axis-angle rotation
def rotate_coords_3d(x, y, z, theta, wx, wy, wz, ox,oy,oz):
    theta = to_radian(theta)
    # make sure w is a unit vetor:
    if ((wx**2 + wy**2 + wz**2 != 1) and (wx**2 + wy**2 + wz**2 != 0)):
        norm = np.sqrt(wx**2 + wy**2 + wz**2)
        wx = wx/norm
        wy = wy/norm
        wz = wz/norm
    s,c = np.sin(theta),np.cos(theta)
    x, y, z = x - ox, y - oy, z - oz
    #print x.shape,y.shape,z.shape
    rotx = c*x+s*(wy*z-wz*y)+(1-c)*(wx*x+wy*y+wz*z)*wx + ox
    roty = c*y+s*(wz*x-wx*z)+(1-c)*(wx*x+wy*y+wz*z)*wy + oy
    rotz = c*z+s*(wx*y-wy*x)+(1-c)*(wx*x+wy*y+wz*z)*wz + oz
    return (rotx,roty,rotz)

# Construct a rotation matrix from quarterions
def rotation_matrix_fromq(theta, ui, uj, uk):
    c = np.cos(theta)
    s = np.sin(theta)
    rotMatrix = np.zeros([3,3])
    
    rotMatrix[0][0] = c + ui**2*(1-c)
    rotMatrix[0][1] = ui*uj*(1-c) - uk*s
    rotMatrix[0][2] = ui*uk*(1-c) + uj*s

    rotMatrix[1][0] = uj*ui*(1-c) + uk*s
    rotMatrix[1][1] = c + uj**2*(1-c)
    rotMatrix[1][2] = uj*uk*(1-c) - ui*s

    rotMatrix[2][0] = uk*ui*(1-c) - uj*s
    rotMatrix[2][1] = uk*uj*(1-c) + ui*s
    rotMatrix[2][2] = c + uk**2*(1-c)
    
    return rotMatrix

# Rotate the coordinates using quarternions
def rotate_coords_3d_matrix_fromq(x, y, z, theta, wx, wy, wz, ox,oy,oz):
    '''
    Rotate arrays of coordinates x, y and z about the point (ox, oy, oz)
    about x axis by gamma degrees
    about y axis by beta degrees
    about z axis by alpha degrees
    '''
    R = rotation_matrix_fromq(theta, wx, wy, wz)
    #tmp = np.vstack([a,np.zeros([1,3])])
    x, y, z = x - ox, y - oy, z - oz
    return ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox, 
            (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy, 
            (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz)


def volrotate_trilinear_matrix(volume_org, gamma, beta, alpha,xx,yy,zz):
    '''
    wx, wy, wz is the unit vector describing the axis of rotation
    theta is the rotation angle
    '''
    volume = volume_org.copy()
    shape = volume.shape
    # find center of the volume
    ox = shape[1]/2.-0.5
    oy = shape[0]/2.-0.5
    oz = shape[2]/2.-0.5
    
    if(shape[0] == 26): res = '10mm'
    elif(shape[0] == 32): res = '8mm'
    else: res = '6_4mm'

    #xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    
    dest_x, dest_y, dest_z = rotate_coords_3d_matrix(xx, yy, zz, gamma, beta, alpha, ox, oy, oz)
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    return dest

def volrotate_trilinear(volume_org, theta, wx, wy, wz,xx=None,yy=None,zz=None):
    '''
    wx, wy, wz is the unit vector describing the axis of rotation
    theta is the rotation angle
    '''
    volume = volume_org.copy()
    shape = volume.shape
    # find center of the volume
    ox = shape[1]/2.-0.5
    oy = shape[0]/2.-0.5
    oz = shape[2]/2.-0.5

    if(shape[0] == 26): res = '10'
    elif(shape[0] == 32): res = '8'
    else: res = '6_4'

    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    
    dest_x, dest_y, dest_z = rotate_coords_3d(xx, yy, zz, theta, wx, wy, wz, ox, oy, oz)
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    return dest

def volrotate_tricubic_matrix(volume_shape, tricubic_cache, gamma, beta, alpha,xx,yy,zz):
    '''
    volume_shape: shape of the input volume
    tricubic_cache: precomputed derivative values for each point
    wx, wy, wz is the unit vector describing the axis of rotation
    theta is the rotation angle
    '''
    # find center of the volume
    ox = volume_shape[1]/2.-0.5
    oy = volume_shape[0]/2.-0.5
    oz = volume_shape[2]/2.-0.5
    
    if(volume_shape[0] == 26): res = '10'
    elif(volume_shape[0] == 32): res = '8'
    else: res = '6_4'

    #xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    
    dest_x, dest_y, dest_z = rotate_coords_3d_matrix(xx, yy, zz, gamma, beta, alpha, ox, oy, oz)

    dest = np.empty(volume_shape)
    for i in xrange(volume_shape[0]):
        dest[i,:,:] = tricubic_interp(volume_shape,tricubic_cache,dest_x[i,:,:],dest_y[i,:,:],dest_z[i,:,:]) 
    return dest

def volrotate_tricubic(volume_shape, tricubic_cache, theta, wx, wy, wz,xx=None,yy=None,zz=None,random_points = False):
    '''
    volume_shape: shape of the input volume
    tricubic_cache: precomputed derivative values for each point
    wx, wy, wz is the unit vector describing the axis of rotation
    theta is the rotation angle
    xx,yy,zz: loaded meshgrid
    '''
    # find center of the volume
    ox = volume_shape[1]/2.-0.5
    oy = volume_shape[0]/2.-0.5
    oz = volume_shape[2]/2.-0.5
    
    if(volume_shape[0] == 26): res = '10'
    elif(volume_shape[0] == 32): res = '8'
    else: res = '6_4'
    
    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    dest_x, dest_y, dest_z = rotate_coords_3d(xx, yy, zz, theta, wx, wy, wz, ox, oy, oz)
    if random_points:
        dest = np.empty(dest_x.shape)
        for i in xrange(dest_x.shape[0]):
            dest[i] = tricubic_interp_1d(volume_shape,tricubic_cache,dest_x[i],dest_y[i],dest_z[i]) 
    else:
        dest = np.empty(volume_shape)
        for i in xrange(volume_shape[0]):
            dest[i,:,:] = tricubic_interp(volume_shape,tricubic_cache,dest_x[i,:,:],dest_y[i,:,:],dest_z[i,:,:]) 
    return dest

def rot_cost_func_3d(vol1, vol2, thetas, wx, wy, wz, xx,yy,zz,interpolation = 'trilinear'):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    xx,yy,zz: loaded meshgrid
    wx, wy, wz is the unit vector describing the axis of rotation
    '''
    cost_func = np.zeros([len(thetas),])
    if (interpolation == 'trilinear'):
        for idx, t in enumerate(thetas):
            new_vol2 = volrotate_trilinear(vol2,t,wx,wy,wz,xx,yy,zz)
            cost_func[idx] = cf_ssd(new_vol2,vol1)
    if(interpolation == 'tricubic'):
        tricubic_cache = tricubic_derivatives(vol2)
        for idx, t in enumerate(thetas):
            new_vol2 = volrotate_tricubic(vol2.shape, tricubic_cache, t, wx,wy,wz,xx,yy,zz)
            cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func





'''
def tricubic_interp(volume, x, y, z):

    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    
    # Take care of boundary conditions
    x0 = x1 - 1
    x2 = x1 + 1
    x3 = x2 + 1
    
    y1 = np.floor(y).astype(int)
    y0 = y1 - 1
    y2 = y1 + 1
    y3 = y2 + 1
        
    z1 = np.floor(z).astype(int)
    z0 = z1 - 1
    z2 = z1 + 1
    z3 = z2 + 1
    
    # we need to clip the range 
    x0 = np.clip(x0, 0, volume.shape[1]-1)
    x1 = np.clip(x1, 0, volume.shape[1]-1)
    x2 = np.clip(x2, 0, volume.shape[1]-1)
    x3 = np.clip(x3, 0, volume.shape[1]-1)
    y0 = np.clip(y0, 0, volume.shape[0]-1)
    y1 = np.clip(y1, 0, volume.shape[0]-1)
    y2 = np.clip(y2, 0, volume.shape[0]-1)
    y3 = np.clip(y3, 0, volume.shape[0]-1)
    z0 = np.clip(z0, 0, volume.shape[2]-1)
    z1 = np.clip(z1, 0, volume.shape[2]-1)
    z2 = np.clip(z2, 0, volume.shape[2]-1)
    z3 = np.clip(z3, 0, volume.shape[2]-1)
    
    # compute the vector of coefficients A
    # first compute vector Y from known points to solve for A
    Y = np.zeros([64,])
    # values of f(x,y,z) at each corner.
    Y[0] = volume[y1,x1,z1]
    Y[1] = volume[y1,x2,z1]
    Y[2] = volume[y2,x1,z1]
    Y[3] = volume[y2,x2,z1]
    Y[4] = volume[y1,x1,z2]
    Y[5] = volume[y1,x2,z2]
    Y[6] = volume[y2,x1,z2]
    Y[7] = volume[y2,x2,z2]
    
    # values of df/dx at each corner.
    Y[8] = ((volume[y1,x2,z1]-volume[y1,x0,z1])/2.)
    Y[9] = ((volume[y1,x3,z1]-volume[y1,x1,z1])/2.)
    Y[10] = ((volume[y2,x2,z1]-volume[y2,x0,z1])/2.)
    Y[11] = ((volume[y2,x3,z1]-volume[y2,x1,z1])/2.)
    Y[12] = ((volume[y1,x2,z2]-volume[y1,x0,z2])/2.)
    Y[13] = ((volume[y1,x3,z2]-volume[y1,x1,z2])/2.)
    Y[14] = ((volume[y2,x2,z2]-volume[y2,x0,z2])/2.)
    Y[15] = ((volume[y2,x3,z2]-volume[y2,x1,z2])/2.)

    # values of df/dy at each corner.
    Y[16] = ((volume[y2,x1,z1]-volume[y0,x1,z1])/2.)
    Y[17] = ((volume[y2,x2,z1]-volume[y0,x2,z1])/2.)
    Y[18] = ((volume[y3,x1,z1]-volume[y1,x1,z1])/2.)
    Y[19] = ((volume[y3,x2,z1]-volume[y1,x2,z1])/2.)
    Y[20] = ((volume[y2,x1,z2]-volume[y0,x1,z2])/2.)
    Y[21] = ((volume[y2,x2,z2]-volume[y0,x2,z2])/2.)
    Y[22] = ((volume[y3,x1,z2]-volume[y1,x1,z2])/2.)
    Y[23] = ((volume[y3,x2,z2]-volume[y1,x2,z2])/2.)
    
    # values of df/dz at each corner.
    Y[24] = ((volume[y1,x1,z2]-volume[y1,x1,z0])/2.)
    Y[25] = ((volume[y1,x2,z2]-volume[y1,x2,z0])/2.)
    Y[26] = ((volume[y2,x1,z2]-volume[y2,x1,z0])/2.)
    Y[27] = ((volume[y2,x2,z2]-volume[y2,x2,z0])/2.)
    Y[28] = ((volume[y1,x1,z3]-volume[y1,x1,z1])/2.)
    Y[29] = ((volume[y1,x2,z3]-volume[y1,x2,z1])/2.)
    Y[30] = ((volume[y2,x1,z3]-volume[y2,x1,z1])/2.)
    Y[31] = ((volume[y2,x2,z3]-volume[y2,x2,z1])/2.)
    
    # values of d2f/dxdy at each corner.
    Y[32] = ((volume[y2,x2,z1]-volume[y2,x0,z1]-volume[y0,x2,z1]+volume[y0,x0,z1])/4.)
    Y[33] = ((volume[y2,x3,z1]-volume[y2,x1,z1]-volume[y0,x3,z1]+volume[y0,x1,z1])/4.)
    Y[34] = ((volume[y3,x2,z1]-volume[y3,x0,z1]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
    Y[35] = ((volume[y3,x3,z1]-volume[y3,x1,z1]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
    Y[36] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y0,x2,z2]+volume[y0,x0,z2])/4.)
    Y[37] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y0,x3,z2]+volume[y0,x1,z2])/4.)
    Y[38] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y1,x2,z2]+volume[y1,x0,z2])/4.)
    Y[39] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y1,x3,z2]+volume[y1,x1,z2])/4.)

    # values of d2f/dxdz at each corner.
    Y[40] = ((volume[y1,x2,z2]-volume[y1,x0,z2]-volume[y1,x2,z0]+volume[y1,x0,z0])/4.)
    Y[41] = ((volume[y1,x3,z2]-volume[y1,x1,z2]-volume[y1,x3,z0]+volume[y1,x1,z0])/4.)
    Y[42] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y2,x2,z0]+volume[y2,x0,z0])/4.)
    Y[43] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y2,x3,z0]+volume[y2,x1,z0])/4.)
    Y[44] = ((volume[y1,x2,z3]-volume[y1,x0,z3]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
    Y[45] = ((volume[y1,x3,z3]-volume[y1,x1,z3]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
    Y[46] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y2,x2,z1]+volume[y2,x0,z1])/4.)
    Y[47] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y2,x3,z1]+volume[y2,x1,z1])/4.)
    
    # values of d2f/dydz at each corner.
    Y[48] = ((volume[y2,x1,z2]-volume[y0,x1,z2]-volume[y2,x1,z0]+volume[y0,x1,z0])/4.)
    Y[49] = ((volume[y2,x2,z2]-volume[y0,x2,z2]-volume[y2,x2,z0]+volume[y0,x2,z0])/4.)
    Y[50] = ((volume[y3,x1,z2]-volume[y1,x1,z2]-volume[y3,x1,z0]+volume[y1,x1,z0])/4.)
    Y[51] = ((volume[y3,x2,z2]-volume[y1,x2,z2]-volume[y3,x2,z0]+volume[y1,x2,z0])/4.)
    Y[52] = ((volume[y2,x1,z3]-volume[y0,x1,z3]-volume[y2,x1,z1]+volume[y0,x1,z1])/4.)
    Y[53] = ((volume[y2,x2,z3]-volume[y0,x2,z3]-volume[y2,x2,z1]+volume[y0,x2,z1])/4.)
    Y[54] = ((volume[y3,x1,z3]-volume[y1,x1,z3]-volume[y3,x1,z1]+volume[y1,x1,z1])/4.)
    Y[55] = ((volume[y3,x2,z3]-volume[y1,x2,z3]-volume[y3,x2,z1]+volume[y1,x2,z1])/4.)
    
    # values of d3f/dxdydz at each corner.
    Y[56] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y0,x2,z2]+volume[y0,x0,z2]
              -volume[y2,x2,z0]-volume[y2,x0,z0]-volume[y0,x2,z0]+volume[y0,x0,z0])/8.)
    Y[57] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y0,x3,z2]+volume[y0,x1,z2]
              -volume[y2,x3,z0]-volume[y2,x1,z0]-volume[y0,x3,z0]+volume[y0,x1,z0])/8.)
    Y[58] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y1,x2,z2]+volume[y1,x0,z2]
              -volume[y3,x2,z0]-volume[y3,x0,z0]-volume[y1,x2,z0]+volume[y1,x0,z0])/8.)
    Y[59] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y1,x3,z2]+volume[y1,x1,z2]
              -volume[y3,x3,z0]-volume[y3,x1,z0]-volume[y1,x3,z0]+volume[y1,x1,z0])/8.)

    Y[60] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y0,x2,z3]+volume[y0,x0,z3]
              -volume[y2,x2,z1]-volume[y2,x0,z1]-volume[y0,x2,z1]+volume[y0,x0,z1])/8.)
    Y[61] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y0,x3,z3]+volume[y0,x1,z3]
              -volume[y2,x3,z1]-volume[y2,x1,z1]-volume[y0,x3,z1]+volume[y0,x1,z1])/8.)
    Y[62] = ((volume[y3,x2,z3]-volume[y3,x0,z3]-volume[y1,x2,z3]+volume[y1,x0,z3]
              -volume[y3,x2,z1]-volume[y3,x0,z1]-volume[y1,x2,z1]+volume[y1,x0,z1])/8.)
    Y[63] = ((volume[y3,x3,z3]-volume[y3,x1,z3]-volume[y1,x3,z3]+volume[y1,x1,z3]
              -volume[y3,x3,z1]-volume[y3,x1,z1]-volume[y1,x3,z1]+volume[y1,x1,z1])/8.)    

    # Compute A
    A = np.dot(X_inv,Y)
    # get vector Y from points that need to be interpolated
    target_Y = get_target_Y(x-x1, y-y1, z-z1)
    # compute result
    result = np.dot(target_Y, A)
    return result[0]
'''
'''
# Tricubic interpolation
def tricubic_interp_A(volume, x, y, z):

    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    
    # Take care of boundary conditions
    x0 = x1 - 1
    x2 = x1 + 1
    x3 = x2 + 1
    
    y1 = np.floor(y).astype(int)
    y0 = y1 - 1
    y2 = y1 + 1
    y3 = y2 + 1
        
    z1 = np.floor(z).astype(int)
    z0 = z1 - 1
    z2 = z1 + 1
    z3 = z2 + 1
    #print '***before clip x1,y1,z1:',x1,y1,z1,
    # print '***before clip x0,y0,z0:',x0,y0,z0
    # print '***before clip x2,y2,z2:',x2,y2,z2
    # print '***before clip x3,y3,z3:',x3,y3,z3
    # we need to clip the range 
    x0 = np.clip(x0, 0, volume.shape[1]-1)
    x1 = np.clip(x1, 0, volume.shape[1]-1)
    x2 = np.clip(x2, 0, volume.shape[1]-1)
    x3 = np.clip(x3, 0, volume.shape[1]-1)
    y0 = np.clip(y0, 0, volume.shape[0]-1)
    y1 = np.clip(y1, 0, volume.shape[0]-1)
    y2 = np.clip(y2, 0, volume.shape[0]-1)
    y3 = np.clip(y3, 0, volume.shape[0]-1)
    z0 = np.clip(z0, 0, volume.shape[2]-1)
    z1 = np.clip(z1, 0, volume.shape[2]-1)
    z2 = np.clip(z2, 0, volume.shape[2]-1)
    z3 = np.clip(z3, 0, volume.shape[2]-1)
    # print 'after clip x1,y1,z1:',x1,y1,z1
    # print 'after clip x0,y0,z0:',x0,y0,z0
    # print 'after clip x2,y2,z2:',x2,y2,z2
    # print 'after clip x3,y3,z3:',x3,y3,z3

    # compute the vector of coefficients A
    # first compute vector Y from known points to solve for A
    Y = np.zeros([64,])
    # values of f(x,y,z) at each corner.
    Y[0] = volume[y1,x1,z1]
    Y[1] = volume[y1,x2,z1]
    Y[2] = volume[y2,x1,z1]
    Y[3] = volume[y2,x2,z1]
    Y[4] = volume[y1,x1,z2]
    Y[5] = volume[y1,x2,z2]
    Y[6] = volume[y2,x1,z2]
    Y[7] = volume[y2,x2,z2]
    
    # values of df/dx at each corner.
    Y[8] = ((volume[y1,x2,z1]-volume[y1,x0,z1])/2.)
    Y[9] = ((volume[y1,x3,z1]-volume[y1,x1,z1])/2.)
    Y[10] = ((volume[y2,x2,z1]-volume[y2,x0,z1])/2.)
    Y[11] = ((volume[y2,x3,z1]-volume[y2,x1,z1])/2.)
    Y[12] = ((volume[y1,x2,z2]-volume[y1,x0,z2])/2.)
    Y[13] = ((volume[y1,x3,z2]-volume[y1,x1,z2])/2.)
    Y[14] = ((volume[y2,x2,z2]-volume[y2,x0,z2])/2.)
    Y[15] = ((volume[y2,x3,z2]-volume[y2,x1,z2])/2.)

    # values of df/dy at each corner.
    Y[16] = ((volume[y2,x1,z1]-volume[y0,x1,z1])/2.)
    Y[17] = ((volume[y2,x2,z1]-volume[y0,x2,z1])/2.)
    Y[18] = ((volume[y3,x1,z1]-volume[y1,x1,z1])/2.)
    Y[19] = ((volume[y3,x2,z1]-volume[y1,x2,z1])/2.)
    Y[20] = ((volume[y2,x1,z2]-volume[y0,x1,z2])/2.)
    Y[21] = ((volume[y2,x2,z2]-volume[y0,x2,z2])/2.)
    Y[22] = ((volume[y3,x1,z2]-volume[y1,x1,z2])/2.)
    Y[23] = ((volume[y3,x2,z2]-volume[y1,x2,z2])/2.)
    
    # values of df/dz at each corner.
    Y[24] = ((volume[y1,x1,z2]-volume[y1,x1,z0])/2.)
    Y[25] = ((volume[y1,x2,z2]-volume[y1,x2,z0])/2.)
    Y[26] = ((volume[y2,x1,z2]-volume[y2,x1,z0])/2.)
    Y[27] = ((volume[y2,x2,z2]-volume[y2,x2,z0])/2.)
    Y[28] = ((volume[y1,x1,z3]-volume[y1,x1,z1])/2.)
    Y[29] = ((volume[y1,x2,z3]-volume[y1,x2,z1])/2.)
    Y[30] = ((volume[y2,x1,z3]-volume[y2,x1,z1])/2.)
    Y[31] = ((volume[y2,x2,z3]-volume[y2,x2,z1])/2.)
    
    # values of d2f/dxdy at each corner.
    Y[32] = ((volume[y2,x2,z1]-volume[y2,x0,z1]-volume[y0,x2,z1]+volume[y0,x0,z1])/4.)
    Y[33] = ((volume[y2,x3,z1]-volume[y2,x1,z1]-volume[y0,x3,z1]+volume[y0,x1,z1])/4.)
    Y[34] = ((volume[y3,x2,z1]-volume[y3,x0,z1]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
    Y[35] = ((volume[y3,x3,z1]-volume[y3,x1,z1]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
    Y[36] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y0,x2,z2]+volume[y0,x0,z2])/4.)
    Y[37] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y0,x3,z2]+volume[y0,x1,z2])/4.)
    Y[38] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y1,x2,z2]+volume[y1,x0,z2])/4.)
    Y[39] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y1,x3,z2]+volume[y1,x1,z2])/4.)

    # values of d2f/dxdz at each corner.
    Y[40] = ((volume[y1,x2,z2]-volume[y1,x0,z2]-volume[y1,x2,z0]+volume[y1,x0,z0])/4.)
    Y[41] = ((volume[y1,x3,z2]-volume[y1,x1,z2]-volume[y1,x3,z0]+volume[y1,x1,z0])/4.)
    Y[42] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y2,x2,z0]+volume[y2,x0,z0])/4.)
    Y[43] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y2,x3,z0]+volume[y2,x1,z0])/4.)
    Y[44] = ((volume[y1,x2,z3]-volume[y1,x0,z3]-volume[y1,x2,z1]+volume[y1,x0,z1])/4.)
    Y[45] = ((volume[y1,x3,z3]-volume[y1,x1,z3]-volume[y1,x3,z1]+volume[y1,x1,z1])/4.)
    Y[46] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y2,x2,z1]+volume[y2,x0,z1])/4.)
    Y[47] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y2,x3,z1]+volume[y2,x1,z1])/4.)
    
    # values of d2f/dydz at each corner.
    Y[48] = ((volume[y2,x1,z2]-volume[y0,x1,z2]-volume[y2,x1,z0]+volume[y0,x1,z0])/4.)
    Y[49] = ((volume[y2,x2,z2]-volume[y0,x2,z2]-volume[y2,x2,z0]+volume[y0,x2,z0])/4.)
    Y[50] = ((volume[y3,x1,z2]-volume[y1,x1,z2]-volume[y3,x1,z0]+volume[y1,x1,z0])/4.)
    Y[51] = ((volume[y3,x2,z2]-volume[y1,x2,z2]-volume[y3,x2,z0]+volume[y1,x2,z0])/4.)
    Y[52] = ((volume[y2,x1,z3]-volume[y0,x1,z3]-volume[y2,x1,z1]+volume[y0,x1,z1])/4.)
    Y[53] = ((volume[y2,x2,z3]-volume[y0,x2,z3]-volume[y2,x2,z1]+volume[y0,x2,z1])/4.)
    Y[54] = ((volume[y3,x1,z3]-volume[y1,x1,z3]-volume[y3,x1,z1]+volume[y1,x1,z1])/4.)
    Y[55] = ((volume[y3,x2,z3]-volume[y1,x2,z3]-volume[y3,x2,z1]+volume[y1,x2,z1])/4.)
    
    # values of d3f/dxdydz at each corner.
    Y[56] = ((volume[y2,x2,z2]-volume[y2,x0,z2]-volume[y0,x2,z2]+volume[y0,x0,z2]
              -volume[y2,x2,z0]-volume[y2,x0,z0]-volume[y0,x2,z0]+volume[y0,x0,z0])/8.)
    Y[57] = ((volume[y2,x3,z2]-volume[y2,x1,z2]-volume[y0,x3,z2]+volume[y0,x1,z2]
              -volume[y2,x3,z0]-volume[y2,x1,z0]-volume[y0,x3,z0]+volume[y0,x1,z0])/8.)
    Y[58] = ((volume[y3,x2,z2]-volume[y3,x0,z2]-volume[y1,x2,z2]+volume[y1,x0,z2]
              -volume[y3,x2,z0]-volume[y3,x0,z0]-volume[y1,x2,z0]+volume[y1,x0,z0])/8.)
    Y[59] = ((volume[y3,x3,z2]-volume[y3,x1,z2]-volume[y1,x3,z2]+volume[y1,x1,z2]
              -volume[y3,x3,z0]-volume[y3,x1,z0]-volume[y1,x3,z0]+volume[y1,x1,z0])/8.)

    Y[60] = ((volume[y2,x2,z3]-volume[y2,x0,z3]-volume[y0,x2,z3]+volume[y0,x0,z3]
              -volume[y2,x2,z1]-volume[y2,x0,z1]-volume[y0,x2,z1]+volume[y0,x0,z1])/8.)
    Y[61] = ((volume[y2,x3,z3]-volume[y2,x1,z3]-volume[y0,x3,z3]+volume[y0,x1,z3]
              -volume[y2,x3,z1]-volume[y2,x1,z1]-volume[y0,x3,z1]+volume[y0,x1,z1])/8.)
    Y[62] = ((volume[y3,x2,z3]-volume[y3,x0,z3]-volume[y1,x2,z3]+volume[y1,x0,z3]
              -volume[y3,x2,z1]-volume[y3,x0,z1]-volume[y1,x2,z1]+volume[y1,x0,z1])/8.)
    Y[63] = ((volume[y3,x3,z3]-volume[y3,x1,z3]-volume[y1,x3,z3]+volume[y1,x1,z3]
              -volume[y3,x3,z1]-volume[y3,x1,z1]-volume[y1,x3,z1]+volume[y1,x1,z1])/8.)    
    #print Y
    # Compute A
    A = np.dot(X_inv,Y)
    # get vector Y from points that need to be interpolated
    target_Y = get_target_Y(x-x1, y-y1, z-z1)
    # compute result
    result = np.dot(target_Y, A)
    #print result[0]
    #return A, result[0]
    return result[0]
'''
