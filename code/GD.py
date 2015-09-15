import numpy as np
import theano
import theano.tensor as T
from GD_help import transform
rng = np.random.RandomState(42)
theano.config.floatX = 'float32'

#Define Theano Variables
T_vol1 = T.tensor3('vol1')
T_vol2 = T.tensor3('vol2')
T_trans = T.vector('trans')
T_rot = T.vector('rot')
size = 40
res = 6.4

# define intermediate functions for GD
transformed_vol2 = transform(T_vol2, T_rot, T_trans, size)
err = T.sum(T.sqr(T_vol1-transformed_vol2))
gd_R, gd_t = T.grad(err,[T_rot,T_trans])

# define theano function for computing gradient
gradient = theano.function(inputs = [T_vol1, T_vol2, T_rot, T_trans], outputs = [err,gd_R,gd_t], allow_input_downcast=True)
# compute cost function to verify the gradient descent restult
cost_func = theano.function(inputs = [T_vol1, T_vol2, T_rot, T_trans], outputs = [err], allow_input_downcast=True)

def gradient_descent(vol1, vol2, R, t, learning_rate, tol=1e-5):
    errors = []
    count = 0
    while True:
        count += 1
        R_old = R.copy()
        t_old = t.copy()
        error,rgd,rgt = gradient(vol1,vol2,R,t)
        errors.append(error)

        R -= learning_rate * rgd
        t -= learning_rate * rgt
        if((abs(np.max(R_old-R)) < tol) & (abs(np.max(t_old-t)) < tol)):
            break
    print 'Minimized error: ', error
    print 'Rotation matrix R:\n', R
    print 'Translation vector t:\n', t*res
    return errors, R, t
