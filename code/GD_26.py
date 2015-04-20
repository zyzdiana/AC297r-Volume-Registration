import numpy as np
import theano
import theano.tensor as T
from Trilinear import transform_26
rng = np.random.RandomState(42)
theano.config.floatX = 'float32'

#Define Theano Variables
T_vol1 = T.tensor3('vol1')
T_vol2 = T.tensor3('vol2')
T_trans = T.vector('trans')
T_rot = T.matrix('rot')

# define intermediate functions for GD
transformed_vol2 = transform_26(T_vol2,T_rot, T_trans)
err = T.sum(T.sqr(T_vol1-transformed_vol2))
gd_R, gd_t = T.grad(err,[T_rot,T_trans])

# define theano function for computing gradient
gradient = theano.function(inputs = [T_vol1, T_vol2, T_rot, T_trans], outputs = [err,gd_R,gd_t])
# compute cost function to verify the gradient descent restult
cost_func = theano.function(inputs = [T_vol1, T_vol2, T_rot, T_trans], outputs = [err])

def gradient_descent(vol1, vol2, R, t, learning_rate):
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
        if((abs(np.max(R_old-R)) < 1e-5) & (abs(np.max(t_old-t)) < 1e-5)):
            break
    print 'Minimized error: ', error
    print 'Rotation matrix R:\n', R
    print 'Translation vector t:\n', t
    return errors, R, t
