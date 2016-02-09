
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.special
import numpy as np
import time
import os
import sys
import numpy.fft as fft
import numpy.ma as ma
import statsmodels.robust
sys.path.insert(1,'/Users/zyzdiana/GitHub/AC297r-Volume-Registration/code')

from utils import to_radian,res_to_rad,ax_to_w,rep_to_angle
from rotation_3d import tricubic_derivatives,tricubic_interp
from rotation_3d import rot_cost_func_3d
from visualize import plot_cost_func,plot_volume,plot_slices
from cost_functions import cf_ssd
from Gauss_Newton import Gauss_Newton, axis_derivatives,sphere_mask,fourier_filter,get_gradient_P, window,get_mask_weights
from Gauss_Newton import rotate_coords_transformation_m,trace_plot,plot_errors

def Tukeys_Biweight(R, c):
    r = np.zeros(R.shape)
    mask = abs(R)<=c
    x = R[mask]
    r[mask]=x*(1-x**2/c**2)**2
    return r

def Tukeys_Biweight_1(x, c):
    if(abs(x) <= c):
        return x*(1-x**2/c**2)**2
    else:
        return 0


def MAD(a, c = 1.4826):
    """
    Median Absolute Deviation along given axis of an array:
    """

    a = ma.masked_where(a!=a, a)
    if a.ndim == 1:
        d = ma.median(a)
        m = c*ma.median(ma.fabs(a - d))
    else:
        d = ma.median(a, axis=axis)
        if axis > 0:
            aswp = ma.swapaxes(a,0,axis)
        else:
            aswp = a
        m = ma.median(c*ma.fabs(aswp - d), axis=0)

    return m


def Gauss_Newton_IRLS(Vol1, Vol1_Grad_P, Vol2, Vol2_derivatives, w,
                      d = 0.4, c = 6., divide_factor = 1., alpha = 1., decrease_factor = 0.25, 
                      P_initial = np.array([0,0,0,0,0,0]), max_iter = 10, plot = True, mask = True, plot_mask = False):
    s0,s1,s2 = Vol1.shape
    if (s0 == 26): res = '10'
    if (s0 == 32): res = '8'
    if (s0 == 40): res = '6_4'
    rad = res_to_rad(res)

    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    ox = s1/2.-0.5
    oy = s0/2.-0.5
    oz = s2/2.-0.5

    if(mask):
        mask_weights = get_mask_weights(Vol1,rad,d)

    Ps = []
    P_old = P_initial.copy()
    P_new = P_old.copy()
    Ps.append(P_new)
    errors = []
    errors.append(np.sum(Vol1))

    E2s = []
    E2s.append(np.sum(Vol1))

    Jr = Vol1_Grad_P.T
    #Jr_T_w = Vol1_Grad_P.dot(w)
    volume_shape = Vol1.shape
    for counter in xrange(max_iter):
        print counter,
        P_old = P_new.copy()
        # Get the new coordinates by rotating the volume by the opposite amount of P_s
        dest_x, dest_y, dest_z = rotate_coords_transformation_m(xx, yy, zz, P_old, ox, oy, oz,divide_factor,volume_shape)
        # Initilization
        dest = np.empty(volume_shape)

        for i in xrange(volume_shape[0]):
            dest[i,:,:] = tricubic_interp(volume_shape,Vol2_derivatives,dest_x[i,:,:],dest_y[i,:,:],dest_z[i,:,:])
        if(mask):
            dest *= mask_weights

        flatR = np.ravel(Vol1-dest)
        error = np.sum(flatR**2)
    
        ## if error is getting larger, go back one step and decrease alpha
        if(error > errors[-1]):
            alpha = alpha * decrease_factor
            P_new = Ps[-1]
        else:
            errors.append(error)
            Ps.append(P_old)

            # compute weights
            mad_r = MAD(flatR)
            if mad_r != 0:
                #print 'MAD: ', mad_r
                R = flatR/mad_r
            else:
                R = flatR.copy()

            W = Tukeys_Biweight(R,c)
            w = np.diag(W)
            E2 = np.sum(W*(R**2))/np.sum(W)
            E2s.append(E2)

            # update parameters with new weights
            Jr_T_w = Vol1_Grad_P.dot(w)
            Jr_rP = -Jr_T_w.dot(R)

            if(plot_mask):
                residual = flatR.reshape([32,32,32])
                plot_slices(residual,1,"residual")
                weights = W.reshape([32,32,32])
                plot_slices(weights*residual,1,"weighted vol1")

            P_new = P_old - alpha*np.dot(np.linalg.inv(np.dot(Jr_T_w,Jr)),Jr_rP)
            if((abs(P_new - P_old) < 1e-5).all()):
                print 'Converged in %s iterations!' % counter
                break

            
            # if(abs(E2-E2s[-1]) < 1e-9):
            #     E2s.append(E2)
            #     print 'mask Converged in %s iterations!' % counter
            #     break
            # E2s.append(E2)
    if(plot):
        trace_plot(Ps, float('.'.join(res.split('_'))))
        plot_errors(errors)
    print 'min error:', errors[-1]
    return Ps, flatR, errors, E2s