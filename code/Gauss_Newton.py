
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.special
import numpy as np
import time
import os
import sys
sys.path.insert(1,'/Users/zyzdiana/GitHub/AC297r-Volume-Registration/code')

from utils import to_radian,res_to_rad,ax_to_w,rep_to_angle
from rotation_3d import tricubic_derivatives,tricubic_interp
from rotation_3d import rot_cost_func_3d
from visualize import plot_cost_func
from cost_functions import cf_ssd
from mask import sphere_mask

# Function to compute the derivatives for the target volume
def axis_derivatives(volume):
    shape = volume.shape
    tricubic_derivative_dict = {}
    for i in xrange(0,shape[0]):
        for j in xrange(0,shape[1]):
            for k in xrange(0,shape[2]):
                # Take care of boundary conditions
                x1 = i
                y1 = j
                z1 = k
                x0 = x1 - 1
                x2 = x1 + 1
                y0 = y1 - 1
                y2 = y1 + 1
                z0 = z1 - 1
                z2 = z1 + 1
                
#                     # Wrap Around
#                 x0 = (x0 + volume.shape[1]) % volume.shape[1]
#                 x2 = (x2 + volume.shape[1]) % volume.shape[1]
#                 y0 = (y0 + volume.shape[0]) % volume.shape[0]
#                 y2 = (y2 + volume.shape[0]) % volume.shape[0]
#                 z0 = (z0 + volume.shape[2]) % volume.shape[2]
#                 z2 = (z2 + volume.shape[2]) % volume.shape[2] 

#                 # Wrap around original grid points
#                 x1 = (x1 + volume.shape[1]) % volume.shape[1]
#                 y1 = (y1 + volume.shape[0]) % volume.shape[0]
#                 z1 = (z1 + volume.shape[2]) % volume.shape[2]
                
                x0 = np.clip(x0, 0, volume.shape[1]-1)
                x1 = np.clip(x1, 0, volume.shape[1]-1)
                x2 = np.clip(x2, 0, volume.shape[1]-1)
                y0 = np.clip(y0, 0, volume.shape[0]-1)
                y1 = np.clip(y1, 0, volume.shape[0]-1)
                y2 = np.clip(y2, 0, volume.shape[0]-1)
                z0 = np.clip(z0, 0, volume.shape[2]-1)
                z1 = np.clip(z1, 0, volume.shape[2]-1)
                z2 = np.clip(z2, 0, volume.shape[2]-1)

                # Compute vector Y from known points
                Y = np.zeros([3,])

                # values of df/dx at each corner.
                Y[1] = ((volume[y1,x2,z1]-volume[y1,x0,z1])/2.)
                # values of df/dy at each corner.
                Y[0] = ((volume[y2,x1,z1]-volume[y0,x1,z1])/2.)
                # values of df/dz at each corner.
                Y[2] = ((volume[y1,x1,z2]-volume[y1,x1,z0])/2.)

                tricubic_derivative_dict[(i,j,k)] = Y
    return tricubic_derivative_dict

def rotate_coords_transformation_m(x, y, z, params, ox,oy,oz):
    l = np.sqrt(params[3]**2+params[4]**2+params[5]**2)/16.
    if(l == 0):
        return (x,y,z)
    s,c = np.sin(l/2.),np.cos(l/2.)
    alpha = c
    beta = s*params[4]/l
    gamma = s*params[3]/l
    delta = s*params[5]/l

    x, y, z = x - ox, y - oy, z - oz
    rotx = (alpha**2+beta**2-gamma**2-delta**2)*x+2*(beta*gamma-alpha*delta)*y+2*(beta*delta+alpha*gamma)*z+ox+params[0]
    roty = (alpha**2-beta**2+gamma**2-delta**2)*y+2*(beta*gamma+alpha*delta)*x+2*(gamma*delta-alpha*beta)*z+oy+params[0]
    rotz = (alpha**2-beta**2-gamma**2+delta**2)*z+2*(gamma*delta+alpha*beta)*y+2*(beta*delta-alpha*gamma)*x+oz+params[0]

    return (rotx,roty,rotz)

def get_M(x1,x2,x3):
    M = np.array([[1,0,0,0,x3/16.,-x2/16.],[0,1,0,-x3/16.,0,x1/16.],[0,0,1,x2/16.,-x1/16.,0]])
    return M

def to_degree(radian):
    return radian*180/np.pi

def trace_plot(Ps,res):
    arr = np.array(Ps)
    plt.figure(figsize = [12,4])
    plt.subplot(1,2,1)
    for i in xrange(3):
        plt.plot(arr[:,i]*res,label = 't_%s'%i)
    plt.legend(loc = 'best')
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Translations (mm)', fontsize = 15)
    plt.subplot(1,2,2)
    for j in xrange(3,6):
        plt.plot(to_degree(arr[:,j]),label = 'R_%s'%j)
    plt.legend(loc = 'best')
    plt.suptitle('Trace Plot for the Transformation Parameters',fontsize=18)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Rotations (degrees)', fontsize = 15)
    plt.show()
    
def plot_errors(errors):
    plt.plot(errors[1:])
    plt.title('Trace Plot for Errors',fontsize = 18)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('SSD Error', fontsize = 15)
    plt.show()
    
def convert_params(params,res):
    print 'translation (in mm):', params[:3]*res
    print 'rotations (in degrees):', params[3:]*180/np.pi
    
def print_results(errors, Ps, res):
    print 'min error: ', errors[-1]
    params = Ps[-1]
    print 'parameters at min error: ', params
    print 'translation (in mm):', params[:3]*res
    print 'rotations (in degrees):', params[3:]*180/np.pi

def Gauss_Newton(Vol1, Vol1_derivatives, Vol2, Vol2_derivatives, alpha = 0.2, decrease_factor = 0.25, P_initial = np.array([0,0,0,0,0,0]), plot = True):

    volume_shape = Vol1.shape
    if (volume_shape[0] == 26): res = '10'
    if (volume_shape[0] == 32): res = '8'
    if (volume_shape[0] == 40): res = '6_4'
    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    ox = volume_shape[1]/2.-0.5
    oy = volume_shape[0]/2.-0.5
    oz = volume_shape[2]/2.-0.5
    Ps = []
    P_old = P_initial.copy()
    P_new = P_old.copy()
    Ps.append(P_new)
    errors = []
    errors.append(1.0)
    
    volume_shape = Vol1.shape
    for counter in xrange(20):
        print counter,
        #print P_s
        P_old = P_new.copy()
        
        # Get the new coordinates by rotating the volume by the opposite amount of P_s
        dest_x, dest_y, dest_z = rotate_coords_transformation_m(xx, yy, zz, -1.*P_old, ox, oy, oz)
        # Initilization
        Jr = np.empty([volume_shape[0]*volume_shape[1]*volume_shape[2],6])
        Jr_rP = np.zeros([6,])
        dest = np.empty(volume_shape)
        idx = 0
        for i in xrange(volume_shape[0]):
            for j in xrange(volume_shape[1]):
                for k in xrange(volume_shape[2]):
                    dest[i,j,k] = tricubic_interp(volume_shape,Vol2_derivatives,dest_x[i,j,k],dest_y[i,j,k],dest_z[i,j,k]) 
                    M = get_M(yy[i,j,k]-ox,xx[i,j,k]-oy,zz[i,j,k]-oz)
                    for ii in xrange(len(P_old)):
                        Jr[idx,ii] = -1.*Vol1_derivatives[j,i,k].dot(M[:,ii])
                        Jr_rP[ii] += Jr[idx,ii]*(Vol1[yy[i,j,k],xx[i,j,k],zz[i,j,k]]-dest[i,j,k])
                    idx += 1
        error = cf_ssd(Vol1,dest)

        ## if error is getting larger, go back one step and decrease alpha
        if(error > errors[-1]):
            alpha = alpha * decrease_factor
            P_new = Ps[-1]
        else:
            #print abs(error-errors[-1])
            errors.append(error)
            Ps.append(P_old)
            P_new = P_old - alpha*np.dot(np.linalg.inv(np.dot(Jr.T,Jr)),Jr_rP)
            if((abs(P_new - P_old) < 1e-5).all()):
                print 'Converged in %s iterations!' % counter
                break
    if(plot):
        trace_plot(Ps, float('.'.join(res.split('_'))))
        plot_errors(errors)
    return errors, Ps