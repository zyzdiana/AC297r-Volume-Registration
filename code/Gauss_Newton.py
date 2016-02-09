
import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.special
import numpy as np
import time
import os
import sys
import numpy.fft as fft
sys.path.insert(1,'/Users/zyzdiana/GitHub/AC297r-Volume-Registration/code')

from utils import to_radian,res_to_rad,ax_to_w,rep_to_angle
from rotation_3d import tricubic_derivatives,tricubic_interp
from rotation_3d import rot_cost_func_3d
from visualize import plot_cost_func
from cost_functions import cf_ssd
#from mask import sphere_mask

def window(n, radius, d=0.4):
    tmp = float(n)/radius - 0.75
    if tmp < 0:
        return 1
    else:
        if((tmp/d > -0.5) and (tmp/d <0.5)):
            return np.cos((np.pi*(tmp/d)))
        else:
            return 0
def sphere_mask(volume, radius, d=0.4):
    origin = np.array([(x - 1.0) / 2.0 for x in volume.shape])
    mask_frequency = np.array([[[window(np.linalg.norm(np.array([x,y,z]) - origin), radius, d) 
                                 for x in range(volume.shape[0])] for y in range(volume.shape[1])] 
                               for z in range(volume.shape[2])])

    return mask_frequency * volume

def get_nonzero_mask(volume, radius, d=0.4):
    origin = np.array([(x - 1.0) / 2.0 for x in volume.shape])
    mask_frequency = np.array([[[window(np.linalg.norm(np.array([x,y,z]) - origin), radius, d) 
                                 for x in range(volume.shape[0])] for y in range(volume.shape[1])] 
                               for z in range(volume.shape[2])])

    return mask_frequency != 0
    
def get_mask_weights(volume, radius, d=0.4):
    origin = np.array([(x - 1.0) / 2.0 for x in volume.shape])
    mask_frequency = np.array([[[window(np.linalg.norm(np.array([x,y,z]) - origin), radius, d) 
                                 for x in range(volume.shape[0])] for y in range(volume.shape[1])] 
                               for z in range(volume.shape[2])])

    return mask_frequency

def fourier_filter(vol, rad):
    return abs(fft.ifftn(fft.ifftshift(sphere_mask(fft.fftshift(fft.fftn(vol)),rad,d=0.375))))

def axis_derivatives(volume):
    shape = volume.shape
    xx,yy,zz = np.meshgrid(np.linspace(0,shape[1]-1,shape[1]),np.linspace(0,shape[0]-1,shape[0]),np.linspace(0,shape[2]-1,shape[2]))
    x1 = xx.astype(int)
    y1 = yy.astype(int)
    z1 = zz.astype(int)

    x0 = x1 - 1
    x2 = x1 + 1
    y0 = y1 - 1
    y2 = y1 + 1
    z0 = z1 - 1
    z2 = z1 + 1

    # wrap around
    x0 = (x0 + shape[1]) % shape[1]
    x2 = (x2 + shape[1]) % shape[1]
    y0 = (y0 + shape[0]) % shape[0]
    y2 = (y2 + shape[0]) % shape[0]
    z0 = (z0 + shape[2]) % shape[2]
    z2 = (z2 + shape[2]) % shape[2] 

    # Wrap around original grid points
    x1 = (x1 + shape[1]) % shape[1]
    y1 = (y1 + shape[0]) % shape[0]
    z1 = (z1 + shape[2]) % shape[2]

    # Compute vector Y from known points
    Y = np.empty([shape[1],shape[0],shape[2],3])
    
    # values of df/dx at each corner.
    Y[:,:,:,0] = ((volume[y1,x1,z2]-volume[y1,x1,z0])/2.)
    # values of df/dy at each corner.
    Y[:,:,:,1] = ((volume[y1,x2,z1]-volume[y1,x0,z1])/2.)
    # values of df/dz at each corner.
    Y[:,:,:,2] = ((volume[y2,x1,z1]-volume[y0,x1,z1])/2.)

    return Y

def rotate_coords_transformation_m(xx, yy, zz, params, ox,oy,oz, k=16.,shape = (32,32,32)):
    rotMatrix = np.identity(3)
    l = np.sqrt(params[4]**2+params[3]**2+params[5]**2)/k
    if(l == 0):
        return ((xx-params[1]+shape[0])%shape[0], (yy-params[0]+shape[1])%shape[1], (zz-params[2]+shape[2])%shape[2])
    s,alpha = np.sin(l/2.),np.cos(l/2.)
    beta = s*params[3]/l
    gamma = s*params[4]/l
    delta = s*params[5]/l

    rotMatrix[0][0] = alpha**2+beta**2-gamma**2-delta**2
    rotMatrix[0][1] = 2*(beta*gamma-alpha*delta)
    rotMatrix[0][2] = 2*(beta*delta+alpha*gamma)

    rotMatrix[1][0] = 2*(beta*gamma+alpha*delta)
    rotMatrix[1][1] = alpha**2-beta**2+gamma**2-delta**2
    rotMatrix[1][2] = 2*(gamma*delta-alpha*beta)

    rotMatrix[2][0] = 2*(beta*delta-alpha*gamma)
    rotMatrix[2][1] = 2*(gamma*delta+alpha*beta)
    rotMatrix[2][2] = alpha**2-beta**2-gamma**2+delta**2
    
    R = np.linalg.inv(rotMatrix)

    y, x, z = xx - ox- params[1], yy - oy- params[0], zz - oz- params[2]

    dest_y = (R[0][0]*x + R[0][1]*y + R[0][2]*z + oy + shape[0]) % shape[0]
    dest_x = (R[1][0]*x + R[1][1]*y + R[1][2]*z + ox + shape[1]) % shape[1]
    dest_z = (R[2][0]*x + R[2][1]*y + R[2][2]*z + oz + shape[2]) % shape[2]
    
    return dest_x,dest_y,dest_z

def get_M(x1_org,x2_org,x3_org,k=16.):
    x1 = x1_org/k
    x2 = x2_org/k
    x3 = x3_org/k
    M = np.array([[0,0,-1,-x2,x1,0],[0,-1,0,x3,0,-x1],[-1,0,0,0,-x3,x2]])
    return M

def get_gradient_P(derivatives, divide_factor=1., mask=True):
    s0,s1,s2,s3 = derivatives.shape
    if (s0 == 26+15): res = '10'
    if (s0 == 32+15): res = '8'
    if (s0 == 40+15): res = '6_4'
    rad = res_to_rad(res)

    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    ox = (s1-15)/2.-0.5
    oy = (s0-15)/2.-0.5
    oz = (s2-15)/2.-0.5

    derivatives = axis_derivatives(volume)
    derivativesP = np.empty([6,s0*s1*s2])

    if(mask):
        mask_weights = get_mask_weights(volume,rad).reshape([s0,s1,s2,1])
        derivatives = mask_weights*derivatives
    idx = 0
    for i in xrange(s0):
        for j in xrange(s1):
            for k in xrange(s2):
                M = get_M(yy[i,j,k]-ox,xx[i,j,k]-oy,zz[i,j,k]-oz,divide_factor)
                derivativesP[:,idx] = (M.T).dot(derivatives[i,j,k])
                idx += 1
    return derivativesP

def to_degree(radian):
    return radian*180/np.pi

def trace_plot(Ps,res):
    arr = np.array(Ps)
    plt.figure(figsize = [12,4])
    plt.subplot(1,2,1)
    t_legends = ['x','y','z']
    for i in xrange(3):
        plt.plot(arr[:,i]*res,label = 't_%s'%t_legends[i])
    plt.legend(loc = 'best')
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Translations (mm)', fontsize = 15)
    plt.subplot(1,2,2)
    R_legends = ['x','y','z']
    for j in xrange(3,6):
        plt.plot(to_degree(arr[:,j]),label = 'R_%s'%R_legends[j-3])
    plt.legend(loc = 'best')
    plt.suptitle('Trace Plot for the Transformation Parameters',fontsize=18)
    plt.xlabel('Iterations', fontsize = 15)
    plt.ylabel('Rotations (degrees)', fontsize = 15)
    plt.show()
    
def plot_errors(errors):
    plt.figure(figsize = [6,4])
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

def Gauss_Newton(Vol1, Vol1_Grad_P, Vol2, Vol2_derivatives, 
                 divide_factor = 1., alpha = 1., decrease_factor = 0.25, 
                 P_initial = np.array([0,0,0,0,0,0]), plot = True, max_iter = 10, mask = True):
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
        mask_weights = get_mask_weights(Vol1,rad)

    Ps = []
    P_old = P_initial.copy()
    P_new = P_old.copy()
    Ps.append(P_new)

    Jr = Vol1_Grad_P.T

    errors = []
    errors.append(np.sum(Vol1))
    
    volume_shape = Vol1.shape
    for counter in xrange(max_iter):
        #print counter,
        P_old = P_new.copy()
        # Get the new coordinates by rotating the volume by the opposite amount of P_s
        dest_x, dest_y, dest_z = rotate_coords_transformation_m(xx, yy, zz, P_old, ox, oy, oz,divide_factor,volume_shape)
        # Initilization
        #Jr = Vol1_Grad_P.T
        dest = np.empty(volume_shape)

        for i in xrange(volume_shape[0]):
            dest[i,:,:] = tricubic_interp(volume_shape,Vol2_derivatives,dest_x[i,:,:],dest_y[i,:,:],dest_z[i,:,:])

        if(mask):
            dest *= mask_weights

        flatR = np.ravel(Vol1-dest)
        error = np.sum(flatR**2)

        # if error is getting larger, go back one step and decrease alpha
        if(error > errors[-1]):
            alpha = alpha * decrease_factor
            P_new = Ps[-1]
        else:
            errors.append(error)
            Ps.append(P_old)
            Jr_rP = -Vol1_Grad_P.dot(flatR)
            P_new = P_old - alpha*np.dot(np.linalg.inv(np.dot(Jr.T,Jr)),Jr_rP)
            if((abs(P_new - P_old) < 1e-5).all()):
                print 'Converged in %s iterations!' % counter
                break
    if(plot):
        trace_plot(Ps, float('.'.join(res.split('_'))))
        plot_errors(errors)
    return errors, Ps

def Gauss_Newton_1(Vol1, Vol2, Vol2_derivatives, 
                 divide_factor = 1., alpha = 1., decrease_factor = 0.25, 
                 P_initial = np.array([0,0,0,0,0,0]), plot = True, max_iter = 10, mask = True):
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
        mask_weights = get_mask_weights(Vol1,rad)

    Ps = []
    P_old = P_initial.copy()
    P_new = P_old.copy()
    Ps.append(P_new)
    Vol2_derivs = Vol2_derivatives[15:-15,15:-15,15:-15,(1,4,16)]
    Vol2_grap_P = get_gradient_P()
    Jr = Vol1_Grad_P.T

    errors = []
    errors.append(np.sum(Vol1))
    
    volume_shape = Vol1.shape
    for counter in xrange(max_iter):
        #print counter,
        P_old = P_new.copy()
        # Get the new coordinates by rotating the volume by the opposite amount of P_s
        dest_x, dest_y, dest_z = rotate_coords_transformation_m(xx, yy, zz, -P_old, ox, oy, oz,divide_factor,volume_shape)
        # Initilization
        #Jr = Vol1_Grad_P.T
        dest = np.empty(volume_shape)

        for i in xrange(volume_shape[0]):
            dest[i,:,:] = tricubic_interp(volume_shape,Vol2_derivatives,dest_x[i,:,:],dest_y[i,:,:],dest_z[i,:,:])

        if(mask):
            dest *= mask_weights

        flatR = np.ravel(Vol1-dest)
        error = np.sum(flatR**2)

        # if error is getting larger, go back one step and decrease alpha
        if(error > errors[-1]):
            alpha = alpha * decrease_factor
            P_new = Ps[-1]
        else:
            errors.append(error)
            Ps.append(P_old)
            Jr_rP = -Vol1_Grad_P.dot(flatR)
            P_new = P_old - alpha*np.dot(np.linalg.inv(np.dot(Jr.T,Jr)),Jr_rP)
            if((abs(P_new - P_old) < 1e-5).all()):
                print 'Converged in %s iterations!' % counter
                break
    if(plot):
        trace_plot(Ps, float('.'.join(res.split('_'))))
        plot_errors(errors)
    return errors, Ps

def LM(Vol1, Vol1_derivatives, Vol2, Vol2_derivatives, 
       divide_factor = 16., lamda = 0.2, decrease_factor = 0.25, 
       P_initial = np.array([0,0,0,0,0,0]), plot = True, max_iter = 20, mask = False):
    volume_shape = Vol1.shape
    if (volume_shape[0] == 26): res = '10'
    if (volume_shape[0] == 32): res = '8'
    if (volume_shape[0] == 40): res = '6_4'
    rad = volume_shape[0]/2
    xx,yy,zz = pickle.load(open('/Users/zyzdiana/Dropbox/THESIS/for_cluster/mesh_grid_%s.p'%res,'rb'))
    ox = volume_shape[1]/2.-0.5
    oy = volume_shape[0]/2.-0.5
    oz = volume_shape[2]/2.-0.5

    mask_weights = get_mask_weights(Vol1,16).reshape([volume_shape[1],volume_shape[0],volume_shape[2],1])

    Ps = []
    P_old = P_initial.copy()
    P_new = P_old.copy()
    Ps.append(P_new)
    errors = []
    errors.append(1.0)
    
    for counter in xrange(max_iter):
        print counter,
        #print P_s
        P_old = P_new.copy()
        
        # Get the new coordinates by rotating the volume by the opposite amount of P_s
        dest_x, dest_y, dest_z = rotate_coords_transformation_m(xx, yy, zz, -1.*P_old, ox, oy, oz,divide_factor)
        #print dest_x
        # Initilization
        Jr = np.empty([volume_shape[0]*volume_shape[1]*volume_shape[2],6])
        Jr_rP = np.zeros([6,])
        dest = np.empty(volume_shape)
        idx = 0
        #Mtensor = np.empty([volume_shape[0],volume_shape[0],volume_shape[0],3,6])
        for i in xrange(volume_shape[0]):
            for j in xrange(volume_shape[1]):
                for k in xrange(volume_shape[2]):
                    dest[i,j,k] = tricubic_interp(volume_shape,Vol2_derivatives,dest_x[i,j,k],dest_y[i,j,k],dest_z[i,j,k])
                    M = get_M(yy[i,j,k]-ox,xx[i,j,k]-oy,zz[i,j,k]-oz,divide_factor)
                    #Mtensor[i,j,k,:] = M
                    for ii in xrange(len(P_old)):
                        Jr[idx,ii] = -1.*Vol1_derivatives[i,j,k].dot(M[:,ii])
                    idx += 1
        
        if(mask):
            dest = mask_weights*dest
        #ImagePoints = np.array(np.memmap('/Users/zyzdiana/Downloads/GaussNewtonInterp_8mm-2/ImagePoints.dat',dtype=np.float32,mode='c',shape=(32,32,32,3)))
        #dest = (ImagePointWeights*dest.ravel())
        flatR = Vol1.ravel()-dest
        Jr_rP = Jr.T.dot(flatR)
        
        error = cf_ssd(Vol1.ravel(),dest)
        print error,

        ## if error is getting larger, go back one step and decrease alpha
        if(error > errors[-1]):
            lamda = lamda * decrease_factor
            P_new = Ps[-1]
        else:
            errors.append(error)
            Ps.append(P_old)
            JrT_Jr = np.dot(Jr.T,Jr)
            delta =  np.dot(np.linalg.inv(JrT_Jr + lamda*np.diag(np.diag(JrT_Jr))),Jr_rP)
            #print delta
            P_new = P_old - delta
            print P_new
            if((abs(delta) < 1e-5).all()):
                print 'Converged in %s iterations!' % counter
                break
    if(plot):
        trace_plot(Ps, float('.'.join(res.split('_'))))
        plot_errors(errors)
    return errors, Ps