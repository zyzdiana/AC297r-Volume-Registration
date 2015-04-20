import matplotlib.pyplot as plt
import os
import math
import time
import numpy as np
import scipy

def Maximum_Likelihood(vol1, vol2, thetas, t):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    prob = np.zeros(len(thetas))
    for idx, th in enumerate(thetas):
        R = rotation_matrix_zyx(0,th,0)
        new_vol2 = transform(vol2, R, t)
        prob[idx] = np.sum(scipy.stats.norm.logpdf((new_vol2-vol1).ravel(),loc = 0,scale=0.5))
    CI_95 = np.percentile(prob,[2.5,97.5])
    CI_95 = np.array([np.mean(prob)-1.96*0.25/np.sqrt(len(prob)),
                     np.mean(prob)+1.96*0.25/np.sqrt(len(prob))])
    CI_theta = np.array([thetas[np.argmin(abs(prob-CI_95[1]))],
                        thetas[np.argmin(abs(prob-CI_95[0]))]])
    pMin = thetas[np.argmax(prob)]
    plt.plot(thetas, prob, label='Max prob occurs at = %s' % (pMin))
    plt.vlines(CI_theta[0],np.min(prob),np.max(prob),color='r', label = '95%% CI')
    plt.vlines(CI_theta[1],np.min(prob),np.max(prob),color='r')
    plt.title('Maximum Likelihood with trilinear interpolation\n', fontsize=14)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Log Likelihood')
    plt.legend(fontsize=12)
    return CI_theta

def transform(volume,R,t):
    # find center of the volume
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5

    tmpx = np.linspace(0, 31, 32)
    tmpy = np.linspace(0, 31, 32)
    tmpz = np.linspace(0, 31, 32)
    
    xx, yy, zz = np.meshgrid(tmpx, tmpx, tmpx)
    x, y, z = xx - ox, yy - oy, zz - oz
    dest_x, dest_y, dest_z = ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox + t[0], 
                              (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy + t[1], 
                              (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz + t[2])
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    return dest

    
# Trilinear Interpolation
def trilinear_interp(volume, x, y, z):
    # find the closest grid of the target points
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1
    '''
    # Clip
    x0 = (x0 + volume.shape[0]) % volume.shape[0]
    x1 = (x1 + volume.shape[0]) % volume.shape[0]
    y0 = (y0 + volume.shape[1]) % volume.shape[1]
    y1 = (y1 + volume.shape[1]) % volume.shape[1]
    z0 = (z0 + volume.shape[2]) % volume.shape[2]
    z1 = (z1 + volume.shape[2]) % volume.shape[2]
    '''
    x0 = x0.clip(0, volume.shape[0]-1)
    x1 = x1.clip(0, volume.shape[0]-1)
    y0 = y0.clip(0, volume.shape[1]-1)
    y1 = y1.clip(0, volume.shape[1]-1)
    z0 = z0.clip(0, volume.shape[2]-1)
    z1 = z1.clip(0, volume.shape[2]-1) 
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
