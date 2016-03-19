import matplotlib.pyplot as plt
import cPickle as pickle
import scipy.special
import numpy as np
import time
import os
import sys
sys.path.insert(1,'/Users/zyzdiana/GitHub/AC297r-Volume-Registration/code')

from utils import to_radian,res_to_rad,ax_to_w,rep_to_angle
from cost_functions import cf_ssd
from mask import sphere_mask

# get axis-angle representation from rotation matrix
def rotation_matrix_to_q(gamma, beta, alpha):
    
    R_ravel = rotation_matrix_zyx(gamma,beta,alpha).ravel()
    a = np.sqrt((1-R_ravel[0]+R_ravel[5]+R_ravel[7])/2)
    b = np.sqrt((1-R_ravel[4]+R_ravel[6]+R_ravel[2])/2)
    c = np.sqrt((1-R_ravel[8]+R_ravel[3]+R_ravel[1])/2)
    x = (c-a+b)/2.
    y = (a-b+c)/2.
    z = (a+b-c)/2.
    w = (R_ravel[3]-R_ravel[1])/(4.0*z)
    theta = np.arccos(w)*2
    norm = np.sin(theta/2.)
    ui = x/norm
    uj = y/norm
    uk = z/norm
    
    return theta*180/np.pi, ui, uj, uk

def rotate_coords_transformation_m(params):
    rotMatrix = np.identity(3)
    l = np.sqrt(params[3]**2+params[4]**2+params[5]**2)
    #print l,l*180/np.pi
    if(l == 0):
        return rotMatrix
    s,c = np.sin(l/2.),np.cos(l/2.)
    alpha = c
    beta = s*params[3]/l
    gamma = s*params[4]/l
    delta = s*params[5]/l

    #rotx = (alpha**2+beta**2-gamma**2-delta**2)*x+2*(beta*gamma-alpha*delta)*y+2*(beta*delta+alpha*gamma)*z+ox+params[0]
    #roty = (alpha**2-beta**2+gamma**2-delta**2)*y+2*(beta*gamma+alpha*delta)*x+2*(gamma*delta-alpha*beta)*z+oy+params[0]
    #rotz = (alpha**2-beta**2-gamma**2+delta**2)*z+2*(gamma*delta+alpha*beta)*y+2*(beta*delta-alpha*gamma)*x+oz+params[0]
    
    rotMatrix[0][0] = alpha**2+beta**2-gamma**2-delta**2
    rotMatrix[0][1] = 2*(beta*gamma-alpha*delta)
    rotMatrix[0][2] = 2*(beta*delta+alpha*gamma)

    rotMatrix[1][0] = 2*(beta*gamma+alpha*delta)
    rotMatrix[1][1] = alpha**2-beta**2+gamma**2-delta**2
    rotMatrix[1][2] = 2*(gamma*delta-alpha*beta)

    rotMatrix[2][0] = 2*(beta*delta-alpha*gamma)
    rotMatrix[2][1] = 2*(gamma*delta+alpha*beta)
    rotMatrix[2][2] = alpha**2-beta**2-gamma**2+delta**2
    return rotMatrix

# Get rotation matrix from axis-angle rotation
def rotation_matrix_fromq(theta, ui, uj, uk):
    '''
    theta: angle in degrees
    '''
    theta = to_radian(theta)
    c = np.cos(theta)
    s = np.sin(theta)
    rotMatrix = np.zeros([3,3])
    
    if (ui**2 + uj**2 + uk**2 != 1):
        norm = np.sqrt(ui**2 + uj**2 + uk**2)
        ui = ui/norm
        uj = uj/norm
        uk = uk/norm
        
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

def rotation_matrix_from_params(params):
    rotMatrix = np.identity(3)
    l = np.sqrt(params[1]**2+params[0]**2+params[2]**2)
    if(l == 0):
        return rotMatrix
    s,alpha = np.sin(l/2.),np.cos(l/2.)
    beta = s*params[0]/l
    gamma = s*params[1]/l
    delta = s*params[2]/l

    rotMatrix[0][0] = alpha**2+beta**2-gamma**2-delta**2
    rotMatrix[0][1] = 2*(beta*gamma-alpha*delta)
    rotMatrix[0][2] = 2*(beta*delta+alpha*gamma)

    rotMatrix[1][0] = 2*(beta*gamma+alpha*delta)
    rotMatrix[1][1] = alpha**2-beta**2+gamma**2-delta**2
    rotMatrix[1][2] = 2*(gamma*delta-alpha*beta)

    rotMatrix[2][0] = 2*(beta*delta-alpha*gamma)
    rotMatrix[2][1] = 2*(gamma*delta+alpha*beta)
    rotMatrix[2][2] = alpha**2-beta**2-gamma**2+delta**2
    return rotMatrix


# Get the three angles of rotation give axis-angle representation
def angles_from_q(theta, ui, uj, uk):
    Q = rotation_matrix_fromq(theta, ui, uj, uk)
    alpha = np.arctan(Q[1,0]/Q[0,0])
    beta = np.arcsin(-Q[2,0])
    gamma = np.arctan(Q[2,1]/Q[2,2])
    return gamma*180/np.pi, beta*180/np.pi,  alpha*180/np.pi


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