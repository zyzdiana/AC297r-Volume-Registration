import os
import numpy as np
'''
Helper function that cleans up list_directory
removes hidden files
'''
def clean(lis):
    try:
         
        lis.remove('.DS_Store')
    except:
        lis = lis
    return lis

'''
Helper function that retrieves all data files
cleans up the directory to leave only .dat files
'''
def get_files(lis):
    try:
        for i in list(lis):
            if('.dat' not in i):
                lis.remove(i)
    except:
        lis = lis
    return lis


'''
Helper function that converts angle from degrees to radian
'''
def to_radian(theta):
    return theta*np.pi/180.

'''
Helper function that calculates the Hann window function
'''
def hann(n, radius):
    if n > radius:
        return 0
    else:
        return 0.5 + 0.5 * np.cos((np.pi*n)/float(radius))

def hann_unsafe(n, radius):
    return 0.5 + 0.5 * np.cos((np.pi*n)/float(radius))

def res_to_rad(res):
    '''
    converts resolution to its corresponding voxel
    '''
    if(res == '5mm'):
        return 52/2
    if(res == '6_4mm'):
        return 40/2
    if(res == '8mm'):
        return 32/2
    if(res == '10mm'):
        return 26/2

# function to help convert rotation axis to the unit vector of rotation
def ax_to_w(rot_ax):
    '''
    rot_ax: roation axis
    wx,wy,wz: corresponding vector of rotation for the code implementation
    '''
    if(rot_ax == 'x'): wx,wy,wz = 0,1,0
    elif(rot_ax == 'y'): wx,wy,wz = 1,0,0
    elif(rot_ax == 'z'): wx,wy,wz = 0,0,1
    elif(rot_ax == 'xy'): wx,wy,wz = 1,1,0
    elif(rot_ax == 'xz'): wx,wy,wz = 0,1,1
    else: wx,wy,wz = 1,0,1
    return wx,wy,wz

def rep_to_angle(rep, rot_range):
    counter = rep % 6
    rotation_angle = 0.0
    translation_step = rep / 6 

    if rot_range == '0_5_to_2_5':
        if(counter != 0):
            rotation_angle = 0.5 * counter
    else: 
        # range = 3.0 to 5.0
        if(counter != 0):
            rotation_angle = 3.0 + (0.5 * (counter - 1))
    return rotation_angle, translation_step