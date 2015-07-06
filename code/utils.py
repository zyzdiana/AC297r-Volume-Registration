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