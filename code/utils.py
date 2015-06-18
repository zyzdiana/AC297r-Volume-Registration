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
def hann(n, N):
    n = abs(n - (N-1.)/2.)
    return 0.5*(1-np.cos(2*np.pi*n/(N-1)))