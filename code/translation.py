import numpy as np
import time
import ghalton
from cost_functions import cf_ssd
def translation(volume, x, y, z):
    
    # find the closest grid of the target points
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1

    # Wrap Around
    x0 = (x0 + volume.shape[1]) % volume.shape[1]
    x1 = (x1 + volume.shape[1]) % volume.shape[1]
    y0 = (y0 + volume.shape[0]) % volume.shape[0]
    y1 = (y1 + volume.shape[0]) % volume.shape[0]
    z0 = (z0 + volume.shape[2]) % volume.shape[2]
    z1 = (z1 + volume.shape[2]) % volume.shape[2] 
    
    # Wrap around original grid points
    x = (x + volume.shape[1]) % volume.shape[1]
    y = (y + volume.shape[0]) % volume.shape[0]
    z = (z + volume.shape[2]) % volume.shape[2]
    
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

def vol_cost_func_t(vol1, vol2, t):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    t: list of translation to try
    cf: cost function
    arg: string for plot titles
    '''
    x = np.linspace(0, vol1.shape[1]-1, vol1.shape[1]).astype(int)
    y = np.linspace(0, vol1.shape[0]-1, vol1.shape[0]).astype(int)
    z = np.linspace(0, vol1.shape[2]-1, vol1.shape[2]).astype(int)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    cost_func = np.zeros(len(t))
    for idx, th in enumerate(t):
        new_vol2 = translation(vol2, xx-th[1], yy-th[0], zz-th[2])
        #new_vol2 = translation1(vol2, th)
        cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func

def halton_cost_func_t(N, vol1, vol2, t, arg, mask=False):
    '''
    N: Number of Halton Sampling Points
    vol1: original image
    vol2: volume to be rotated
    theta: rotation by degrees
    t: list of translations to try
    arg: string for plot titles
    '''
    cost_func = np.zeros(len(t))
    t0 = time.time()
    # generate Halton Sequence in the 32*32*32 grid
    sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:3])
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
    x1 = 31 * pts[:,0]
    y1 = 31 * pts[:,1]
    z1 = 31 * pts[:,2]
    #x1, y1, z1 = np.meshgrid(x1, y1, z1)
    for idx, th in enumerate(t):
        v2 = translation(vol2, x1+th[0], y1+th[1], z1+th[2])
        v1 = translation(vol1, x1, y1, z1)
        #new_vol2 = translation(v2, th)
        cost_func[idx] = cf_ssd(v2,v1)
    t1 = time.time()
    print 'Time for %s halton samples is %s seconds' % (N,t1-t0) 
    return cost_func