import numpy as np
import scipy.special
from cost_functions import cf_ssd,cf_L1,cf_L2
from mask import sphere_mask
# Trilinear interplation
def trilinear_interp(volume, x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    # find the closest grid of the target points
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    z0 = np.floor(z).astype(int)
    z1 = z0 + 1
    
    # Clip
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


def rotate_coords_3d(x, y, z, gamma, beta, alpha, ox, oy, oz):
    """
    Rotate arrays of coordinates x, y and z about the point (ox, oy, oz)
    about x axis by gamma degrees
    about y axis by beta degrees
    about z axis by alpha degrees
    """

    R = rotation_matrix_zyx(gamma, beta, alpha)
    #tmp = np.vstack([a,np.zeros([1,3])])
    x, y, z = x - ox, y - oy, z - oz
    return ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox, 
            (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy, 
            (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz)

def volrotate(volume_org, gamma, beta, alpha):
    #about x axis by gamma degrees
    #about y axis by beta degrees
    #about z axis by alpha degrees
    volume = volume_org.copy()
    
    # find center of the volume
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5
    
    x = np.linspace(0, volume.shape[1]-1, volume.shape[1]).astype(int)
    y = np.linspace(0, volume.shape[0]-1, volume.shape[0]).astype(int)
    z = np.linspace(0, volume.shape[2]-1, volume.shape[2]).astype(int)
    xx, yy, zz = np.meshgrid(x, y, z)
    
    dest_x, dest_y, dest_z = rotate_coords_3d(xx, yy, zz, gamma, beta, alpha, ox, oy, oz)
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    return dest


def rot_cost_func_3d(vol1, vol2, thetas, mask=False):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    arg: string for plot titles
    '''
    cost_func = np.zeros([len(thetas),])
    for idx, t in enumerate(thetas):
        new_vol2 = volrotate(vol2, t[0],t[1],t[2])
        cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func