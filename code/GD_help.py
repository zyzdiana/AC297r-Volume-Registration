
import numpy as np
import theano
import theano.tensor as T
rng = np.random.RandomState(42)
theano.config.floatX = 'float32'
def transform_10mm(volume,R,t):
        
    # find center of the volume
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5
    
    tmpx = np.linspace(0, 25, 26).astype(int)
    tmpy = np.linspace(0, 25, 26).astype(int)
    tmpz = np.linspace(0, 25, 26).astype(int)
    xx, yy, zz = np.meshgrid(tmpx, tmpx, tmpx)
    x, y, z = xx - ox, yy - oy, zz - oz
    dest_x, dest_y, dest_z = ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox + t[0], 
                              (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy + t[1], 
                              (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz + t[2])
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    #dest = translation(dest,t)
    return dest

def transform_6_4mm(volume,R,t):
        
    # find center of the volume
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5
    
    tmpx = np.linspace(0, 39, 40).astype(int)
    tmpy = np.linspace(0, 39, 40).astype(int)
    tmpz = np.linspace(0, 39, 40).astype(int)
    xx, yy, zz = np.meshgrid(tmpx, tmpx, tmpx)
    x, y, z = xx - ox, yy - oy, zz - oz
    dest_x, dest_y, dest_z = ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox + t[0], 
                              (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy + t[1], 
                              (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz + t[2])
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    #dest = translation(dest,t)
    return dest

def transform_8mm(volume,R,t):
        
    # find center of the volume
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5
    
    tmpx = np.linspace(0, 31, 32).astype(int)
    tmpy = np.linspace(0, 31, 32).astype(int)
    tmpz = np.linspace(0, 31, 32).astype(int)
    xx, yy, zz = np.meshgrid(tmpx, tmpx, tmpx)
    x, y, z = xx - ox, yy - oy, zz - oz
    dest_x, dest_y, dest_z = ((R[0][0]*x + R[0][1]*y + R[0][2]*z) + ox + t[0], 
                              (R[1][0]*x + R[1][1]*y + R[1][2]*z) + oy + t[1], 
                              (R[2][0]*x + R[2][1]*y + R[2][2]*z) + oz + t[2])
    dest = trilinear_interp(volume, dest_x, dest_y, dest_z)
    #dest = translation(dest,t)
    return dest
    
def trilinear_interp(volume, x, y, z):
    
    # find the closest grid of the target points
    x0 = T.floor(x).astype('int32')
    x1 = x0 + 1
    y0 = T.floor(y).astype('int32')
    y1 = y0 + 1
    z0 = T.floor(z).astype('int32')
    z1 = z0 + 1

    # Clip
    x0 = (x0 + volume.shape[0]) % volume.shape[0]
    x1 = (x1 + volume.shape[0]) % volume.shape[0]
    y0 = (y0 + volume.shape[0]) % volume.shape[0]
    y1 = (y1 + volume.shape[0]) % volume.shape[0]
    z0 = (z0 + volume.shape[0]) % volume.shape[0]
    z1 = (z1 + volume.shape[0]) % volume.shape[0] 
    
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
