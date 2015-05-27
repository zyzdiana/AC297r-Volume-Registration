import numpy as np
import scipy.special
from cost_functions import cf_ssd,cf_L1,cf_L2
# Bilinear interplation
def bilinear_interp(image, x, y):
    
    x = np.asarray(x)
    y = np.asarray(y)

    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    
    # Take care of boundary conditions
    # check if the input grid x, y is already on the original grid (i.e. special rotations)
    if (np.allclose(x.astype(int),x1)):
        return image[y.astype(int),x.astype(int)]
    
    x2 = x1 + 1
    y1 = np.floor(y).astype(int)
    y2 = y1 + 1
    
    
    # we need to clip the range 
    x1 = np.clip(x1, 0, image.shape[1]-1)
    x2 = np.clip(x2, 0, image.shape[1]-1)
    y1 = np.clip(y1, 0, image.shape[0]-1)
    y2 = np.clip(y2, 0, image.shape[0]-1)

    # get the four know points
    Q11 = image[y1, x1]
    Q21 = image[y1, x2]
    Q12 = image[y2, x1]
    Q22 = image[y2, x2]

    # get weights, note that here we are dealing with 1 grid, so c+d = a+b = 1
    a = x2 - x
    b = x - x1
    c = y - y1
    d = y2 - y

    return a*d*Q11 + b*d*Q21 + a*c*Q12 + b*c*Q22

# Bicubic Interpolation setup
X_inv = np.zeros([16,16])
X_inv[0][0] = 1
X_inv[1][4] = 1
X_inv[2][0:6] = [-3,3,0,0,-2,-1]
X_inv[3][0:6] = [2,-2,0,0,1,1]
X_inv[4][8] = 1 
X_inv[5][12] = 1
X_inv[6][8:14] = [-3,3,0,0,-2,-1]
X_inv[7][8:14] = [2,-2,0,0,1,1]
X_inv[8][0:3] = [-3,0,3]
X_inv[8][8:11] = [-2,0,-1]
X_inv[9][4:7] = [-3,0,3]
X_inv[9][12:15] = [-2,0,-1]
X_inv[10] = [9,-9,-9,9, 6,3,-6,-3,6,-6,3,-3,4,2,2,1]
X_inv[11] = [-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1]
X_inv[12][0:3] = [2,0,-2]
X_inv[12][8:11:2] = 1
X_inv[13][4:7] = [2,0,-2]
X_inv[13][12:15:2] = 1
X_inv[14] = [-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1]
X_inv[15] = [4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1]


def get_target_Y(x,y):
    Y = np.zeros([len(x),16])
    Y[:,0] = 1.
    Y[:,1] = x
    Y[:,2] = x**2
    Y[:,3] = x**3
    Y[:,4] = y
    Y[:,5] = x*y
    Y[:,6] = x**2*y
    Y[:,7] = x**3*y
    Y[:,8] = y**2
    Y[:,9] = x*y**2
    Y[:,10] = x**2*y**2
    Y[:,11] = x**3*y**2
    Y[:,12] = y**3
    Y[:,13] = x*y**3
    Y[:,14] = x**2*y**3
    Y[:,15] = x**3*y**3
    return Y

def bicubic_interp(image, x, y):
    xr = x.ravel()
    yr = y.ravel()
    
    # find the closes grid of the target points
    x1 = np.floor(x).astype(int)
    
    # Take care of boundary conditions
    # check if the input grid x, y is already on the original grid (i.e. special rotations)
    if (len(x)==len(image)):
        if(np.allclose(x.astype(int),x1)):
            return image[y.astype(int),x.astype(int)]
    x0 = x1 - 1
    x2 = x1 + 1
    x3 = x2 + 1
    y1 = np.floor(y).astype(int)
    y0 = y1 - 1
    y2 = y1 + 1
    y3 = y2 + 1
    
    # we need to clip the range 
    x0 = np.clip(x0, 0, image.shape[1]-1)
    x1 = np.clip(x1, 0, image.shape[1]-1)
    x2 = np.clip(x2, 0, image.shape[1]-1)
    x3 = np.clip(x3, 0, image.shape[1]-1)
    y0 = np.clip(y0, 0, image.shape[0]-1)
    y1 = np.clip(y1, 0, image.shape[0]-1)
    y2 = np.clip(y2, 0, image.shape[0]-1)
    y3 = np.clip(y3, 0, image.shape[0]-1)
    
    # compute the vector of coefficients A
    # first compute vector Y from known points to solve for A
    Y = np.zeros([16,len(xr)])
    Y[0] = image[y1,x1].ravel()
    Y[1] = image[y1,x2].ravel()
    Y[2] = image[y2,x1].ravel()
    Y[3] = image[y2,x2].ravel()
    
    Y[4] = ((image[y1,x2]-image[y1,x0])/2.).ravel()
    Y[5] = ((image[y1,x3]-image[y1,x1])/2.).ravel()
    Y[6] = ((image[y2,x2]-image[y2,x0])/2.).ravel()
    Y[7] = ((image[y2,x3]-image[y2,x1])/2.).ravel()
    
    Y[8] = ((image[y2,x1]-image[y0,x1])/2.).ravel()
    Y[9] = ((image[y2,x2]-image[y0,x2])/2.).ravel()
    Y[10] = ((image[y3,x1]-image[y1,x1])/2.).ravel()
    Y[11] = ((image[y3,x2]-image[y1,x2])/2.).ravel()
    
    Y[12] = ((image[y2,x2]-image[y2,x0]-image[y1,x2]+image[y1,x0])/4.).ravel()
    Y[13] = ((image[y0,x3]-image[y0,x1]-image[y2,x3]+image[y2,x1])/4.).ravel()
    Y[14] = ((image[y3,x2]-image[y3,x0]-image[y1,x2]+image[y1,x0])/4.).ravel()
    Y[15] = ((image[y3,x3]-image[y3,x1]-image[y1,x3]+image[y1,x1])/4.).ravel()
    # Compute A
    A = np.dot(X_inv,Y)
    
    # get vector Y from points that need to be interpolated
    target_Y = get_target_Y(xr-np.floor(xr), yr-np.floor(yr))
    # compute result
    result = np.dot(target_Y,A)
    # only need the diagonal values and reshape them back to original shape
    return np.reshape(result.diagonal(),x.shape)


# Rotation Setup
def to_radian(theta):
    return theta*np.pi/180.

def rotate_coords(x, y, theta, ox, oy):
    """
    Rotate arrays of coordinates x and y by theta radians about the
    point (ox, oy).

    """
    s, c = np.sin(theta), np.cos(theta)
    x, y = np.asarray(x) - ox, np.asarray(y) - oy
    return x * c - y * s + ox, x * s + y * c + oy

def circle_mask(image):
    tmp = image.copy()
    ox = image.shape[1]/2.-0.5
    oy = image.shape[0]/2.-0.5
    r = image.shape[0]/2.-0.5
    y, x = np.ogrid[-ox:image.shape[0]-ox, -oy:image.shape[0]-oy]
    mask = x*x + y*y <= r*r
    tmp[~mask] = 0
    return tmp

def bessel_rotate(image, theta):
    Ib = np.zeros(image.shape)

    s = (image.shape[0]-1)/2.

    x = np.linspace(-s, s, image.shape[1])
    y = np.linspace(-s, s, image.shape[0])
    
    xx, yy = np.meshgrid(x,y)
    
    rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    for i in np.arange(-s,s+1):
        for j in np.arange(-s,s+1):
            x = np.dot(rM, np.array([i,j]))

            if(np.sum(abs(x)>s)):
                Ib[i+s,j+s]=0
                
            else:
                R = np.sqrt((xx-x[1])**2 + (yy-x[0])**2)
                mask = (R == 0)
                Bess = np.zeros(R.shape)
                Bess[~mask] = scipy.special.j1(np.pi*R[~mask])/(np.pi*R[~mask])
                Bess[mask] = 0.5
                Ib[i+s,j+s] = (np.sum(image*Bess))*np.pi/2
    return Ib


def imrotate(image, theta, interpolation = 'bilinear', mask=False, x=None, y=None):
    theta = to_radian(theta)
    ox = image.shape[1]/2.-0.5
    oy = image.shape[0]/2.-0.5
    
    if((x == None) and (y == None)): #i.e. x and y not specified
        x = np.linspace(0, image.shape[1]-1, image.shape[1]).astype(int)
        y = np.linspace(0, image.shape[0]-1, image.shape[0]).astype(int)
    
    xx, yy = np.meshgrid(x,y)
    
    dest_x, dest_y = rotate_coords(xx, yy, theta, ox, oy)
    
    if(interpolation == 'bicubic'):
        dest = bicubic_interp(image, dest_x, dest_y)
    if(interpolation == 'bilinear'):
        dest = bilinear_interp(image, dest_x, dest_y)
    if(interpolation == 'bessel'):
        dest = bessel_rotate(image, theta)            
    if(mask):
        dest = circle_mask(dest)
    return dest


def rot_cost_func(vol1, vol2, thetas, axis, interpolation='bilinear',mask=False):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    arg: string for plot titles
    '''
    cost_func = np.zeros([len(thetas),])
    for idx, t in enumerate(thetas):
        new_vol2 = np.ones(vol2.shape)
        for i in xrange(len(vol2)):
            if(axis == 0):
                sub = vol2[i,:,:]
            elif(axis == 1):
                sub = vol2[:,i,:]
            else:
                sub = vol2[:,:,i]
            
            rot = imrotate(sub, t, interpolation, mask)

            if(axis == 0):
                new_vol2[i,:,:] = rot
            elif(axis == 1):
                new_vol2[:,i,:] = rot
            else:
                new_vol2[:,:,i] = rot
        cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func