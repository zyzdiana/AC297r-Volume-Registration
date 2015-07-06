import numpy as np
import scipy.special
import ghalton
from utils import to_radian, hann, hann_unsafe
from cost_functions import cf_ssd
from mask import circle_mask

def bessel_rotate(image_org, theta, xx, yy):
    image = image_org.copy()
    Ib = np.zeros(image.shape)
    
    theta = to_radian(theta)
    s = (image.shape[0]-1)/2.
    rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    for i in np.arange(-s, s+1.0):
        for j in np.arange(-s, s+1.0):
            new_x = np.dot(rM, np.array([i,j]))

            if(np.sum(abs(np.round(new_x,5))>s)):
                Ib[i+s,j+s] = 0
            else:
                R = np.sqrt((xx-new_x[1])**2 + (yy-new_x[0])**2)
                mask_R = (R == 0)
                Bess = np.zeros(R.shape)
                Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])/(np.pi*R[~mask_R])
                Bess[mask_R] = 0.5
                Ib[i+s,j+s] = np.dot(image.ravel(),Bess.ravel())/np.sum(Bess)
    return Ib

def generate_hann_windowed_bessel_interpolation(window_radius):
    def hann_windowed_bessel_rotate(image_org, theta, xx, yy):
        image = image_org.copy()
        Ib = np.zeros(image.shape)

        theta = to_radian(theta)

        s = (image.shape[0]-1)/2.

        rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        for i in np.arange(-s, s+1.0):
            for j in np.arange(-s, s+1.0):
                new_x = np.dot(rM, np.array([i,j]))

                if(np.sum(abs(np.round(new_x,5))>s)):
                    Ib[i+s,j+s] = 0
                else:
                    R = np.sqrt((xx-new_x[1])**2 + (yy-new_x[0])**2)
                    mask_origin_R = (R == 0)
                    mask_inner_R = (R < window_radius) & ~mask_origin_R
                    Bess = np.zeros(R.shape)
                    Bess[mask_inner_R] = hann_unsafe(R[mask_inner_R], window_radius) * scipy.special.j1(np.pi*R[mask_inner_R])/(np.pi*R[mask_inner_R]) 
                    Bess[mask_origin_R] = 0.5
                    Ib[i+s,j+s] = np.dot(image.ravel(),Bess.ravel())/np.sum(Bess)
        return Ib
    return hann_windowed_bessel_rotate

def bessel_cost_func(vol1_org, vol2_org, interp, thetas, axis):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    vol1 = vol1_org.copy()
    vol2 = vol2_org.copy()
    cost_func = np.zeros([len(thetas),])
    
    s = (vol2.shape[0]-1)/2.
    x = np.linspace(-s, s, vol2.shape[1])
    y = np.linspace(-s, s, vol2.shape[0])
    xx, yy = np.meshgrid(x,y)
    
    for idx, t in enumerate(thetas):
        new_vol2 = np.zeros(vol2.shape)
        for i in xrange(len(vol2)):
            if(axis == 0):
                new_vol2[i,:,:] = interp(vol2[i,:,:], t, xx, yy)
            elif(axis == 1):
                new_vol2[:,i,:] = interp(vol2[:,i,:], t, xx, yy)
            else:
                new_vol2[:,:,i] = interp(vol2[:,:,i], t, xx, yy)
        cost_func[idx] = cf_ssd(new_vol2, vol1)
    return cost_func
    
def generate_halton_points(N, dims, image_width):
    # generate Halton sample points
    sequencer = ghalton.Halton(dims)
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
    return float(image_width) * pts - (image_width / 2.0) 

def circle_mask_points(pts, radius):
    mask = np.array([np.linalg.norm(x) < radius for x in pts])
    return pts[mask]

def bessel_rotate_arbitrary_points(image, theta, x1, y1):
    Ib = []
    theta = to_radian(theta)
    s = (image.shape[0]-1)/2.
    
    rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    x = []
    for i in np.arange(-s,s+1):
        for j in np.arange(-s,s+1):
            x.append(np.dot(rM, np.array([i,j])))
    x = np.array(x)
    for idx in xrange(len(x1)):
        R = np.sqrt((x[:,0]-x1[idx])**2 + (x[:,1]-y1[idx])**2)
        mask_R = (R == 0)
        Bess = np.zeros(R.shape)
        Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])/(np.pi*R[~mask_R])
        Bess[mask_R] = 0.5
        Ib.append(np.dot(image.ravel(), Bess) / np.sum(Bess))
    return np.array(Ib)


def generate_hann_windowed_bessel_rotate_arbitrary_points_interpolator(window_radius):

    def hann_windowed_bessel_rotate_arbitrary_points(image, theta, x1, y1):
        Ib = []
        theta = to_radian(theta)
        s = (image.shape[0]-1)/2.0

        rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        rotatedImagePoints = []
        for i in np.arange(-s,s+1.0, 1.0):
            for j in np.arange(-s,s+1.0, 1.0):
                rotatedImagePoints.append(np.dot(rM, np.array([i,j])))
        rotatedImagePoints = np.array(rotatedImagePoints)
        for idx in xrange(len(x1)):
            R = np.sqrt((rotatedImagePoints[:,0]-x1[idx])**2 + (rotatedImagePoints[:,1]-y1[idx])**2)
            mask_origin_R = (R == 0)
            mask_inner_R = (R < window_radius) & ~mask_origin_R
            Bess = np.zeros(R.shape)
            Bess[mask_inner_R] = hann_unsafe(R[mask_inner_R], window_radius) * scipy.special.j1(np.pi*R[mask_inner_R])/(np.pi*R[mask_inner_R]) 
            Bess[mask_origin_R] = 0.5
            Ib.append(np.dot(image.ravel(), Bess) / np.sum(Bess))
        return np.array(Ib)
    return hann_windowed_bessel_rotate_arbitrary_points


def interp_rotate_2D_arbitrary_points_cost_func(vol1, vol2, interp, pts, thetas, axis):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    # initialize vector to store cost
    cost_func = np.zeros([len(thetas),])
    x1 = pts[:,0] 
    y1 = pts[:,1] 
    new_vol1 = np.empty([len(vol1),pts.shape[0]])
    for i in xrange(len(vol1)):
        if(axis == 0):
            sub1 = vol1[i,:,:]
        elif(axis == 1):
            sub1 = vol1[:,i,:]
        else:
            sub1 = vol1[:,:,i]

        new_vol1[i] = interp(sub1, 0, x1, y1)
    for idx, t in enumerate(thetas):
        #print t, 
        new_vol2 = np.empty([len(vol2),pts.shape[0]])
        for i in xrange(len(vol2)):
            if(axis == 0):
                sub2 = vol2[i,:,:]
            elif(axis == 1):
                sub2 = vol2[:,i,:]
            else:
                sub2 = vol2[:,:,i]
            new_vol2[i] = interp(sub2, t, x1, y1)
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func

def bessel_halton_cost_func(vol1, vol2, N, thetas, axis):
    return interp_rotate_2D_arbitrary_points_cost_func(vol1, vol2, bessel_rotate_arbitrary_points, generate_halton_points(N, 2, len(vol1) - 1), thetas, axis)

def bessel_halton_in_circ_cost_func(vol1, vol2, radius, N, thetas, axis):
    pts = generate_halton_points(N * 2, 2, len(vol1) - 1)
    pts = circle_mask_points(pts, radius)
    return interp_rotate_2D_arbitrary_points_cost_func(vol1, vol2, bessel_rotate_arbitrary_points, pts[:N], thetas, axis)

def hann_windowed_bessel_halton_in_circ_cost_func(vol1, vol2, point_mask_radius, N, interp_window_radius, thetas, axis):
    pts = generate_halton_points(N * 2, 2, len(vol1) - 1)
    pts = circle_mask_points(pts, point_mask_radius)
    return interp_rotate_2D_arbitrary_points_cost_func(vol1, vol2, generate_hann_windowed_bessel_rotate_arbitrary_points_interpolator(interp_window_radius), pts[:N], thetas, axis)