import numpy as np
import scipy.special
from utils import to_radian, hann, hann_unsafe
from cost_functions import cf_ssd

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