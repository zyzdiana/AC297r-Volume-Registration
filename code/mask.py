import numpy as np
from utils import hann
'''
Masking Function, puts a circular mask on a square image
'''
def circle_mask(image_org, smooth = True, mode = 1):
    '''
    image_org: The input image
    smooth: puts a smooth mask on the input image if true, 
            simple circular mask if False
    mode: controls the radius of the mask.
    '''
    image = image_org.copy()
    ox = image.shape[1]/2.-0.5
    oy = image.shape[0]/2.-0.5
    r = image.shape[0]/2.-0.5
    y, x = np.ogrid[-ox:image.shape[0]-ox, -oy:image.shape[0]-oy]
    rad = np.sqrt(x*x + y*y)
    mask = x*x + y*y < r*r
    image[~mask] = 0
    if(smooth):
        image[mask] = image[mask]*hann(rad[mask], image.shape[0]*np.sqrt(mode))
    return image
    
def sphere_mask(volume, radius):
    ox = volume.shape[1]/2.-0.5
    oy = volume.shape[0]/2.-0.5
    oz = volume.shape[2]/2.-0.5
    
    r = len(volume)/2.-0.5
    
    x,y,z = np.ogrid[-ox:volume.shape[1]-ox, -oy:volume.shape[0]-oy,-oz:volume.shape[2]-oz]
    mask = (x*x + y*y + z*z <= r*r)
    vol = np.array(volume)
    vol[~mask] = 0
    origin = np.array([(x - 1.0) / 2.0 for x in volume.shape])

    mask_frequency = np.array([[[hann(np.linalg.norm(np.array([x,y,z]) - origin), radius) for x in range(volume.shape[0])] for y in range(volume.shape[1])] for z in range(volume.shape[2])])

    return mask_frequency * vol

def get_nonzero_mask(volume_shape, radius):
    ox = volume_shape[1]/2.-0.5
    oy = volume_shape[0]/2.-0.5
    oz = volume_shape[2]/2.-0.5
    
    r = volume_shape[0]/2.-0.5
    
    x,y,z = np.ogrid[-ox:volume_shape[1]-ox, -oy:volume_shape[0]-oy,-oz:volume_shape[2]-oz]
    mask = (x*x + y*y + z*z <= r*r)
    return mask