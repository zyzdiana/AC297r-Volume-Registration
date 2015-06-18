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