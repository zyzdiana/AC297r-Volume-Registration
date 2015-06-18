import numpy as np
import scipy.special
import ghalton
from utils import to_radian,hann
from mask import circle_mask
import time

def bessel_rotate(image_org, theta, mask = True, smooth = False, mode = 1):
    image = image_org.copy()
    if(mask):
        image = circle_mask(image, smooth, mode)
    Ib = np.zeros(image.shape)
    theta = to_radian(theta)
    s = (image.shape[0]-1)/2.
    x = np.linspace(-s, s, image.shape[1])
    y = np.linspace(-s, s, image.shape[0])
    
    xx, yy = np.meshgrid(x,y)
    
    rM = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    for i in np.arange(-s,s+1):
        for j in np.arange(-s,s+1):
            new_x = np.dot(rM, np.array([i,j]))

            if(np.sum(abs(np.round(new_x,5))>s)):
                Ib[i+s,j+s] = 0
            else:
                R = np.sqrt((xx-new_x[1])**2 + (yy-new_x[0])**2)
                mask_R = (R == 0)
                Bess = np.zeros(R.shape)
                Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])*hann(R[~mask_R],image.shape[0]*mode)/(np.pi*R[~mask_R])
                Bess[mask_R] = 0.5
                Bess = Bess/np.sum(Bess)
                tmp = image*Bess
                Ib[i+s,j+s] = np.sum(tmp) #np.round(np.sum(tmp),10)
    return Ib

def bessel_cost_func(vol1_org, vol2_org, thetas, axis, mask=False, smooth = False, mode = 1):
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
    for idx, t in enumerate(thetas):
        print t,
        new_vol2 = np.ones(vol2.shape)
        for i in xrange(len(vol2)):
            if(axis == 0):
                new_vol2[i,:,:] = bessel_rotate(vol2[i,:,:], t, mask)
                if(mask):
                    vol1[i,:,:] = circle_mask(vol1[i,:,:], smooth, mode)
            elif(axis == 1):
                new_vol2[:,i,:] = bessel_rotate(vol2[:,i,:], t, mask)
                if(mask):
                    vol1[:,i,:] = circle_mask(vol1[:,i,:], smooth, mode)
            else:
                new_vol2[:,:,i] = bessel_rotate(vol2[:,:,i], t, mask)
                if(mask):
                    vol1[:,:,i] = circle_mask(vol1[:,:,i], smooth, mode)
        cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func



def bessel_rotate_halton(image, theta, x1, y1):
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
        R = np.sqrt((x[:,1]-x1[idx])**2 + (x[:,0]-y1[idx])**2)
        mask_R = (R == 0)
        Bess = np.zeros(R.shape)
        Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])*hann(R[~mask_R],image.shape[0])/(np.pi*R[~mask_R])
        Bess[mask_R] = 0.5
        Bess = Bess/np.sum(Bess)
        tmp = image.ravel()*Bess
        Ib.append(np.round(np.sum(tmp),10))
    return np.array(Ib)

def bessel_halton_cost_func(vol1_org, vol2, N, thetas, axis):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    vol1 = vol1_org.copy()
    # initialize vector to store cost
    cost_func = np.zeros([len(thetas),])
    # generate Halton sample points
    s = (len(vol1)-1)/2.
    sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:3])
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
    x1 = (len(vol1)-1) * pts[:,0] - s
    y1 = (len(vol1)-1) * pts[:,1] - s
    new_vol1 = np.empty([len(vol1),N])
    for i in xrange(len(vol1)):
        if(axis == 0):
            sub1 = circle_mask(vol1[i,:,:])
        elif(axis == 1):
            sub1 = circle_mask(vol1[:,i,:])
        else:
            sub1 = circle_mask(vol1[:,:,i])
        rot = bessel_rotate_halton(sub1, 0, x1, y1)
        new_vol1[i] = rot
    for idx, t in enumerate(thetas):
        print t, 
        new_vol2 = np.empty([len(vol2),N])
        for i in xrange(len(vol2)):
            if(axis==0):
                sub2 = circle_mask(vol2[i,:,:])
            elif(axis==1):
                sub2 = circle_mask(vol2[:,i,:])
            else:
                sub2 = circle_mask(vol2[:,:,i])
            new_vol2[i] = bessel_rotate_halton(sub2, t, x1, y1)
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func

def bessel_halton_cost_func_circle(vol1, vol2, N, thetas, axis):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    cost_func = np.zeros([len(thetas),])
    # generate Halton sample points
    s = (len(vol1)-1)/2.
    sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:3])
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
    xx1 = (len(vol1)-1) * pts[:,0] - s
    yy1 = (len(vol1)-1) * pts[:,1] - s
    mask = np.sqrt(xx1**2+yy1**2) < s - 1
    x1 = xx1[mask]
    y1 = yy1[mask]
    new_vol1 = np.zeros([len(vol1),len(x1)])
    print len(x1),
    for i in xrange(len(vol1)):
        if(axis == 0):
            sub1 = circle_mask(vol1[i,:,:])
        elif(axis == 1):
            sub1 = circle_mask(vol1[:,i,:])
        else:
            sub1 = circle_mask(vol1[:,:,i])
        rot = bessel_rotate_halton(sub1, 0, x1, y1)
        new_vol1[i] = rot
    for idx, t in enumerate(thetas):
        print t, 
        new_vol2 = np.empty([len(vol2),len(x1)])
        for i in xrange(len(vol2)):
            if(axis==0):
                sub2 = circle_mask(vol2[i,:,:])
            elif(axis==1):
                sub2 = circle_mask(vol2[:,i,:])
            else:
                sub2 = circle_mask(vol2[:,:,i])
            rot = bessel_rotate_halton(sub2, t, x1, y1)
            new_vol2[i] = rot
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func