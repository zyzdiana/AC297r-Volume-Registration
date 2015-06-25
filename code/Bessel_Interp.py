from sys import exit
import numpy as np
import scipy.special
import ghalton
from utils import to_radian,hann
from cost_functions import cf_ssd
from mask import circle_mask
import time

def bessel_rotate(image_org, theta, mask = False, smooth = False, mode = 1):
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
    if(mask):
        for i in xrange(len(vol1)):
            if(axis == 0):
                vol1[i,:,:] = circle_mask(vol1[i,:,:], smooth, mode)
            elif(axis == 1):
                vol1[:,i,:] = circle_mask(vol1[:,i,:], smooth, mode)
            else:
                vol1[:,:,i] = circle_mask(vol1[:,:,i], smooth, mode)

    for idx, t in enumerate(thetas):
        print t,
        new_vol2 = np.zeros(vol2.shape)
        for i in xrange(len(vol2)):
            if(axis == 0):
                new_vol2[i,:,:] = bessel_rotate(vol2[i,:,:], t, mask, smooth, mode)
            elif(axis == 1):
                new_vol2[:,i,:] = bessel_rotate(vol2[:,i,:], t, mask, smooth, mode)
            else:
                new_vol2[:,:,i] = bessel_rotate(vol2[:,:,i], t, mask, smooth, mode)
        cost_func[idx] = cf_ssd(new_vol2, vol1)
    return cost_func


def bessel_rotate_arbitrary_points(image, theta, x1, y1):
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
		# mdt Jun 25/15 I flipped x to be [:,0] and y to be [:,1] to match the
		# pixel ordering that is imposed by the ravel() function below
		R = np.sqrt((rotatedImagePoints[:,0]-x1[idx])**2 + (rotatedImagePoints[:,1]-y1[idx])**2)
		mask_R = (R == 0)
		Bess = np.zeros(R.shape)
		#Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])*hann(R[~mask_R],image.shape[0])/(np.pi*R[~mask_R])
		Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])/(np.pi*R[~mask_R])
		Bess[mask_R] = 0.5
		# mdt Jun 25/15 Moved this division to after the summation; no need to
		# lots of division ops when one will suffice. Similarly, replaced
		# product followed by summation with dot operator.
		#Bess = Bess/np.sum(Bess)
		#tmp = image.ravel()*Bess
		#Ib.append(np.sum(tmp) / np.sum(Bess))
		Ib.append(np.dot(image.ravel(), Bess) / np.sum(Bess))
	return np.array(Ib)

def generate_halton_points(N, dims, image_width):	
    # generate Halton sample points
    sequencer = ghalton.Halton(dims)
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
# mdt Jun 24/15 -- This typecasting seems to be necessary in order to get the correct points
#    return image_width * pts - image_width / 2 
    return float(image_width) * pts - (image_width / 2.0) 

def bessel_halton_cost_func(vol1, vol2, N, thetas, axis):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
    arg: string for plot titles
    '''
    # initialize vector to store cost
    cost_func = np.zeros([len(thetas),])
    pts = generate_halton_points(N,2,len(vol1)-1)
    x1 = pts[:,0] 
    y1 = pts[:,1] 
    new_vol1 = np.empty([len(vol1),N])
    for i in xrange(len(vol1)):
		if(axis == 0):
			sub1 = vol1[i,:,:]
		elif(axis == 1):
			sub1 = vol1[:,i,:]
		else:
			sub1 = vol1[:,:,i]
		
		new_vol1[i] = bessel_rotate_arbitrary_points(sub1, 0, x1, y1)
    for idx, t in enumerate(thetas):
        #print t, 
        new_vol2 = np.empty([len(vol2),N])
        for i in xrange(len(vol2)):
			if(axis == 0):
				sub2 = vol2[i,:,:]
			elif(axis == 1):
				sub2 = vol2[:,i,:]
			else:
				sub2 = vol2[:,:,i]
			new_vol2[i] = bessel_rotate_arbitrary_points(sub2, t, x1, y1)
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func


def bessel_halton_cost_func_circle(vol1, vol2, N, thetas, axis, smooth = True, mode = 1):
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
    #sequencer = ghalton.GeneralizedHalton(ghalton.EA_PERMS[:3])
    sequencer = ghalton.Halton(2)
    sequencer.reset()
    points = sequencer.get(N)
    pts = np.array(points)
    xx1 = (len(vol1)-1) * pts[:,0] - s
    yy1 = (len(vol1)-1) * pts[:,1] - s
    mask = np.sqrt(xx1**2+yy1**2) < s*0.7
    x1 = xx1[mask]
    y1 = yy1[mask]
    new_vol1 = np.zeros([len(vol1),len(x1)])
    #print len(x1),
    for i in xrange(len(vol1)):
        if(axis == 0):
            sub1 = circle_mask(vol1[i,:,:], smooth = True, mode = 1)
        elif(axis == 1):
            sub1 = circle_mask(vol1[:,i,:], smooth = True, mode = 1)
        else:
            sub1 = circle_mask(vol1[:,:,i], smooth = True, mode = 1)
        rot = bessel_rotate_halton(sub1, 0, x1, y1)
        new_vol1[i] = rot
    for idx, t in enumerate(thetas):
        print t, 
        new_vol2 = np.empty([len(vol2),len(x1)])
        for i in xrange(len(vol2)):
            if(axis==0):
                sub2 = circle_mask(vol2[i,:,:], smooth = True, mode = 1)
            elif(axis==1):
                sub2 = circle_mask(vol2[:,i,:], smooth = True, mode = 1)
            else:
                sub2 = circle_mask(vol2[:,:,i], smooth = True, mode = 1)
            rot = bessel_rotate_halton(sub2, t, x1, y1)
            new_vol2[i] = rot
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func


