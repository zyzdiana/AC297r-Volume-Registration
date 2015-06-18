import numpy as np
import scipy.special
# Bessel Rotation
def to_radian(theta):
    return theta*np.pi/180.

def circle_mask(image):
    ox = image.shape[1]/2.-0.5
    oy = image.shape[0]/2.-0.5
    r = image.shape[0]/2.-0.5
    y, x = np.ogrid[-ox:image.shape[0]-ox, -oy:image.shape[0]-oy]
    mask = x*x + y*y <= r*r
    image[~mask] = 0
    return image
    
def bessel_rotate(image, theta, mask = False):
    t0 = time.time()
    Ib = np.zeros(image.shape)
    theta = to_radian(theta)
    #image = np.fft.fft2(image)
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
                mask_R = (R == 0)
                Bess = np.zeros(R.shape)
                Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])/(np.pi*R[~mask_R])
                Bess[mask_R] = 0.5
                tmp = image*Bess
                Ib[i+s,j+s] = np.sum(tmp)*np.pi/2
    if(mask):
        Ib = circle_mask(Ib)
    t1 = time.time()
    return Ib

def bessel_rotate_halton(image, theta, x1, y1, mask = False):
    t0 = time.time()
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
        Bess[~mask_R] = scipy.special.j1(np.pi*R[~mask_R])/(np.pi*R[~mask_R])
        Bess[mask_R] = 0.5
        #Bess = Bess/(2*np.pi*np.sum(Bess*R))
        tmp = image.ravel()*Bess
        Ib.append(np.sum(tmp)*np.pi/2)
    if(mask):
        Ib = circle_mask(Ib)
    t1 = time.time()
    return np.array(Ib)
    
def cf_ssd(J, I):
    return np.sum((J-I)**2)

def bessel_halton_cost_func(vol1, vol2, N, thetas, axis, mask=False):
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
    x1 = (len(vol1)-1) * pts[:,0] - s
    y1 = (len(vol1)-1) * pts[:,1] - s
    new_vol1 = np.zeros([len(vol1),N])
    for i in xrange(len(vol1)):
        if(axis == 0):
            sub1 = vol1[i,:,:]
        elif(axis == 1):
            sub1 = vol1[:,i,:]
        else:
            sub1 = vol1[:,:,i]
        rot = bessel_rotate_halton(sub1, 0, x1, y1)
        new_vol1[i] = rot
    for idx, t in enumerate(thetas):
        print t, 
        new_vol2 = np.empty([len(vol2),N])
        for i in xrange(len(vol2)):
            if(axis==0):
                sub2 = vol2[i,:,:]
            elif(axis==1):
                sub2 = vol2[:,i,:]
            else:
                sub2 = vol2[:,:,i]
            rot = bessel_rotate_halton(sub2, t, x1, y1, mask)
            new_vol2[i] = rot
        cost_func[idx] = cf_ssd(new_vol2,new_vol1)
    return cost_func

def bessel_cost_func(vol1, vol2, thetas, axis, mask=False):
    '''
    vol1: original image
    vol2: volume to be rotated
    thetas: list of degress to try
    cf: cost function
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
            
            rot = bessel_rotate(sub, t, mask)

            if(axis == 0):
                new_vol2[i,:,:] = rot
            elif(axis == 1):
                new_vol2[:,i,:] = rot
            else:
                new_vol2[:,:,i] = rot
        cost_func[idx] = cf_ssd(new_vol2,vol1)
    return cost_func