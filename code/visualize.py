import matplotlib.pyplot as plt
import numpy as np

def plot_volumes(files):
	# Visualize the selected volumes
	fig, ax = plt.subplots(5,3)
	fig.set_figwidth(9)
	fig.set_figheight(15)
	[ax[0][i].imshow(files[0].max(axis=i),interpolation='None') for i in xrange(3)]
	[ax[0][i].set_title('Position1 (Original)') for i in xrange(3)]
	[ax[1][i].imshow(files[1].max(axis=i),interpolation='None') for i in xrange(3)]
	[ax[1][i].set_title('Position2') for i in xrange(3)]
	[ax[2][i].imshow(files[2].max(axis=i),interpolation='None') for i in xrange(3)]
	[ax[2][i].set_title('Position3') for i in xrange(3)]
	[ax[3][i].imshow(files[3].max(axis=i),interpolation='None') for i in xrange(3)]
	[ax[3][i].set_title('Position4') for i in xrange(3)]
	[ax[4][i].imshow(files[4].max(axis=i),interpolation='None') for i in xrange(3)]
	[ax[4][i].set_title('Position5') for i in xrange(3)]
	plt.show()
	plt.close(fig)

def plot_cost_func_all(thetas,cost_func,res,interp):
	# plot the cost function and find the minimum angle
    fig, ax = plt.subplots(1,3)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    arg = ['SSD', 'L1', 'L2']
    angMin = thetas[np.argmin(cost_func,axis=0)]
    for i in xrange(3):
        ax[i].plot(thetas,cost_func[:,i],label='Min angle = %s\n Resolution = %smm' % (angMin[i], res))
        ax[i].set_title('%s Cost function with %s interpolation' % (arg[i],interp), fontsize=14)
        ax[i].set_xlabel('Angle (degrees)')
        ax[i].set_ylabel('Cost function')
        ax[i].legend(loc='best')
    plt.show()
    plt.close(fig)
    
def plot_cost_func(thetas,cost_func,res,interp,rot, coil):
    # plot the cost function and find the minimum angle
    angMin = thetas[np.argmin(cost_func,axis=0)]
    plt.plot(thetas,cost_func,label='Min angle = %s\n Resolution = %s \n rotation = %s' % (angMin, res, rot))
    plt.title('SSD with %s interpolation, %s coil' % (interp,coil), fontsize=14)
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Cost function')
    plt.legend(loc='best')