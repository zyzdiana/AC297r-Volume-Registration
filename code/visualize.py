import matplotlib.pyplot as plt

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
