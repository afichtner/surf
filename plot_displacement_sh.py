
import numpy as np
import matplotlib.pyplot as plt

def plot_displacement_sh(filename,show=True):

	"""
		Plot displacement function for Love waves. 
		plot_displacement_sh(filename,show=True)
		filename: filename including path
		show: set to True for showing the plot, and to False for not showing it. show=False can be useful when plotting multiple displacement functions.
	"""

	#- open file and read header information ------------------------------------------------------

	f=open(filename, 'r')

	f.readline()
	n=int(f.readline())
	f.readline()

	l1=np.zeros(n)
	l2=np.zeros(n)
	r=np.zeros(n)

	#- march through the depth levels and read file -----------------------------------------------

	for k in range(n):
		dummy=f.readline().strip().split(' ')
		r[k]=dummy[0]
		l1[k]=dummy[1]
		l2[k]=dummy[2]

	#- normalise and clean up ---------------------------------------------------------------------

	l1=l1/np.max(l1)
	l2=l2/np.max(l1)
	r=r/1000.0

	f.close()

	#- plot results -------------------------------------------------------------------------------

	plt.plot(l1,r,'k')
	plt.xlabel('normalised displacement')
	plt.ylabel('radius [km]')
	
	if show==True: plt.show()

	plt.plot(l2,r,'k')
	plt.xlabel('displacement-normalised stress')
	plt.ylabel('radius [km]')
	
	if show==True: plt.show()