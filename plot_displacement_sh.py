"""
Plotting Love-wave displacement and stress functions.

:copyright:
    Andreas Fichtner (andreas.fichtner@erdw.ethz.ch), December 2020
:license:
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""


import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times"
plt.rcParams.update({'font.size': 30})
plt.rcParams['xtick.major.pad']='12'
plt.rcParams['ytick.major.pad']='12'

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

	fig, (ax1, ax2) = plt.subplots(1,2,sharey='row',figsize=(30,50))

	ax1.plot(l1,r,'k')
	ax2.plot(l2,r,'k')

	ax1.grid()
	ax2.grid()

	ax1.set_title('displacement $y_1$',pad=30)
	ax1.set(xlabel='$y_1$',ylabel='z [km]')

	ax2.set_title('stress $y_2$',pad=30)
	ax2.set(xlabel='$y_2$')
	
	if show==True: plt.show()