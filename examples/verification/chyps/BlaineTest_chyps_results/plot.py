import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_probe(filename, lab):
	#Load probe
	A = np.loadtxt(filename, skiprows=1)
	
	#Extract data
	time = A[:, 1]   # density
	T   = A[:, -2]  # temperature
	
	#Plot Temperature vs time
	plt.figure(1)
	plt.plot(time, T, label=lab)
	
	return A


#Load CHyPS probes
load_probe("probes/T0", 'probe 0')
load_probe("probes/T1", 'probe 1')
load_probe("probes/T2", 'probe 2')
load_probe("probes/T3", 'probe 3')
load_probe("probes/T4", 'probe 4')
load_probe("probes/T5", 'probe 5')
load_probe("probes/T6", 'probe 6')
load_probe("probes/T7", 'probe 7')
load_probe("probes/T8", 'probe 8')




#Label temperature vs time
plt.figure(1)
plt.xlabel('Time [s]')
plt.ylabel('Temperature [K]')
plt.legend()
plt.title('CHyPS')
plt.tight_layout()


plt.show()




