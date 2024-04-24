import matplotlib.pyplot as plt
from transfer_function import TransferFunction
import numpy as np

# FSRの説明図を作成するために用いたもの

L = np.array([100e-6,100e-6])
K = np.array([0.5,0.1,0.5])
xaxis = np.arange(1540e-9,1580e-9,0.01e-9) 
axis = np.arange(1540,1580,0.01)
data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
data1 = data.simulate(xaxis)

L = np.array([150e-6,150e-6])
data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
data2 = data.simulate(xaxis)

L = np.array([100e-6,150e-6])
data = TransferFunction(L,K,config={'center_wavelength':1550e-9,'eta':0.996,'n_eff':2.2,'n_g':4.4,'alpha':52.96})
data3 = data.simulate(xaxis)

plt.rcParams["xtick.direction"]="in"
plt.rcParams["ytick.direction"]="in"
plt.plot(axis,data1,label="L = 100(µm)")
plt.plot(axis,data2,label="L = 100(µm)")
plt.plot(axis,data3,label="L = 100,150(µm)")
plt.xlabel("Wavelength (nm)",fontsize=13)
plt.ylabel("Transmittance (dB)",fontsize = 13)
plt.ylim(-20,0)
# plt.xticks(np.arange(1540,1590,1))
# plt.minorticks_on()
plt.legend(bbox_to_anchor=(1,0),loc="lower right")
plt.show()