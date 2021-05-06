import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift

### inlezen en synchroniseren
contents = pd.read_csv('xyz_KortSt.csv')
accel_z = contents[['a_z']].values.flatten()
az_laser = 9.81*accel_z[:-100]

contents = pd.read_csv('xyz_Dick.csv')
accel_z = contents[['a_z']].values.flatten()
az_Dick = 9.81*accel_z

bestand_laser = pd.read_csv('D:\\UGent docs\\VOP_docs\\Metingen paasvakantie\\laser-node\\run 2 - kortrijkse steenweg sensor.csv')
bestand_Dick = pd.read_csv('D:\\UGent docs\\VOP_docs\\Metingen paasvakantie\\dick-node\\158033857_120.csv')

time_laser = bestand_laser[['time']].values.flatten()
begint = time_laser[0][:-8] + str(int(time_laser[0][-8:-6])-2) + time_laser[0][-6:]
eindt = time_laser[-1][:-8] + str(int(time_laser[-1][-8:-6])-2) + time_laser[-1][-6:]

time_Dick = bestand_Dick[['time']].values.flatten()
beginint = int(np.where(time_Dick == begint)[0])
eindint = int(np.where(time_Dick == eindt)[0])

az_Dick = az_Dick[beginint*100+400:(eindint+1)*100+120000+400]

t = np.arange(len(az_laser))/100.0

offset = 27400
eindint = len(az_laser)+offset

az_Dick = az_Dick[offset:eindint]

### Fourier plotten

N = len(az_laser)
T = 1/100
g = np.mean(az_laser[24600:26500])

yf = fft(az_laser - g)
xf = fftfreq(N, T)
xf = fftshift(xf)
# yplot = fftshift(yf)
plt.xlim(0, 50)
plt.plot(xf, 2.0/N * np.abs(yf))
plt.title('Fourier spectrum van verticale trillingen in laser node')
plt.xlabel('frequentie [Hz]')
# plt.ylabel('Modulus Fourier getransformeerde')
plt.grid()
plt.show()

print(xf[np.where(yf[15000:] == np.max(yf[15000:]))[0]+15000], np.where(yf[15000:] == np.max(yf[15000:]))) # max waar frequentie > 0

g_Dick = np.mean(az_Dick[24600:26500])
az_Dcorr = az_Dick - g_Dick

yf = fft(az_Dcorr)
xf = fftfreq(N, T)
xf = fftshift(xf)
yplot = fftshift(yf)
plt.xlim(0, 50)
plt.plot(xf, 2.0/N * np.abs(yplot))
plt.title('Fourier spectrum van verticale trillingen in koffer node')
plt.xlabel('frequentie [Hz]')
# plt.ylabel('Modulus Fourier getransformeerde')
plt.grid()
plt.show()

print(xf[np.where(yplot[15000:] == np.max(yplot[15000:]))[0]+15000], np.where(yplot[15000:] == np.max(yplot[15000:])))
