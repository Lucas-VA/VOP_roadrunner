import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import signal

### data inlezen
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

contents = pd.read_csv('laser_KortSt_kort.csv') # 100 hertz versie van de laserdata zodat dt gelijk is aan de dt van de acceleratie en de snelheid

time = contents[['time']].values.flatten()
laser_data = contents[['afstand_laser']].values.flatten()
time = time-time[0]

wegafstand = np.mean(laser_data[29000:30400])

maxindex = 3559
lengte = len(az_laser)
laser_rit = laser_data[maxindex:maxindex+lengte]
time_rit = time[maxindex:maxindex+lengte]-time[maxindex:maxindex+lengte][0]



### gemeten afstand corrigeren (echte werk)

### kijken tussen seconde x1/100 en x2/100
x1 = 10000
x2 = 12000

g = np.mean(az_laser[24600:26500]) # gemeten z-acceleratie in laser node in rust

az_corr = az_laser[x1:x2]-g # z-acceleratie zonder invloed zwaartekracht

plt.plot(t[x1:x2], az_corr)
plt.title('verticale versnelling zonder invloed zwaartekracht')
plt.xlabel('tijd [s]')
plt.ylabel('versnelling [m/s²]')
plt.show()

# frequenties < 40Hz = trillingen van wagen -> lowpass filter van 40Hz om trillingen van de wagen weg te werken

sos = signal.butter(2, 40, btype='lowpass', fs=100, output='sos')
az_filtered = signal.sosfilt(sos, az_corr)

plt.plot(t[x1:x2], az_filtered)
plt.title('verticale versnelling zonder invloed zwaartekracht na lowpass filter van 40Hz')
plt.xlabel('tijd [s]')
plt.ylabel('versnelling [m/s²]')
plt.show()

snelh_corr = integrate.cumtrapz(az_corr, dx=1/100.0, initial=0) # moeilijk om een beginvoorwaarde aan te geven
afstand_corr = integrate.cumtrapz(snelh_corr, dx=1/100.0, initial=0) # idem, moeilijke beginvoorwaarde

snelh_filt = integrate.cumtrapz(az_filtered, dx=1/100.0, initial=0)
afstand_filt = integrate.cumtrapz(snelh_filt, dx=1/100.0, initial=0)


plt.plot(t[x1:x2], afstand_corr)
plt.title('verticale versnelling 2 keer geïntegreerd')
plt.xlabel('tijd [s]')
plt.ylabel('afstand [m]')
plt.show()

plt.plot(t[x1:x2], afstand_filt)
plt.title('verticale versnelling < 40Hz 2 keer geïntegreerd')
plt.xlabel('tijd [s]')
plt.ylabel('afstand [m]')
plt.show()

## lowpass filter heeft geen invloed


plt.plot(t[x1:x2], wegafstand - laser_rit[x1:x2], label='gemeten afstand tov wegdek')
plt.plot(t[x1:x2], wegafstand - (laser_rit[x1:x2]+afstand_corr), label='afstand tov wegdek na correctie')
plt.title('profiel van het wegdek')
plt.xlabel('tijd [s]')
plt.ylabel('afstand tov wegdek [mm]')
plt.legend()
plt.show()

# na een tijd domineren de meetfouten van de versnelling omdat de z-richting van de accelerometer niet perfect horizontaal is
