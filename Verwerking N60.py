import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.interpolate import interp1d

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)
  
### inlezen

contents = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\xyz_N60.csv')

accel_x = contents[['a_x']].values.flatten()
ax_laser = 9.81*accel_x
accel_z = contents[['a_z']].values.flatten()
az_laser = 9.81*accel_z

contents = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\xyz_Dick.csv')

accel_x = contents[['a_x']].values.flatten()
ax_Dick = 9.81*accel_x
accel_z = contents[['a_z']].values.flatten()
az_Dick = 9.81*accel_z

bestand_laser = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\laser-measurements\\laser-node\\run 0 - N60 sensor.csv')
bestand_Dick = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\laser-measurements\\dick-node\\158033857_120.csv')

time_laser = bestand_laser[['time']].values.flatten()
begint = time_laser[0][:-8] + str(int(time_laser[0][-8:-6])-2) + time_laser[0][-6:]
eindt = time_laser[-1][:-8] + str(int(time_laser[-1][-8:-6])-2) + time_laser[-1][-6:]

time_Dick = bestand_Dick[['time']].values.flatten()
beginint = int(np.where(time_Dick == begint)[0])
eindint = int(np.where(time_Dick == eindt)[0])

vx_GPS = bestand_laser[['GPSshort-speed']].values.flatten()
vx_GPS_Dick = bestand_Dick[['GPSau-speed']].values.flatten()[beginint:eindint+1]*5/18

ax_Dick = ax_Dick[beginint*100:(eindint+1)*100]
az_Dick = az_Dick[beginint*100:(eindint+1)*100]

t_laser = np.arange(len(ax_laser))/100.0
t_Dick = np.arange(len(ax_Dick))/100.0
t_GPS = np.arange(len(vx_GPS_Dick))

### Herschaling tijds as az_Dick en ax_Dick

arg_tijd = np.arange(t_Dick[0],t_Dick[-1]-7,(t_Dick[-1]-7)/len(t_Dick))
flinearz = interp1d(arg_tijd, az_Dick)
flinearx = interp1d(arg_tijd, ax_Dick)

az_Dick_new = flinearz(t_Dick[:-900])
ax_Dick_new = flinearx(t_Dick[:-900])

snelh_Dick = integrate.cumtrapz(ax_Dick-np.average(ax_Dick[132000:]), dx=1/100.0, initial=0)
snelh_Dick_new = integrate.cumtrapz(ax_Dick_new-np.average(ax_Dick_new[132000:]), dx=1/100.0, initial=0)

plt.figure()
plt.plot(t_laser, az_laser, label='verticale versnelling in lasernode')
plt.plot(t_Dick, az_Dick, label='verticale versnelling in dicknode')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, N60')
plt.xlabel('tijd [s]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()
### Visueel: Komen overeen in het begin, op het einde Dicknode 8s later

plt.figure()
plt.plot(t_GPS, vx_GPS, label='GPS snelheid lasernode')
plt.plot(t_GPS, vx_GPS_Dick, label='GPS snelheid Dicknode')
plt.legend()
plt.title('GPS-snelheid')
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [m/s]')
plt.show()

### Visueel: Komen overeen


plt.figure()
plt.plot(t_Dick, snelh_Dick, label='integratie Dicknode')
plt.plot(t_GPS, vx_GPS_Dick, label='GPS snelheid Dicknode')
plt.legend()
plt.title('GPS-snelheid')
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [m/s]')
plt.show()

### Visueel: Komen overeen in het begin, op het einde Dicknode 8s later

plt.figure()
plt.plot(t_laser, az_laser, label='verticale versnelling in lasernode')
plt.plot(t_Dick[:-900], az_Dick_new, label='verticale versnelling in dicknode')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, N60')
plt.xlabel('tijd [s]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()

plt.figure()
plt.plot(t_Dick[:-900], snelh_Dick_new, label='integratie Dicknode')
plt.plot(t_GPS, vx_GPS_Dick, label='GPS snelheid Dicknode')
plt.legend()
plt.title('GPS-snelheid')
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [m/s]')
plt.show()

### Na herschaling komt alles redelijk overeen