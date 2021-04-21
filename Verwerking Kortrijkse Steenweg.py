import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)
  
### inlezen

contents = pd.read_csv('xyz_KortSt.csv')

accel_x = contents[['a_x']].values.flatten()
ax_laser = 9.81*accel_x
accel_z = contents[['a_z']].values.flatten()
az_laser = 9.81*accel_z

contents = pd.read_csv('xyz_Dick.csv')

accel_x = contents[['a_x']].values.flatten()
ax_Dick = 9.81*accel_x
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

vx_GPS = bestand_laser[['GPSshort-speed']].values.flatten()
vx_GPS_Dick = bestand_Dick[['GPSau-speed']].values.flatten()[beginint:eindint+1]*5/18

ax_Dick = ax_Dick[beginint*100+500:(eindint+1)*100+500]   # versnellingsdata loopt 5 seconden achter op GPS data voor de een of andere reden
az_Dick = az_Dick[beginint*100+500:(eindint+1)*100+500]

t = np.arange(len(ax_laser))/100.0
t_GPS = np.arange(len(vx_GPS))

lettergrootte = 20
plt.figure(figsize=(16,9))
plt.title('Snelheid in functie van de tijd, Kortrijkse Steenweg', fontsize=lettergrootte)
plt.plot(t_GPS/60, vx_GPS*18/5, label='GPS-snelheid lasernode')
plt.plot(t_GPS/60, vx_GPS_Dick*18/5, label='GPS-snelheid Dick node')
plt.legend()
#plt.axis([0,np.max(t[:-100]/60)+0.5, 0, np.max(vx_corr*18/5)+5])
plt.xticks(fontsize=lettergrootte)
plt.yticks(fontsize=lettergrootte)
plt.xlabel('tijd [min]', fontsize=lettergrootte)
plt.ylabel('snelheid [km/h]', fontsize=lettergrootte)
plt.show()

### komt goed overeen

### z-acceleratie vergelijken
sec1 = 0
sec2 = int(t[-1])
plt.figure(figsize=(9,6))
plt.plot(t[sec1*100:sec2*100]/60, az_laser[sec1*100:sec2*100], label='verticale versnelling in lasernode')
plt.plot(t[sec1*100:sec2*100]/60, az_Dick[sec1*100:sec2*100], label='verticale versnelling in Dick node', color='tab:orange')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, Kortrijkse Steenweg')
plt.xlabel('tijd [min]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()
### opvallend lage waarden van z-versnelling in lasernode tussen 3.9 en 4.5 minuten en in Dicknode tussen 3 en 3.6 minuten
### versnelling van laser-node ondervindt te veel invloed van trillingen constructie -> versnelling en GPS-snelheid van Dick node gebruiken voor afstandsprofiel

### correctie op versnelling

vx0 = vx_GPS_Dick[0]

corrections = [0.0]
correcties = [0.0]

dt = 1/100

for i in range(1, len(vx_GPS)):
    corr = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_Dick[(i-1)*100+1:i*100+1])*dt)
    b = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_Dick[(i-1)*100+1:i*100+1]+interpol(corrections[-1], corr, 100))*dt)
    correcties += list(interpol(corrections[-1], corr, 100)+b)

    corrections.append(corr)

#         correcties += list(interpol(correcties[-1], corr, 100))
correcties = np.array(correcties)

ax_corr = ax_Dick[:-99] + correcties
vx_corr = np.cumsum(ax_corr*dt) + vx0
dx_corr = np.cumsum(vx_corr*dt)
dx_GPS_Dick = np.cumsum(vx_GPS_Dick*1)

### bewijs dat versnellingsdata van Dick node 5 seconden achterloopt op zijn GPS-snelheden
k = 40
dt = 1/100
plt.figure(figsize=(16,6))
plt.plot(np.arange(len(ax_Dick[:k*100]))/100, np.cumsum(ax_Dick[:k*100]*dt)+vx0, label='Geïntegreerde versnelling')
plt.plot(np.arange(len(vx_GPS_Dick[1:1+k])), vx_GPS_Dick[1:1+k], label='GPS snelheid Dick node')
plt.legend()
plt.grid()
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [m/s]')
plt.title('Snelheid in de eerste {} seconden van Kortrijkse Steenweg'.format(k))
plt.show()

### lasermetingen verwerken
#### kruiscorrelatie
contents = pd.read_csv('laser_KortSt_kort.csv') # 100 hertz versie van de laserdata zodat dt gelijk is aan de dt van de acceleratie en de snelheid

time = contents[['time']].values.flatten()
laser_data = contents[['afstand_laser']].values.flatten()
time = time-time[0]
print(time[-1])
# kruiscorrelatie werkt niet met az_laser om de een of andere reden
correlatie = np.correlate((laser_data-np.average(laser_data))/max(np.abs(laser_data-np.average(laser_data))), (az_Dick-np.average(az_Dick))/max(np.abs(az_Dick-np.average(az_Dick))), "valid")
print(az_laser[:20])

checknan = np.isnan(az_laser)
value = np.sum(checknan)
nans = np.where(checknan)
print(value, nans)
### laatste 77 waarden van az_laser zijn nan -> verklaring waarom correlate niet werkt

plt.figure(figsize=(16,9))
plt.plot(np.abs(correlatie)/max(np.abs(correlatie)))
plt.show()

### als tijd in laserbestand klopt dan verwachten we een zeer vroege piek: laserbestand begint op 13:21:05.8 en versnellingsbestand op 13:21:07 

maxindex = int(np.where(np.abs(correlatie[:200]) == max(np.abs(correlatie[:200])))[0])
print(maxindex) # is 1.4 seconden, komt overeen met de verwachting
lengte = len(az_laser)
laser_rit = laser_data[maxindex:maxindex+lengte]
time_rit = time[maxindex:maxindex+lengte]
wegafstand = laser_data[420]

plt.figure(figsize=(16,9))
plt.plot(time_rit, wegafstand - laser_rit, label='opgemeten laserdata')
plt.plot(t, az_laser, label='verticale versnelling in lasernode')
plt.plot(t, az_Dick, label='verticale versnelling in Dick node')
plt.legend()
plt.xlabel('tijd [s]')
plt.ylabel('afstand [mm], versnelling [m/s²]')
plt.title('laserdata 100Hz')
plt.show()
