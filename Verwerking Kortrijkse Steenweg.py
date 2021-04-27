import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)
  
### inlezen

contents = pd.read_csv('xyz_KortSt.csv')

accel_x = contents[['a_x']].values.flatten()
ax_laser = 9.81*accel_x[:-100]  # in laatste seconden zitten er NaN values tussen
accel_z = contents[['a_z']].values.flatten()
az_laser = 9.81*accel_z[:-100]

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
vx_GPS_Dick = bestand_Dick[['GPSau-speed']].values.flatten()[beginint:eindint+1+1200]*5/18

ax_Dick = ax_Dick[beginint*100+400:(eindint+1)*100+120000+400]   # versnellingsdata loopt 4 seconden achter op GPS data voor de een of andere reden
az_Dick = az_Dick[beginint*100+400:(eindint+1)*100+120000+400]

t_laser = np.arange(len(ax_laser))/100.0
t_Dick = np.arange(len(ax_Dick))/100.0
t_GPS = np.arange(len(vx_GPS_Dick))

### stuk op snelheid = 0
sec1 = 24500
sec2 = 26500
plt.figure()
plt.plot(t_laser[sec1:sec2], az_laser[sec1:sec2], label='verticale versnelling in lasernode')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, Kortrijkse Steenweg')
plt.xlabel('tijd [s]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()
sec3 = 51900    # offset = 519s - 245s = 274s
sec4 = 53900
plt.figure()
plt.plot(t_Dick[sec3:sec4], az_Dick[sec3:sec4], label='verticale versnelling in Dick node', color='tab:orange')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, Kortrijkse Steenweg')
plt.xlabel('tijd [s]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()

### z-acceleratie vergelijken
offset = 27400
eindint = len(az_laser)+offset
plt.figure()
plt.title('Snelheid in functie van de tijd, Kortrijkse Steenweg')
plt.plot(t_GPS[int(offset/100):int(eindint/100)], vx_GPS_Dick[int(offset/100):int(eindint/100)]*18/5, label='GPS-snelheid Dick node')
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [km/h]')
plt.show()

plt.figure()
plt.plot(t_laser, az_laser, label='verticale versnelling in lasernode')
plt.plot(t_laser, az_Dick[offset:eindint], label='verticale versnelling in Dick node', color='tab:orange')
plt.legend()
plt.title('z-acceleratie in functie van de tijd, Kortrijkse Steenweg')
plt.xlabel('tijd [s]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()
### versnelling van laser-node ondervindt te veel invloed van trillingen constructie -> versnelling en GPS-snelheid van Dick node gebruiken voor afstandsprofiel

### herdefinieer de snelheid en versnelling in de Dick node
vx_GPS_Dick = vx_GPS_Dick[int(offset/100):int(eindint/100)]
t_GPS = np.arange(len(vx_GPS_Dick))
az_Dick = az_Dick[offset:eindint]
ax_Dick = ax_Dick[offset:eindint]
# t_Dick is nu gewoon gelijk aan t_laser

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

### Correcties op versnelling verbeteren, vooral tussen 232s en 243s, tussen 245s en 247s en tussen 265s en 270s

correcties[232*100+1:233*100+1] = interpol(corrections[232], corrections[234], 100)

correcties[236*100+1:238*100+1] = interpol(corrections[236], corrections[238], 200)

x1 = 239
x2 = 241
corr = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1])*dt)
b = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1] + interpol(corrections[x1], corr, (x2-x1)*100))*dt)
correcties[x1*100+1:x2*100+1] = interpol(corrections[x1], corr, (x2-x1)*100) + b/(x2-x1)

x1 = 241
x2 = 243
corr = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1])*dt)
b = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1] + interpol(corrections[x1], corr, (x2-x1)*100))*dt)
correcties[x1*100+1:(x2-1)*100+1] = interpol(corrections[x1], corr, (x2-x1)*100)[:100] + b/(x2-x1)

correcties[242*100+1:243*100+1] = interpol(interpol(corrections[x1], corr, (x2-x1)*100)[100], corrections[243], 100)

x1 = 245
x2 = 249
corr = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1])*dt)
b = float(vx_GPS_Dick[x2]-vx_GPS_Dick[x1]-np.sum(ax_Dick[x1*100+1:x2*100+1] + interpol(corrections[x1], corr, (x2-x1)*100))*dt)
correcties[x1*100+1:x2*100+1] = interpol(corrections[x1], corr, (x2-x1)*100) + b/(x2-x1)

ax_corr = ax_Dick[:-99] + correcties
vx_corr = np.cumsum(ax_corr*dt) + vx0
dx_corr = np.cumsum(vx_corr*dt)
dx_GPS_Dick = np.cumsum(vx_GPS_Dick*1)

### bewijs dat versnellingsdata van Dick node 4 seconden achterloopt op zijn GPS-snelheden
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

k1 = 230
k2 = 270

major_ticks = np.arange(k1, k2+1, 5)
minor_ticks = np.arange(k1, k2+1, 1)

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(1, 1, 1)

ax.plot(t[k1*100:k2*100], ax_Dick[k1*100:k2*100], label='gemeten versnelling')
ax.plot(t[k1*100:k2*100], ax_corr[k1*100:k2*100], label='gecorrigeerde versnelling')
ax.legend()
ax.set_xlabel('tijd [s]')
ax.set_ylabel('versnelling [m/s²]')
ax.set_title('Versnelling na geïnterpoleerde correctie per seconde + constante waarde')
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.grid(which='both')
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
