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

arg_tijd = np.arange(t_Dick[0],t_Dick[-1]-7,(t_Dick[-1]-7)/len(t_Dick))
flinearz = interp1d(arg_tijd, az_Dick)
flinearx = interp1d(arg_tijd, ax_Dick)

az_Dick_new = flinearz(t_Dick[:-900])
ax_Dick_new = flinearx(t_Dick[:-900])
t_Dick_new = t_Dick[:-900]
t_GPS = t_GPS[:-9]
vx_GPS_Dick = vx_GPS_Dick[:-9]

GPS_lon = bestand_Dick[['GPSau-lon']].values.flatten()[beginint:eindint+1][:-9]
GPS_lat = bestand_Dick[['GPSau-lat']].values.flatten()[beginint:eindint+1][:-9]


### Corrigeren versnellingsdata x-richting
vx0 = vx_GPS_Dick[0]

corrections = [0.0]
correcties = [0.0]

dt = 1/100

for i in range(1, len(vx_GPS_Dick)):
    corr = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_Dick_new[(i-1)*100+1:i*100+1])*dt)
    b = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_Dick_new[(i-1)*100+1:i*100+1]+interpol(corrections[-1], corr, 100))*dt)
    correcties += list(interpol(corrections[-1], corr, 100)+b)

    corrections.append(corr)

#         correcties += list(interpol(correcties[-1], corr, 100))
correcties = np.array(correcties)

ax_corr = ax_Dick_new[:-99] + correcties
vx_corr = np.cumsum(ax_corr*dt) + vx0
dx_corr = np.cumsum(vx_corr*dt)
dx_GPS_Dick = np.cumsum(vx_GPS_Dick*1)

###korte laserdata inlezen

# contents = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\laser_N60_kort.csv') 
# time = contents[['time']].values.flatten()
# laser_data = contents[['afstand_laser']].values.flatten()
# time = time-time[0]
# wegafstand = np.mean(laser_data)

# ### verschuiving van laser zodat het overeenkomt met versnellingsdata
# time = time[:-14197+900]
# laser_data = laser_data[14197-900:]

t_pol = np.arange(t_Dick_new[0],t_Dick_new[-100],t_Dick_new[-100]/len(t_Dick_new)/320)
flineardx = interp1d(t_Dick_new[:-99], dx_corr)
dx_pol = flineardx(t_pol)

###lange laserdata inlezen
contents = pd.read_csv('E:\\jan\\vop_3de_bach\\ritten_zwijnaarde_1\\laser_N60.csv')

rawtime = contents[['time']].values.flatten()
rawlaserfull = contents[['afstand_laser']].values.flatten()

time = rawtime[:(-14197+900)*320]
laser_full = rawlaserfull[(14197-900)*320:]

time = time-time[0]

time = time[:len(dx_pol)-len(time)]
laser_data = laser_full[:len(dx_pol)-len(laser_full)]

wegafstand = np.mean(laser_full[-32000*5:])



### laserdata in functie van afstand

flinearlaser = interp1d(dx_pol - dx_pol[0], laser_data)

dx_new = np.arange(0, dx_pol[-1]-dx_pol[0], 0.0005) # Het laaste argument is de afgelegede afstand tussen samples dx

linearlaser = flinearlaser(dx_new)

### lengte- en breedtegraad ook naar 32kHz brengen, is makkelijker voor te synchroniseren

flinearlontijd = interp1d(t_GPS, GPS_lon)
flinearlattijd = interp1d(t_GPS, GPS_lat)
lon_pol = flinearlontijd(t_pol)
lat_pol = flinearlattijd(t_pol)


### lengte- en breedtegraad in functie van de afstand


flinearlat = interp1d(dx_pol - dx_pol[0], lat_pol)
flinearlon = interp1d(dx_pol - dx_pol[0], lon_pol)


dx_new_coord = np.arange(0, dx_pol[-1]-dx_pol[0], 0.1) # Het laaste argument is de afgelegede afstand tussen samples dx

linearlat = flinearlat(dx_new_coord)
linearlon = flinearlon(dx_new_coord)

### mean profile depth

dist = dx_new
profile = wegafstand - linearlaser

baselength = 100 # base length in mm
dx = (dist[1] - dist[0])*10**3 # afstand tussen twee opeenvolgende intervallen in mm
intervals = int(baselength/dx) # het aantal intervallen dat over de base length gaat
MPD = []

for i in range(int(len(profile)/intervals)):
    wegdek = profile[i*intervals:(i+1)*intervals]
    PP = np.mean(wegdek)
    M1 = np.max(wegdek[:int(intervals/2)])
    M2 = np.max(wegdek[int(intervals/2):])
    MPD.append((M1 + M2)/2 - PP)

print(len(MPD), len(linearlon))
MPD = np.array(MPD)
peaks = np.where(MPD > 3)[0]
for e in peaks:  # vervelende pieken die eruit schieten wegwerken, klopt niet echt maar is beter voor de visualisatie 
    MPD[e] = (MPD[e-1]+MPD[e+1])/2

### kaart plotten met kleurcode van de MPD op elke plaats
kaart = plt.imread('mapN60.png')

MPD_norm = MPD/np.max(MPD)

BBox = (3.6591, 3.7201, 50.9744, 51.0152)

fig, ax = plt.subplots(figsize = (16, 14))

lettergrootte = 20

ax.scatter(linearlon[:-1], linearlat[:-1], zorder=1, alpha=1, c=MPD_norm, s=5, cmap='hot_r')
ax.set_title('Afgelegde weg', fontsize=lettergrootte)
ax.set_xlabel('lengtegraad', fontsize=lettergrootte)
ax.set_ylabel('breedtegraad', fontsize=lettergrootte)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(kaart, zorder=0, extent = BBox, aspect= 'equal')
plt.show()