import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
from scipy import signal

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)
  
  ### inlezen

# stuk Kortrijkse steenweg is van 41*60s tot 52*60s voor GPS-signaal
begint = 2220+274
eindt = 52*60+20

contents = pd.read_csv('xyz_Dick.csv')

accel_x = contents[['a_x']].values.flatten()
ax_Dick = 9.81*accel_x
accel_z = contents[['a_z']].values.flatten()
az_Dick = 9.81*accel_z

bestand_Dick = pd.read_csv('D:\\UGent docs\\VOP_docs\\Metingen paasvakantie\\dick-node\\158033857_120.csv')

vx_GPS_Dick = bestand_Dick[['GPSau-speed']].values.flatten()[begint:eindt]*5/18

ax_Dick = ax_Dick[begint*100+400:(eindt)*100+400]   # versnellingsdata loopt 4 seconden achter op GPS data voor de een of andere reden
az_Dick = az_Dick[begint*100+400:(eindt)*100+400]

t = np.arange(len(ax_Dick))/100.0
t_GPS = np.arange(len(vx_GPS_Dick))

GPS_lon = bestand_Dick[['GPSau-lon']].values.flatten()[begint:eindt]
GPS_lat = bestand_Dick[['GPSau-lat']].values.flatten()[begint:eindt]

indices_fout = np.where(GPS_lon == 0)[0]
print(indices_fout) # geen ontbrekende metingen van GPS
print(len(ax_Dick), len(vx_GPS_Dick))
vx0 = vx_GPS_Dick[0]

# op einde loopt versnelling 2 seconden achter op snelheid

arg_tijd = np.arange(t[0],t[-1]+2,(t[-1]+2)/len(t))
flinearz = interp1d(arg_tijd, az_Dick)
flinearx = interp1d(arg_tijd, ax_Dick)

az_new = flinearz(t)
ax_new = flinearx(t)

BBox = (GPS_lon.min(), GPS_lon.max(), GPS_lat.min(), GPS_lat.max())
BBox = (3.6596, 3.7167, 51.0136, 51.0308)
print(BBox)

kaart = plt.imread('mapKortrijkse_Steenweg.png')

fig, ax = plt.subplots(figsize = (16, 14))

lettergrootte = 20

ax.scatter(GPS_lon, GPS_lat, zorder=1, alpha= 1, c='b', s=5)
ax.scatter(GPS_lon[0], GPS_lat[0], zorder=1, c='r', s=20)
ax.scatter(GPS_lon[-1], GPS_lat[-1], zorder=1, c='lime', s=20)
ax.set_title('Afgelegde weg', fontsize=lettergrootte)
ax.set_xlabel('lengtegraad', fontsize=lettergrootte)
ax.set_ylabel('breedtegraad', fontsize=lettergrootte)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(kaart, zorder=0, extent = BBox, aspect= 'equal')

### correctie op versnelling

vx0 = vx_GPS_Dick[0]

corrections = [0.0]
correcties = [0.0]

dt = 1/100

for i in range(1, len(vx_GPS_Dick)):
    corr = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_new[(i-1)*100+1:i*100+1])*dt)
    b = float(vx_GPS_Dick[i]-vx_GPS_Dick[i-1]-np.sum(ax_new[(i-1)*100+1:i*100+1]+interpol(corrections[-1], corr, 100))*dt)
    correcties += list(interpol(corrections[-1], corr, 100)+b)

    corrections.append(corr)

#         correcties += list(interpol(correcties[-1], corr, 100))

correcties = np.array(correcties)

ax_corr = ax_new[:-99] + correcties
vx_corr = np.cumsum(ax_corr*dt) + vx0
dx_corr = np.cumsum(vx_corr*dt)
dx_GPS_Dick = np.cumsum(vx_GPS_Dick*1)

### laser

maxindex = 3700 #- 3100
lengte = len(az_new)

# 32kHz data
contents = pd.read_csv('laser_KortSt.csv')

time = contents[['time']].values.flatten()
laser_full = contents[['afstand_laser']].values.flatten()
time = time-time[0]

wegafstand = np.mean(laser_full[29000*320:30400*320])

flinearz = interp1d(time*255/287, laser_full)
t_new = np.arange(0, time[-1]*255/287, 1/32000)
laser_new = flinearz(t_new)

maxindex = 400
laser_newer = laser_new[maxindex*320:(maxindex+lengte)*320]
t_newer = t_new[maxindex*320:(maxindex+lengte)*320]-time[maxindex*320:(maxindex+lengte)*320][0]

shift = 5100

plt.figure(figsize=(16,9))
plt.plot(t_newer, (wegafstand - laser_newer)/10, label='opgemeten laserdata')
plt.plot(t, az_new, label='verticale versnelling in koffernode')
plt.plot(t_GPS, vx_GPS_Dick, label='GPS-snelheid koffernode')
plt.legend()
plt.xlabel('tijd [s]')
plt.ylabel('afstand [cm], snelheid [m/s], versnelling [m/s²]')
plt.title('laserdata 32kHz')
# plt.xlim(220, 290)
plt.show()

# na resamplen van de tijd komt het nog steeds niet overeen


dx_pol = [dx_corr[0]]
print(dx_pol)
t_pol = [t[0]]
for i in range(len(dx_corr)-1):
    dx_extra = interpol(dx_corr[i], dx_corr[i+1], 320)
    t_extra = interpol(t[i], t[i+1], 320)
    dx_pol += list(dx_extra)
    t_pol += list(t_extra)
dx_pol = np.array(dx_pol)
t_pol = np.array(t_pol)

laser_data = laser_newer[:-31999]

### laserdata in functie van afstand

flinearlaser = interp1d(dx_pol - dx_pol[0], laser_data)

dist = np.arange(0, dx_pol[-1]-dx_pol[0], 0.0005) # Het laaste argument is de afgelegede afstand tussen samples dx

linearlaser = flinearlaser(dist)

# plt.figure(figsize=(16,9))
# plt.plot(dist, wegafstand - linearlaser)
# plt.show()

lon_pol = [GPS_lon[0]]
t_pol = [t[0]]
for i in range(len(GPS_lon)-1):
    extra = interpol(GPS_lon[i], GPS_lon[i+1], 32000)
    lon_pol += list(extra)
lon_pol = np.array(lon_pol)

lat_pol = [GPS_lat[0]]
t_pol = [t[0]]
for i in range(len(GPS_lat)-1):
    extra = interpol(GPS_lat[i], GPS_lat[i+1], 32000)
    lat_pol += list(extra)
lat_pol = np.array(lat_pol)

### lengte- en breedtegraad in functie van de afstand

flinearlat = interp1d(dx_pol - dx_pol[0], lat_pol)
flinearlon = interp1d(dx_pol - dx_pol[0], lon_pol)


dx_new = np.arange(0, dx_pol[-1]-dx_pol[0], 0.1) # Het laaste argument is de afgelegede afstand tussen samples dx

linearlat = flinearlat(dx_new)
linearlon = flinearlon(dx_new)

kaart = plt.imread('mapKortrijkse_Steenweg.png')

fig, ax = plt.subplots(figsize = (16, 14))

lettergrootte = 20

ax.scatter(linearlon, linearlat, zorder=1, alpha= 1, c='b', s=1)
ax.scatter(GPS_lon[0], GPS_lat[0], zorder=1, c='r', s=20)
ax.scatter(GPS_lon[-1], GPS_lat[-1], zorder=1, c='lime', s=20)
ax.set_title('Afgelegde weg', fontsize=lettergrootte)
ax.set_xlabel('lengtegraad', fontsize=lettergrootte)
ax.set_ylabel('breedtegraad', fontsize=lettergrootte)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(kaart, zorder=0, extent = BBox, aspect= 'equal')

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

dist_MPD = np.arange(0, dist[-1], baselength*10**(-3))

plt.plot(dist_MPD[:-1], MPD)
plt.title('MPD stuk Kortrijkse Steenweg')
plt.xlabel('afstand [m]')
plt.show()

plt.plot(dist, profile)
plt.title('Wegprofiel stuk Kortrijkse Steenweg')
plt.ylabel('afstand ten opzichte van het wegdek [mm]')
plt.xlabel('afstand [m]')
plt.show()

MPD_norm = np.clip(MPD, 0, 3)/np.max(np.clip(MPD, 0, 3))

plt.plot(dist_MPD[:-1], MPD_norm)
plt.title('genormeerde MPD stuk Kortrijkse Steenweg')
plt.xlabel('afstand [m]')
plt.show()

kaart = plt.imread('mapKortrijkse_Steenweg.png')

BBox = (3.6596, 3.7167, 51.0136, 51.0308)

fig, ax = plt.subplots(figsize = (16, 14))

lettergrootte = 20

ax.scatter(linearlon[:-1], linearlat[:-1], zorder=1, alpha=1, c=MPD_norm, s=0.5, cmap='hot_r')
ax.set_title('Afgelegde weg', fontsize=lettergrootte)
ax.set_xlabel('lengtegraad', fontsize=lettergrootte)
ax.set_ylabel('breedtegraad', fontsize=lettergrootte)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(kaart, zorder=0, extent = BBox, aspect= 'equal')

