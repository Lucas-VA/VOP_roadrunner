import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)

name = 'xyz_Zwijnaarde' + '3' + '.csv'
contents = pd.read_csv(name)

accel_x = contents[['a_x']].values.flatten()
ax = 9.81*accel_x[:-100]    # in de laatste seconde zijn de metingen van de accelerometer onvolledig
accel_z = contents[['a_z']].values.flatten()
az = 9.81*accel_z[:-100]

bestandsnaam = 'D:\\UGent docs\\VOP_docs\\Data testrit Zwijnaarde\\158033857_' + '3' + '.csv'

vx_GPS = pd.read_csv(bestandsnaam)[['GPSau-speed']].values.flatten()[:-1]*5/18
lengtegraad = pd.read_csv(bestandsnaam)[['GPSau-lon']].values.flatten()
breedtegraad = pd.read_csv(bestandsnaam)[['GPSau-lat']].values.flatten()
index = next(i for i,v in enumerate(lengtegraad) if v != 0) # seconden waar GPS niet meet wegdoen

lengtegraad = lengtegraad[index:]
breedtegraad = breedtegraad[index:]
ax = ax[int(index*100):]
az = az[int(index*100):]
vx_GPS = vx_GPS[index:]
t = np.arange(len(ax))/100.0

indices_fout = np.where(lengtegraad == 0)[0]
print(indices_fout)
for ind, el in enumerate(indices_fout):
    if indices_fout[ind-1] == el - 1:
        j += 1
        vx_GPS[el] = waarden_vx[j]
        lengtegraad[el] = waarden_lon[j]
        breedtegraad[el] = waarden_lat[j]
    elif ind != len(indices_fout)-1:
        j = ind
        while j != len(indices_fout)-1 and indices_fout[j+1] == indices_fout[j]+1:
            j += 1
        eind = indices_fout[j] + 1
        waarden_vx = interpol(vx_GPS[el-1], vx_GPS[eind], eind - el + 1)
        waarden_lon = interpol(lengtegraad[el-1], lengtegraad[eind], eind - el + 1)
        waarden_lat = interpol(breedtegraad[el-1], breedtegraad[eind], eind - el + 1)
        vx_GPS[el] = waarden_vx[0]
        lengtegraad[el] = waarden_lon[0]
        breedtegraad[el] = waarden_lat[0]
        j = 0
    else:
        vx_GPS[el] = (vx_GPS[el+1] + vx_GPS[el-1])/2
        lengtegraad[el] = (lengtegraad[el+1] + lengtegraad[el-1])/2
        breedtegraad[el] = (breedtegraad[el+1] + breedtegraad[el-1])/2

vx0 = vx_GPS[0]

corrections = [0.0]
correcties = [0.0]

dt = 1/100

for i in range(1, len(vx_GPS)):
    corr = float(vx_GPS[i]-vx_GPS[i-1]-np.sum(ax[(i-1)*100+1:i*100+1])*dt)
    b = float(vx_GPS[i]-vx_GPS[i-1]-np.sum(ax[(i-1)*100+1:i*100+1]+interpol(corrections[-1], corr, 100))*dt)
    correcties += list(interpol(corrections[-1], corr, 100)+b)

    corrections.append(corr)

#         correcties += list(interpol(correcties[-1], corr, 100))
correcties = np.array(correcties)

corr = float(vx_GPS[36]-vx_GPS[33]-np.sum(ax[33*100+1:36*100+1])*dt)
b = float(vx_GPS[36]-vx_GPS[33]-np.sum(ax[33*100+1:36*100+1] + interpol(corrections[33], corr, 300))*dt)
correcties[33*100+1:36*100+1] = interpol(corrections[33], corr, 300) + b/3

correction = interpol(corrections[33], corr, 300)[-1]
print(correction)

corr = float(vx_GPS[42]-vx_GPS[36] - np.sum(ax[36*100+1:42*100+1])*dt)
print(corr)
b = float(vx_GPS[42]-vx_GPS[36] - np.sum(ax[36*100+1:42*100+1] + interpol(correction, corr, 600))*dt)
print(b)
correcties[36*100+1:42*100+1] = interpol(correction, corr, 600) + b/6

corr = float(vx_GPS[123]-vx_GPS[121]-np.sum(ax[121*100+1:123*100+1])*dt)
b = float(vx_GPS[123]-vx_GPS[121]-np.sum(ax[121*100+1:123*100+1] + interpol(corrections[121], corr, 200))*dt)
correcties[121*100+1:123*100+1] = interpol(corrections[121], corr, 200) + b/2
    
ax_corr = ax[:-99] + correcties
vx_corr = np.cumsum(ax_corr*dt) + vx0
dx_corr = np.cumsum(vx_corr*dt)
dx_GPS = np.cumsum(vx_GPS*1)

print(t[-1], t[-1]/60)

plt.figure()
plt.plot(t/60, az)
plt.title('z-acceleratie in functie van de tijd')
plt.xlabel('tijd [min]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()

# contents = pd.read_csv('laser_rit1.csv')

# time = contents[['time']].values.flatten()
# laser_data = contents[['afstand_laser']].values.flatten()
# time = time-time[0]

# ### versnellingsdata is 253.99 s lang

# eindt = 8.6297
# eindsample = int(eindt*60*32000)    # 32000 samples per seconde
# beginsample = int((eindt*60 - 253.99)*32000)

contents = pd.read_csv('laser_rit1_kort.csv') # Ik gebruik hier de 100 hertz versie van de laserdata zodat dt gelijk is aan de dt van de acceleratie en de snelheid

time = contents[['time']].values.flatten()
laser_data = contents[['afstand_laser']].values.flatten()
time = time-time[0]

# correlatie = np.correlate(laser_data,vx_corr,"valid") # snelheids correlatie, lijkt niet zo intressant
correlatie = np.correlate(laser_data-np.average(laser_data),az-np.average(az),"valid")
plt.figure(figsize=(16,9))
plt.plot(correlatie/np.max(correlatie))
plt.show()

#beginsample = np.argmax(correlatie)
beginsample = 26364 # index die overeenkomt met de 2de hoogste piek
#eindsample = beginsample + len(vx_corr)
eindsample = beginsample + len(az)


tijd = time[beginsample:eindsample]-time[beginsample:eindsample][0]
datalaser = laser_data[beginsample:eindsample]

plt.figure(figsize=(16,9))
plt.plot(tijd/60, (datalaser[0]-datalaser)/10+10, label='opgemeten laserdata')  # +10 is simpelweg om de laserdata en versnellingsdata beter te kunnen vergelijken, heeft geen fysische betekenis
plt.plot(t/60, vx_corr, label='opgemeten snelheid')
plt.legend()
plt.xlabel('tijd [min]')
plt.ylabel('afstand [cm], snelheid [m/s]')
plt.title('laserdata overeenkomstig met snelheidsdata')
plt.show()

plt.figure(figsize=(16,9))
plt.plot(tijd/60, (datalaser[0]-datalaser)/10+10, label='opgemeten laserdata')  # +10 is simpelweg om de laserdata en versnellingsdata beter te kunnen vergelijken, heeft geen fysische betekenis
plt.plot(t/60, az, label='opgemeten versnelling')
plt.legend()
plt.xlabel('tijd [min]')
plt.ylabel('afstand [cm], versnelling [m/s²]')
plt.title('laserdata overeenkomstig met versnellingsdata')
plt.show()

### ingezoomd tussen minuut 3.3 en 3.8
sec1 = 3.3*60
sec2 = 3.8*60
slice1 = int(sec1*32000)
slice2 = int(sec2*32000)
tijd_zoom = tijd[slice1:slice2]
datalaser_zoom = datalaser[slice1:slice2]
t_zoom = t[int(sec1*100):int(sec2*100)]
az_zoom = az[int(sec1*100):int(sec2*100)]

plt.figure(figsize=(16,9))
plt.plot(tijd_zoom, (datalaser_zoom[0]-datalaser_zoom)/10+9, label='opgemeten laserdata')   # +9 is simpelweg om de laserdata en versnellingsdata beter te kunnen vergelijken, heeft geen fysische betekenis
plt.plot(t_zoom, az_zoom, label='opgemeten versnelling')
plt.legend()
plt.xlabel('tijd [sec]')
plt.ylabel('afstand [cm], versnelling [m/s²]')
plt.title('laserdata overeenkomstig met versnellingsdata')
plt.show()

### versnellingsdata plotten met de 32 kHz laserdata

contents = pd.read_csv('laser_rit1.csv')

time = contents[['time']].values.flatten()
laser_data = contents[['afstand_laser']].values.flatten()
time = time-time[0]

beginsample = int(beginsample*320)
eindsample = int(eindsample*320)    # 32000 samples per seconde
tijd = time[beginsample:eindsample]-time[beginsample:eindsample][0]
datalaser = laser_data[beginsample:eindsample]

lettergrootte = 20
plt.figure(figsize=(16,9))
plt.plot(tijd/60, (datalaser[0]-datalaser)/10+6, label='opgemeten laserdata')
plt.plot(t/60, az, label='opgemeten versnelling')
plt.legend(fontsize=lettergrootte)
plt.xlabel('tijd [min]', fontsize=lettergrootte)
plt.ylabel('afstand [cm], versnelling [m/s²]', fontsize=lettergrootte)
plt.title('laserdata overeenkomstig met versnellingsdata', fontsize=lettergrootte)
plt.show()

### laserafstand in functie van afgelegde afstand

dx_pol = [dx_corr[0]]
t_pol = [t[0]]
for i in range(len(dx_corr)-1):
    dx_extra = interpol(dx_corr[i], dx_corr[i+1], 320)
    t_extra = interpol(t[i], t[i+1], 320)
    dx_pol += list(dx_extra)
    t_pol += list(t_extra)
dx_pol = np.array(dx_pol)
t_pol = np.array(t_pol)

laser_afstand = datalaser[:-31999]

plt.figure(figsize=(16,9))
plt.plot(dx_pol, (laser_afstand[0]-laser_afstand)/10)
plt.title('wegprofiel in functie van de afstand', fontsize=lettergrootte)
plt.xlabel('afstand [m]', fontsize=lettergrootte)
plt.ylabel('gemeten afstand ten opzichte van wegdek [cm]', fontsize=lettergrootte)
plt.show()

flinearlaser = interp1d(dx_pol, laser_afstand)
fcubiclaser = interp1d(dx_pol, laser_afstand, kind='cubic')

dx_new = np.arange(1,dx_corr[-1],0.1) #Het laaste argument is de afgelegede afstand tussen samples dx

linearlaser = flinearlaser(dx_new)
cubiclaser = fcubiclaser(dx_new)

plt.figure(figsize=(16,9))
plt.plot(dx_new, linearlaser)
plt.show()