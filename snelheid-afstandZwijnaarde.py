import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def interpol(y1, y2, intervallen):
    t_interval = (np.arange(intervallen)+1)/intervallen
    return y1 + t_interval*(y2-y1)

accel = dict()     # dictionary voor de gemeten versnelling in x-richting en de tijd van de versnelling
GPS_snel = dict()    # dictionary voor de GPS snelheid, tijd en afstand
GPS_coord = dict()    # dictionary voor lengte- en breedtegraad

### Algemene verwerking data

for i in range(3, 4):
    name = 'xyz_Zwijnaarde' + str(i) + '.csv'
    contents = pd.read_csv(name)
    
    accel_x = contents[['a_x']].values.flatten()
    ax = 9.81*accel_x
    accel_z = contents[['a_z']].values.flatten()
    az = 9.81*accel_z
    
    bestandsnaam = 'D:\\UGent docs\\VOP_docs\\Data testrit Zwijnaarde\\158033857_' + str(i) + '.csv'    # plaats waar ik het bestand heb opgeslagen, pas dit zelf aan
    
    vx_GPS = pd.read_csv(bestandsnaam)[['GPSau-speed']].values.flatten()*5/18
    lengtegraad = pd.read_csv(bestandsnaam)[['GPSau-lon']].values.flatten()
    breedtegraad = pd.read_csv(bestandsnaam)[['GPSau-lat']].values.flatten()
    index = next(i for i,v in enumerate(lengtegraad) if v != 0) # seconden waar GPS niet meet wegdoen
    
    lengtegraad = lengtegraad[index:]
    breedtegraad = breedtegraad[index:]
    ax = ax[int(index*100):]
    az = az[int(index*100):]
    vx_GPS = vx_GPS[index:]
    
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
    
    accel['az_'+str(i)] = az
    accel['ax_'+str(i)] = ax
    accel['t_'+str(i)] = np.arange(len(ax))/100.0
    GPS_snel['vx_'+str(i)] = vx_GPS
    GPS_snel['t_'+str(i)] = np.arange(len(vx_GPS))
    GPS_coord['lon_'+str(i)] = lengtegraad
    GPS_coord['lat_'+str(i)] = breedtegraad
    

dt = 1/100 #breedte van 1 sample is 1/100 seconden

corrs = dict()    # dictionary voor gecorrigeerde versnelling, snelheid en afstand

for j in range(3, 4):   # enkel rit 3 wordt hier verwerkt, 4 en 5 zijn te kort en 1 en 2 zijn andere ritten
    vx_GPS = GPS_snel['vx_'+str(j)]
    ax = accel['ax_'+str(j)]
    
    vx0 = vx_GPS[0]

    corrections = [0.0]
    correcties = [0.0]

    for i in range(1, len(vx_GPS)):
        corr = float(vx_GPS[i]-vx_GPS[i-1]-np.sum(ax[(i-1)*100+1:i*100+1])*dt)
        b = float(vx_GPS[i]-vx_GPS[i-1]-np.sum(ax[(i-1)*100+1:i*100+1]+interpol(corrections[-1], corr, 100))*dt)
        correcties += list(interpol(corrections[-1], corr, 100)+b)

        corrections.append(corr)
        
#         correcties += list(interpol(correcties[-1], corr, 100))
    correcties = np.array(correcties)
    
    # foute metingen van de GPS tussen seconde 33 en 42, meting op seconde 36 lijkt wel correct. Ook foute meting op seconde 122
    if j == 3:
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
    
    corrs['ax_'+str(j)] = ax_corr
    corrs['vx_'+str(j)] = vx_corr
    corrs['dx_'+str(j)] = dx_corr
    GPS_snel['dx_'+str(j)] = dx_GPS
   

### plot GPS-positie
rit = 3
GPS_lon = GPS_coord['lon_'+str(rit)]
GPS_lat = GPS_coord['lat_'+str(rit)]
#BBox = (GPS_lon.min(), GPS_lon.max(), GPS_lat.min(), GPS_lat.max())
eps1 = 0.00011
eps2 = 0.00005
BBox = (3.70587+eps1, 3.71463+eps1, 51.00877+eps2, 51.01341+eps2)
print(BBox)

kaart = plt.imread('mapZwijnaarde.png')

fig, ax = plt.subplots(figsize = (16, 14))

lettergrootte = 20

ax.scatter(GPS_lon, GPS_lat, zorder=1, alpha= 1, c='b', s=10)
ax.scatter(GPS_lon[0], GPS_lat[0], zorder=1, c='r', s=20)
ax.scatter(GPS_lon[-1], GPS_lat[-1], zorder=1, c='lime', s=20)
ax.set_title('Afgelegde weg rit {}'.format(rit), fontsize=lettergrootte)
ax.set_xlabel('lengtegraad', fontsize=lettergrootte)
ax.set_ylabel('breedtegraad', fontsize=lettergrootte)
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

ax.imshow(kaart, zorder=0, extent = BBox, aspect= 'equal')

### Andere plots van de verwerkte data
vx_GPS = GPS_snel['vx_3']
t_GPS = GPS_snel['t_3']
vx_corr = corrs['vx_3']
dx_corr = corrs['dx_3']
dx_GPS = GPS_snel['dx_3']
t = accel['t_3']
ax_corr = corrs['ax_3']
az = accel['az_3']
ax = accel['ax_3']

# snelheidsprofiel rit
lettergrootte = 20
plt.figure(figsize=(16,9))
plt.title('Snelheid in functie van de tijd', fontsize=lettergrootte)
plt.plot(t_GPS/60, vx_GPS*18/5, label='gemeten snelheid met GPS')
plt.plot(t[:-99]/60, vx_corr*18/5, label='geïntegreerde versnelling na correctie')
plt.legend()
#plt.axis([0,np.max(t[:-100]/60)+0.5, 0, np.max(vx_corr*18/5)+5])
plt.xticks(fontsize=lettergrootte)
plt.yticks(fontsize=lettergrootte)
plt.xlabel('tijd [min]', fontsize=lettergrootte)
plt.ylabel('snelheid [km/h]', fontsize=lettergrootte)
plt.show()

# (deel van) gemeten GPS-snelheid vergelijken met geïntegreerde versnelling
slice1 = 31
slice2 = 46
vx_dom = np.cumsum(ax*dt)+vx_GPS[0]
lettergrootte = 20
plt.figure(figsize=(16,9))
plt.title('Snelheid in functie van de tijd', fontsize=lettergrootte)
plt.plot(t_GPS[slice1:slice2+1], vx_GPS[slice1:slice2+1]*18/5, label='gemeten snelheid met GPS')
plt.plot(t[slice1*100:slice2*100], vx_dom[slice1*100:slice2*100]*18/5, label='geïntegreerde versnelling')
plt.legend(fontsize=lettergrootte)
plt.xticks(fontsize=lettergrootte)
plt.yticks(fontsize=lettergrootte)
plt.xlabel('tijd [min]', fontsize=lettergrootte)
plt.ylabel('snelheid [km/h]', fontsize=lettergrootte)
plt.show()

# gecorrigeerde versnelling tov gemeten versnelling in eerste k seconden
k = 50
plt.figure(figsize=(9,6))
plt.plot(np.arange(len(ax[:k*100]))/100, ax[:k*100], label='gemeten versnelling')
plt.plot(np.arange(len(ax_corr[:k*100]))/100, ax_corr[:k*100], label='gecorrigeerde versnelling')
plt.legend()
plt.xlabel('tijd [s]')
plt.ylabel('versnelling [m/s²]')
plt.title('Versnelling na geïnterpoleerde correctie per seconde + constante waarde')
plt.show()

# GPS-snelheid en gecorrigeerde versnelling geïntegreerd
plt.plot(np.arange(len(vx_corr[:1000]))/100, vx_corr[:1000], label='geïntegreerde versnelling')
plt.plot(vx_GPS[:10], label='GPS-snelheid')
plt.title('Snelheid in de eerste 10 seconden')
plt.xlabel('tijd [s]')
plt.ylabel('snelheid [m/s]')
plt.show()

#afstandsprofiel
lettergrootte = 20
plt.figure(figsize=(8,6))
plt.title('Afstand in functie van de tijd', fontsize=lettergrootte)
plt.plot(t_GPS/60, dx_GPS/1000, label='afstand berekend met GPS data')
plt.plot(t[:-99]/60, dx_corr/1000, label='afstand berekend met versnelling na correctie met interpolatie')
#plt.legend(loc=2)
#plt.axis([0,np.max(t[:-100]/60)+0.5, 0, np.max(dx_corr/1000)+0.5])
plt.xticks(fontsize=lettergrootte)
plt.yticks(fontsize=lettergrootte)
plt.xlabel('tijd [min]', fontsize=lettergrootte)
plt.ylabel('afstand [km]', fontsize=lettergrootte)
plt.show()

# verticale versnelling ifv tijd
plt.figure(figsize=(16,9))
plt.plot(t/60, az)
plt.title('z-acceleratie in functie van de tijd')
plt.xlabel('tijd [min]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()

#verticale versnelling ifv afstand
plt.plot(dx_corr/1000, az[:-99])
plt.title('z-acceleratie in functie van de afstand')
plt.xlabel('afstand [km]')
plt.ylabel('z-acceleratie [m/s²]')
plt.show()
