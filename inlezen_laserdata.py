import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

laserdata_flat = []

for k in range(104):
    #print(k)
    bestandsnaam = 'D:\\UGent docs\\VOP_docs\\laserdata\\' + str(k) + '.txt'    # plaats waar ik bestanden heb opgeslagen, pas dit zelf aan
    
    data = pd.read_csv(bestandsnaam,
                       sep=" ",
                       header=None, 
                       delimiter = "\t")
    
    data = data.values

    ### Measuring rate = 32kHz = 32000 metingen per seconde

    for i in range(data.shape[0]):
        rij = data[i, 3].split(",")
        
        for e in rij:
            if e == '':
                laserdata_flat.append(laserdata_flat[-1])
            else:
                laserdata_flat.append(float(e))

laserdata_flat = np.array(laserdata_flat)

time = np.arange(len(laserdata_flat))/32000     # gebruik sample frequentie als maat voor de tijd, de tijd in de tekstbestanden komt van de Raspberry Pi en is niet altijd correct

# pd.DataFrame({'time': time[int(0.08*10**8):int(0.27*10**8)], 'afstand_laser': laserdata_flat[int(0.08*10**8):int(0.27*10**8)]}).to_csv('laser_rit1.csv')  # maakt csv bestand aan van de eerste laserrit
# pd.DataFrame({'time': time[int(0.34*10**8):int(0.58*10**8)], 'afstand_laser': laserdata_flat[int(0.34*10**8):int(0.58*10**8)]}).to_csv('laser_rit2.csv')  #csv bestand voor rit 2 en 3, niet zinvol want daar is te weinig GPS data voor
# pd.DataFrame({'time': time[int(0.73*10**8):], 'afstand_laser': laserdata_flat[int(0.73*10**8):]}).to_csv('laser_rit3.csv')

### Aapassing sample hertz
nieuwe_freq = 100
nyquist_freq = 16000 #32000Hz/2
cutoff_freq = 40 # moet kleiner zijn dan de helft van nieuwe sample frequentie

b, a = butter(2, cutoff_freq/nyquist_freq, btype='low', analog=False)
laserdata_lowpass = filtfilt(b, a, laserdata_flat)

laserdata_kort = np.zeros(int(len(laserdata_lowpass)*nieuwe_freq/32000))
tijd_kort = np.zeros(int(len(laserdata_lowpass)*nieuwe_freq/32000))
for i in range(int(len(laserdata_lowpass)*nieuwe_freq/32000)):
    laserdata_kort[i] = laserdata_lowpass[i*32000/nieuwe_freq]
    tijd_kort[i] = time[i*32000/nieuwe_freq]

# pd.DataFrame({'time': tijd_kort[int(0.08/320*10**8):int(0.27/320*10**8)], 'afstand_laser': laserdata_kort[int(0.08/320*10**8):int(0.27/320*10**8)]}).to_csv('laser_rit1_kort.csv')
# pd.DataFrame({'time': tijd_kort[int(0.34/320*10**8):int(0.58/320*10**8)], 'afstand_laser': laserdata_kort[int(0.34/320*10**8):int(0.58/320*10**8)]}).to_csv('laser_rit2_kort.csv')
# pd.DataFrame({'time': tijd_kort[int(0.73/320*10**8):], 'afstand_laser': laserdata_kort[int(0.73/320*10**8):]}).to_csv('laser_rit3_kort.csv')
