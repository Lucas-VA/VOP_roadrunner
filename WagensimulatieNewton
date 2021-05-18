import numpy as np
import matplotlib.pyplot as plot

#Simulatie functie definieren
def simuleer(weg, dt=0.005, vh=np.array([None])):
    vh = vh[:len(weg)]
    if vh.any()==None:
        vh = 20*np.ones(len(weg))
    #Definieer constante autoparameters
    m = 1800
    m1 = 170
    m2 = 200
    I = 4800
    l1 = 1.25
    l2 = 1.50
    d = 1.1
    k1 = 53000
    k2 = 45200
    c1 = 3600
    c2 = 2950
    kt1 = 310000
    kt2 = 390000
    h = 0.225

    #Definieer gezochte parameters in functie van tijd
    x = np.zeros(len(weg))
    x1 = np.zeros(len(weg))
    x2 = np.zeros(len(weg))
    theta = np.zeros(len(weg))
    v = np.zeros(len(weg))
    v1 = np.zeros(len(weg))
    v2 = np.zeros(len(weg))
    omega = np.zeros(len(weg))

    #Bepalen vorm weg
    dx = dt*vh
    xt1 = weg
    xt2 = np.zeros(len(xt1))
    xtd = np.zeros(len(xt2))
    for i, e in enumerate(dx):
        if i-int(((l1+l2)/e)) >= 0:
            xt2[i] = xt1[i-int((l1+l2)/e)]
        else:
            xt2[i] = 0
        if i-int(((l1+l2+d)/e)) >= 0:
            xtd[i] = xt1[i-int((l1+l2+d)/e)]
        else:
            xtd[i] = 0

    #Opstellen toestandsmodel
    A = np.matrix([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                   [-(k1+k2)/m, k1/m, k2/m, (k2*l2-k1*l1)/m, -(c1+c2)/m, c1/m, c2/m, (c2*l2-c1*l1)/m],
                   [k1/m1, -(k1+kt1)/m1, 0, k1*l1/m1, c1/m1, -c1/m1, 0, c1*l1/m1], [k2/m2, 0, -(k2+kt2)/m2, -k2*l2/m2, c2/m2, 0, -c2/m2, -c2*l2/m2],
                   [(k2*l2-k1*l1)/I, k1*l1/I, -k2*l2/I, -(k1*l1**2+k2*l2**2)/I, (c2*l2-c1*l1)/I, c1*l1/I, -c2*l2/I, -(c1*l1**2+c2*l2**2)/I]])
    D = np.transpose(np.matrix([[0, 0, 0, 0, 0, kt1/m1, 0, 0], [0, 0, 0, 0, 0, 0, kt2/m2, 0]]))

    #Iteratief proces
    for i in range(len(weg)-1):
        C = A @ [x[i], x1[i], x2[i], theta[i], v[i], v1[i], v2[i], omega[i]] + D @ [xt1[i], xt2[i]]
        x[i + 1] = x[i] + dt*C[0, 0]
        x1[i + 1] = x1[i] + dt * C[0, 1]
        x2[i + 1] = x2[i] + dt * C[0, 2]
        theta[i + 1] = theta[i] + dt * C[0, 3]
        v[i + 1] = v[i] + dt * C[0, 4]
        v1[i + 1] = v1[i] + dt * C[0, 5]
        v2[i + 1] = v2[i] + dt * C[0, 6]
        omega[i + 1] = omega[i] + dt * C[0, 7]

    #Plotten van resultaten

    plot.plot(dt*np.arange(len(x)), xt1, 'r', label='Hoogte wegdek')
    plot.plot(dt*np.arange(len(x)), x, 'b', label='Trilling wagen')
    plot.plot(dt*np.arange(len(x)), x-(l2+d)*theta, 'g', label='Trilling laser')
    plot.plot(dt*np.arange(len(x)), x-(l2+d)*theta-xtd, 'y', label='Hoogtemeting laser')
    plot.title('Trillingshoogte van de wagen bij bepaald wegdekprofiel (Newton)')
    plot.xlabel('Tijd (s)')
    plot.ylabel('Hoogte (m)')
    plot.legend()
    plot.show()

    #plot.plot(dt*np.arange(len(x)), x+(l2+d)*theta-xtd, 'y', label='Hoogtemeting laser')
    #plot.plot(dt*np.arange(len(x)), np.append(np.zeros(int((l1+l2+d)/dx)), xt1)[:len(xt1)], 'r', label='Hoogte wegdek')
    #plot.title('Trillingshoogte van de wagen bij bepaald wegdekprofiel')
    #plot.xlabel('Tijd (s)')
    #plot.xlabel('Afstand (m)')
    #plot.ylabel('Hoogte (m)')
    #plot.legend()
    #plot.show()

    #plot.plot(dt*np.arange(len(theta)), theta, 'b', label='Trillingshoek wagen')
    #plot.title('Trillingshoek van de wagen bij bepaald wegdekprofiel')
    #plot.xlabel('Tijd (s)')
    #plot.xlabel('Afstand (m)')
    #plot.ylabel('Hoek (rad)')
    #plot.legend()
    #plot.show()

    #fl = np.fft.fft(x-weg)
    #freq = np.fft.fftfreq((x-weg).size, d=dt)
    #plot.plot(freq, abs(fl))
    #plot.show()
    return x-(l2+d)*theta-xtd


#TEST WEGPROFIELEN

#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(1000)), dt=0.005))
#print(simuleer(np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005))
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), -0.01*np.arange(500)), dt=0.005))
#print(simuleer(np.append(np.vectorize(lambda t: -0.05*np.sin(t/50))(np.arange(157)), np.zeros(500))))
#print(simuleer(np.append(np.append(np.array([0, 0, 0, 0, 0]), -0.001*np.arange(200)), 0.001*np.arange(500)-0.2), dt=0.005))
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), 0.01*np.sin(np.arange(1500)/25))))
#print(simuleer(np.append(np.append(np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.02*np.sin(np.arange(100)/64)), 0.01/50*np.arange(50)+0.02), 0.03*np.cos(np.arange(200)/42)), np.zeros(100))))

#print(simuleer(np.append(np.append(np.zeros(100), 0.1*np.ones(500)), np.zeros(1000))))#, vh=30+20*np.sin(np.arange(5000)/250)))
#print(simuleer(np.append(np.vectorize(lambda t: -0.05*np.sin(t/50))(np.arange(157)), np.zeros(750)), vh=30+20*np.cos(np.arange(5000)/250)))
#print(simuleer(np.append(np.append(np.append(np.array([0, 0, 0, 0, 0]), -0.001*np.arange(200)), 0.001*np.arange(500)-0.2), 0.3*np.ones(1000)), vh=30+20*np.sin(np.arange(5000)/250)))
#print(simuleer(np.append(np.append(np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.02*np.sin(np.arange(100)/64)), 0.01/50*np.arange(50)+0.02), 0.03*np.cos(np.arange(200)/42)), np.zeros(1000)), vh=30+20*np.sin(np.arange(5000)/250)))
