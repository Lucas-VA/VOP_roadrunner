import numpy as np
import matplotlib.pyplot as plot
import itertools
import math
J1 = []
J2 = []
J3 = []
J4 = []
J5 = []
J = []
F1_tot = []
a = np.array(sinusoidal_speedbump_wbreak(10.0, 0.15, 5.0, 100))
y2 = []
#l1 = 1.2
#l2 = 1.6

#Simulatie functie definieren
def simuleer(param, weg, dt=0.005, vh=20):
    #Definieer constante autoparameters
    m = 1800
    #m1 = 120
    #m1 = 80
    #m2 = 110
    m1 = param[4]
    m2 = param[5]
    #m2 = 120
    #I = 3800
    #I = 3000
    I = param[6]
    l1 = param[7]
    #l1 = 1.2
    l2 = 2.7-l1
    #l1 = 1.2
    #l2 = 1.6
    k1 = param[0]
    k2 = param[1]
    c1 = param[2]
    c2 = param[3]
    #k1 = 74000
    #k2 = 74000
    #k1 = 70000
    #k2 = 53000
    #c1 = 3400
    #c1 = 6375
    #c2 = 2700
    #c2 = 3200
    #kt1 = 460000
    #kt2 = 460000
    kt1 = param[8]
    kt2 = param[9]

    #Definieer gezochte parameters in functie van tijd
    x = np.zeros(len(weg))
    x1 = np.zeros(len(weg))
    x2 = np.zeros(len(weg))
    theta = np.zeros(len(weg))
    v = np.zeros(len(weg))
    v1 = np.zeros(len(weg))
    v2 = np.zeros(len(weg))
    omega = np.zeros(len(weg))
    acc_z = np.zeros(len(weg))
    acc_hoek = np.zeros(len(weg))
    y1_som = 0
    y2_som = 0
    #Bepalen vorm weg
    dx = dt*vh
    xt1 = weg
    xt2 = np.append(np.zeros(int((l1+l2)/dx)), weg)[0:len(weg)]

    #Opstellen toestandsmodel
    A = np.matrix([[0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1],
                   [-(k1+k2)/m, k1/m, k2/m, (k2*l2-k1*l1)/m, -(c1+c2)/m, c1/m, c2/m, (c2*l2-c1*l1)/m],
                   [k1/m1, -(k1+kt1)/m1, 0, k1*l1/m1, c1/m1, -c1/m1, 0, c1*l1/m1], [k2/m2, 0, -(k2+kt2)/m2, -k2*l2/m2, c2/m2, 0, -c2/m2, -c2*l2/m2],
                   [(k2*l2-k1*l1)/I, k1*l1/I, -k2*l2/I, -(k1*l1**2+k2*l2**2)/I, (c2*l2-c1*l1)/I, c1*l1/I, -c2*l2/I, -(c1*l1**2+c2*l2**2)/I]])
    D = np.transpose(np.matrix([[0, 0, 0, 0, 0, kt1/m1, 0, 0], [0, 0, 0, 0, 0, 0, kt2/m2, 0]]))

    #Iteratief proces
    for i in range(len(weg)-1):
        C = A @ [x[i], x1[i], x2[i], theta[i], v[i], v1[i], v2[i], omega[i]] + D @ [xt1[i], xt2[i]]
        acc_z[i] = C[0, 4]
        acc_hoek[i] = C[0, 7]
        y1_som += abs((x1[i]+x1[i-1])/2)**2*dt
        y2_som += abs((x2[i]+x2[i-1])/2)**2*dt
        x[i + 1] = x[i] + dt*C[0, 0]
        x1[i + 1] = x1[i] + dt * C[0, 1]
        x2[i + 1] = x2[i] + dt * C[0, 2]
        theta[i + 1] = theta[i] + dt * C[0, 3]
        v[i + 1] = v[i] + dt * C[0, 4]
        v1[i + 1] = v1[i] + dt * C[0, 5]
        v2[i + 1] = v2[i] + dt * C[0, 6]
        omega[i + 1] = omega[i] + dt * C[0, 7]
        
    """
    #Plotten van resultaten
    plot.plot(dt*np.arange(len(x)), x, 'b', label='Trilling wagen')
    plot.plot(dt*np.arange(len(x)), xt1, 'r', label='Hoogte wegdek')
    plot.title('Trillingshoogte van de wagen bij bepaald wegdekprofiel')
    plot.xlabel('Tijd (s)')
    #plot.xlabel('Afstand (m)')
    plot.ylabel('Hoogte (m)')
    plot.legend()
    plot.show()

    plot.plot(dt*np.arange(len(theta)), theta, 'b', label='Trillingshoek wagen')
    plot.xlabel('Tijd (s)')
    # plot.xlabel('Afstand (m)')
    plot.ylabel('Hoek (rad)')
    plot.legend()
    plot.show()
    """
    """
    for i in range(len(x2)-math.floor((l1+l2)/dx)):
        y2.append(x2[i+math.floor((l1+l2)/dx)])
    for i in range(len(x2)-len(y2)):
        y2.append(0)
    """
    #return [max(acc_z), max(acc_hoek), y1_som, y2_som] #voor optimalisatie_1
    #return [dx, x1, x2]   #voor optimalisatie_2
    return [acc_z, acc_hoek, x1, x2, xt1]  #voor het plotten van resultaten

"""
#Optimalisatie adhv oppervlakten en maxima
def optimalisatie():
    k1 = np.linspace(0, 1000, num = 10)  #parameters hier werden telkens manueel aangepast
    k2 = np.linspace(0, 1000, num = 10)
    c1 = np.linspace(3000, 5000, num = 5)
    c2 = np.linspace(1, 1.5, num = 5)
    optim = list(itertools.product(k1, k2))
    for i in range(len(optim)):
        e = simuleer(optim[i], np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)
        J1.append(e[0])
        J2.append(e[1])
        J3.append(e[2])
        J4.append(e[3])
        J.append(J1[i] + J2[i] + J3[i] + J4[i])
    return optim[J.index(min(J))]
idi = optimalisatie()
"""
"""
#Optimalisatie adhv verschil tussen voorwiel en achterwiel
def optimalisatie_2():
    k1 = np.linspace(500000, 550000, num = 15)    #parameters hier werden telkens manueel aangepast
    k2 = np.linspace(450000, 500000, num = 15)
    c1 = np.linspace(2999, 3000, num = 5)
    c2 = np.linspace(1.1, 1.3, num = 5)
    F1 = 0
    optim = list(itertools.product(k1, k2))
    for i in range(len(optim)):
        dx, y1, y2 = simuleer(optim[i], a[1], dt=0.005)
        for i in range(math.floor((len(a[1])*dx-(l1+l2))/dx-0.5)):
            F1 += (y1[i]-y2[i+math.floor((l1+l2)/dx)])**2
        F1_tot.append(F1)
        F1 = 0
    return optim[F1_tot.index(min(F1_tot))]
A = optimalisatie_2()
"""
#e = simuleer([460000, 460000], np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)
#f = simuleer([521429, 471429], np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)
#vergelijken van resultaten
e = simuleer([74000, 74000, 3400, 3200, 120, 120, 3800, 1.2, 460000, 460000], np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)
f = simuleer([53000, 45200, 3600, 2950, 170, 200, 4800, 1.25, 310000, 390000], np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)

plot.plot(0.005*np.arange(len(f[0])), f[0], 'r', label='Geoptimaliseerde parameters')
plot.plot(0.005*np.arange(len(e[0])), e[0], 'b', label='Oorspronkelijke parameters')
plot.title('Z-acceleratie van de wagen')
plot.xlabel('Tijd (s)')
#plot.xlabel('Afstand (m)')
plot.ylabel('Acceleratie (m/s^2)')
plot.legend()
plot.show()
plot.plot(0.005*np.arange(len(f[0])), f[1], 'r', label='Geoptimaliseerde parameters')
plot.plot(0.005*np.arange(len(e[0])), e[1], 'b', label='Oorspronkelijke parameters')
plot.title('Hoek-acceleratie van de wagen')
plot.xlabel('Tijd (s)')
#plot.xlabel('Afstand (m)')
plot.ylabel('Acceleratie (m/s^2)')
plot.legend()
plot.show()
plot.plot(0.005*np.arange(len(e[0])), e[2], 'b', label='Oorspronkelijke parameters')
plot.plot(0.005*np.arange(len(f[0])), f[2], 'r', label='Geoptimaliseerde parameters')
plot.plot(0.005*np.arange(len(f[0])), f[4], 'g', label='Hoogte wegdek')
plot.title('Uitwijking voorwiel')
plot.xlabel('Tijd (s)')
#plot.xlabel('Afstand (m)')
plot.ylabel('Verplaatsing (m)')
plot.legend()
plot.show()
#simuleer(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(1000)), dt=0.005)
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(1000)), dt=0.005))
#print(simuleer(np.append(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005))
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), -0.01*np.arange(500)), dt=0.005))
#print(simuleer(np.append(np.append(np.array([0, 0, 0, 0, 0]), -0.001*np.arange(200)), 0.001*np.arange(200)-0.2), dt=0.005))
#--> v horizontaal kan tijdsafhankelijk zijn. Hier constant.
