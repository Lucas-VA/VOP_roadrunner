import numpy as np
import matplotlib.pyplot as plot
import math
m = 1800
m1 = 120
m2 = 120
I = 3800
l1 = 1.2
l2 = 1.6
k1 = 74000
k2 = 74000
c1 = 3400
c2 = 3200
kt1 = 460000
kt2 = 460000
h_laser_i = 0.225
lcg = l1 + 0.5
gamma = 0.8
beta = 0.25*(0.5+gamma)**2
acc_z = [0.0]
vers_hoek = [0.0]
x_initieel = []
v_initieel = []
alpha_initieel = []
theta_initieel = []
Mv = np.matrix([[m, 0, 0, 0], [0, I, 0, 0], [0, 0, m1, 0], [0, 0, 0, m2]])
Cv = np.matrix([[c1+c2, l1*c1 - l2*c2, -c1, -c2], [l1*c1 - l2*c2, l1**2*c1+l2**2*c2, -l1*c1, l2*c2], [-c1, -l1*c1, c1, 0], [-c2, l2*c2, 0, c2]])
Kv = np.matrix([[k1+k2, l1*k1 - l2*k2, -k1, -k2], [l1*k1 - l2*k2, l1**2*k1+l2**2*k2, -l1*k1, l2*k2], [-k1, -l1*k1, k1+kt1, 0], [-k2, l2*k2, 0, k2+kt2]])
a = np.array(trapezoid_speedbump(10.0, 2.5, 0.15, Hz))
#d = np.append(np.append(np.zeros(2000).T,a[1]), np.zeros(2000).T)
K1_opt = []



#Simulatie functie definieren
def simuleer(weg, dt=1/200, vh=20):
    #Definieer constante autoparameters
    

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
        if i == 0:
            alpha_initieel.append(C[0, 7])
        acc_z.append(C[0, 4])
        vers_hoek.append(C[0, 3])
        x[i + 1] = x[i] + dt*C[0, 0]
        x1[i + 1] = x1[i] + dt * C[0, 1]
        x2[i + 1] = x2[i] + dt * C[0, 2]
        theta[i + 1] = theta[i] + dt * C[0, 3]
        v[i + 1] = v[i] + dt * C[0, 4]
        v1[i + 1] = v1[i] + dt * C[0, 5]
        v2[i + 1] = v2[i] + dt * C[0, 6]
        omega[i + 1] = omega[i] + dt * C[0, 7]
    x_initieel.append(x[0])
    v_initieel.append(v[0])
    theta_initieel.append(theta[0])
    
    #Plotten van resultaten
    plot.plot(dx*np.arange(len(x)), x, 'b', label='Trilling wagen')
    plot.plot(dx*np.arange(len(x)), xt1, 'r', label='Hoogte wegdek')
    plot.title('Trillingshoogte van de wagen bij bepaald wegdekprofiel')
    plot.xlabel('Tijd (s)')
    plot.xlabel('Afstand (m)')
    plot.ylabel('Hoogte (m)')
    plot.legend()
    plot.show()

    plot.plot(dx*np.arange(len(theta)), theta, 'b', label='Trillingshoek wagen')
    plot.xlabel('Tijd (s)')
    plot.xlabel('Afstand (m)')
    plot.ylabel('Hoek (rad)')
    plot.legend()
    plot.show()
    return x, theta, xt1
lijst_1 = np.array(np.zeros(150).tolist())
lijst_2 = np.array(np.ones(150).tolist())/50
c = simuleer(a[1], dt = 0.005)
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), 0.1*np.ones(1000)), dt=0.005))
#a = simuleer(np.append(np.append(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), 0.1*np.ones(500)), np.zeros(750)), dt=0.005)
#print(simuleer(np.append(np.array([0, 0, 0, 0, 0]), -0.01*np.arange(500)), dt=0.005))
#print(simuleer(np.append(np.append(np.array([0, 0, 0, 0, 0]), -0.001*np.arange(200)), 0.001*np.arange(200)-0.2), dt=0.005))

#--> v horizontaal kan tijdsafhankelijk zijn. Hier constant.


#Newmark beta procedure
def inverted_model(dt = 1/200):
    x = np.zeros(len(acc_z))
    x1 = np.zeros(len(acc_z))
    x2 = np.zeros(len(acc_z))
    theta = np.zeros(len(acc_z))
    v = np.zeros(len(acc_z))
    v1 = np.zeros(len(acc_z))
    v2 = np.zeros(len(acc_z))
    eff_force = np.zeros(len(acc_z)).tolist()
    force = np.zeros(len(acc_z)).tolist()
    Uv = np.zeros(len(acc_z)).tolist()
    Uv_snel = np.zeros(len(acc_z)).tolist()
    Uv_vers = np.zeros(len(acc_z)).tolist()
    alpha = np.zeros(len(acc_z))
    acc_z1 = np.zeros(len(acc_z))
    acc_z2 = np.zeros(len(acc_z))
    y1 = np.zeros(len(acc_z))
    y2 = np.zeros(len(acc_z))
    a0 = 1/(beta * dt**2)
    a1 = gamma/(dt*beta)
    a2 = 1/(beta*dt)
    a3 = 1/(2*beta)-1
    a4 = gamma/beta-1
    a5 = dt/2*(gamma/beta-2)
    a6 = (1-gamma)*dt
    a7 = gamma*dt
    x[0] = x_initieel[0]
    v[0] = v_initieel[0]
    alpha[0] = alpha_initieel[0]
    theta[0] = theta_initieel[0]
    K = Kv + a0*Mv + a1*Cv
    Uv[0] = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    Uv_snel[0] = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    Uv_vers[0] = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    force[0] = np.matrix([[0.0], [0.0], [0.0], [0.0]])
    
    for i in range(len(acc_z)-1):
        x[i + 1] = (acc_z[i+1] + a2*v[i] + a3*acc_z[i])/a0 + x[i]
        v[i + 1] = v[i] + a6*acc_z[i] + a7*acc_z[i + 1]
        theta[i + 1] = (vers_hoek[i+1] + a4*vers_hoek[i] + a5*alpha[i])/a1 + theta[i]
        alpha[i + 1] = a0*(theta[i + 1] - theta[i]) - a2*vers_hoek[i] - a3*alpha[i]
        x1[i + 1] = (l2*m*acc_z[i+1] + I*alpha[i+1] + (l2*c1 + l1*c1)*v[i+1] + (l2*l1*c1 + l1**2*c1)*vers_hoek[i+1] + (l2*k1 + l1*k1)*x[i+1] + (l2*l1*k1 + l1**2*k1)*theta[i+1] + (l2*c1+l1*c1)*(a1*x1[i] + a4*v1[i] + a5*acc_z1[i]))/(l2*k1 + l1*k1 + (l2*c1 + l1*c1)*a1)
        acc_z1[i + 1] = a0*(x1[i+1] - x1[i]) - a2*v1[i] - a3*acc_z1[i]
        v1[i + 1] = v1[i] + a6*acc_z1[i] + a7*acc_z1[i+1]
        x1[i + 1] = (l2*m*acc_z[i+1] + I*alpha[i+1] + (l2*c1 + l1*c1)*v[i+1] + (l2*l1*c1 + l1**2*c1)*vers_hoek[i+1] + (l2*k1 + l1*k1)*x[i+1] + (l2*l1*k1 + l1**2*k1)*theta[i+1] + (l2*c1+l1*c1)*(a1*x1[i] + a4*v1[i] + a5*acc_z1[i]))/(l2*k1 + l1*k1 + (l2*c1 + l1*c1)*a1)
        acc_z1[i + 1] = a0*(x1[i+1] - x1[i]) - a2*v1[i] - a3*acc_z1[i]
        v1[i + 1] = v1[i] + a6*acc_z1[i] + a7*acc_z1[i+1]
        x2[i + 1] = (l1*m*acc_z[i+1] - I*alpha[i+1] + (l1*c2 + l2*c2)*v[i+1] - (l2*l1*c2 + l2**2*c2)*vers_hoek[i+1] + (l1*k2 + l2*k2)*x[i+1] - (l2*l1*k2 + l2**2*k2)*theta[i+1] + (l1*c2+l2*c2)*(a1*x2[i] + a4*v2[i] + a5*acc_z2[i]))/(l1*k2 + l2*k2 + (l1*c2 + l2*c2)*a1)
        acc_z2[i + 1] = a0*(x2[i+1] - x2[i]) - a2*v2[i] - a3*acc_z2[i]
        v2[i + 1] = v2[i] + a6*acc_z2[i] + a7*acc_z2[i+1]
        Uv[i + 1] = np.matrix([[x[i+1]], [theta[i+1]], [x1[i+1]], [x2[i+1]]])
        Uv_snel[i + 1] = np.matrix([[v[i+1]], [vers_hoek[i+1]], [v1[i+1]], [v2[i+1]]])
        Uv_vers[i + 1] = np.matrix([[acc_z[i+1]], [alpha[i+1]], [acc_z1[i+1]], [acc_z2[i+1]]])
        eff_force[i + 1] = np.dot(K, Uv[i+1])
        force[i + 1] = eff_force[i+1] - np.dot(Mv, a0*Uv[i]) -np.dot(Mv, a2*Uv_snel[i]) - np.dot(Mv, a3*Uv_vers[i]) - np.dot(Cv, a1*Uv[i]) -np.dot(Cv, a4*Uv_snel[i]) - np.dot(Cv, a5*Uv_vers[i])
        y1[i + 1] = force[i+1][2]/kt1
        y2[i + 1] = force[i+1][3]/kt2
    return y1

"""
#omzetten naar metingen van de laser
def laser_meting(x, theta, h):
    laser = []
    for i in range(len(x)):
        laser.append(x[i] + math.sin(theta[i])*lcg + h_laser_i*math.cos(theta[i]))
    return laser
p = laser_meting(a[0], a[1], a[2])
"""
b = inverted_model()
RMS = np.sum(np.sqrt(abs(b-c[2])**2)/len(b))
print(RMS)
plot.plot(1/500*np.arange(len(c[2])), c[2], 'b', label='Hoogte wegdek')
plot.plot(1/500*np.arange(len(c[2])), b, 'r', label='recon')
plot.xlabel('Afstand (m)')
plot.ylabel('Hoogte (m)')
plot.legend()
plot.show()