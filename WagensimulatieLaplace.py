import numpy as np
import matplotlib.pyplot as plt
import time

#Afwijking: imulatie funtie definieren
def simuleerSnelheidAfw(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1) / (A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    plt.plot(t, u, 'r', label='Hoogte wegdek')
    plt.plot(t, y, 'b', label='Trilling wagen')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y


def invsimuleerSnelheidAfw(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2) / (A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    plt.plot(t, u, 'g', label='Trilling wagen')
    plt.plot(t, y, 'y', label='Hoogte wegdek')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y



#Hoek: Simulatie funtie definieren
def simuleerSnelheidHoek(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

    # Definieer constante autoparameters
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

    # Defineer TransferFunctie
    A1 = m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u))
    B1 = (((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2
    C1 = (l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2))
    D1 = ((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 = (2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 = (2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 = 120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2
    H1 = 120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2
    I1 = np.zeros(len(u))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m + m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)* (c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1) / (A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    plt.plot(t, y, 'b', label='Hoek wagen')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y


def invsimuleerSnelheidHoek(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

    # Definieer constante autoparameters
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

    # Defineer TransferFunctie
    A1 = m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u))
    B1 = (((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2
    C1 = (l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2))
    D1 = ((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 = (2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 = (2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 = 120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2
    H1 = 120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2
    I1 = np.zeros(len(u))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m + m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)* (c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2) / (A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    plt.plot(t, y-y[0], 'b', label='Hoogte wegdek')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y-y[0]



#Laser: imulatie funtie definieren
def simuleerSnelheidLaser(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A1 -= (l2+d)*(m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u)))
    B1 -= (l2+d)*((((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2)
    C1 -= (l2+d)*((l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2)))
    D1 = D1 - (l2+d)*((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 -= (l2+d)*(2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 -= (l2+d)*(2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 -= (l2+d)*(120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2)
    H1 -= (l2+d)*(120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2)
    I1 -= (l2+d)*(np.zeros(len(u)))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1) / (A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    #plt.plot(t, u, 'r', label='Hoogte wegdek')
    #plt.plot(t, y, 'g', label='Trilling Laser')
    #plt.xlabel("Tijd (s)")
    #plt.ylabel("Amplitude")
    #plt.legend()
    #plt.show()
    return y


def invsimuleerSnelheidLaser(u, dt=0.005, vh=np.array([None]), N=3000):
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A1 -= (l2+d)*(m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u)))
    B1 -= (l2+d)*((((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2)
    C1 -= (l2+d)*((l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2)))
    D1 = D1 - (l2+d)*((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 -= (l2+d)*(2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 -= (l2+d)*(2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 -= (l2+d)*(120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2)
    H1 -= (l2+d)*(120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2)
    I1 -= (l2+d)*(np.zeros(len(u)))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)
    #A1 = np.append(np.flip(A1), A1), B1 = np.append(np.flip(B1), B1), C1 = np.append(np.flip(C1), C1), D1 = np.append(np.flip(D1), D1), E1 = np.append(np.flip(E1), E1), F1 = np.append(np.flip(F1), F1), G1 = np.append(np.flip(G1), G1), H1 = np.append(np.flip(H1), H1), I1 = np.append(np.flip(I1), I1), A2 = np.append(np.flip(A2), A2), B2 = np.append(np.flip(B2), B2), C2 = np.append(np.flip(C2), C2), D2 = np.append(np.flip(D2), D2), E2 = np.append(np.flip(E2), E2), F2 = np.append(np.flip(F2), F2), G2 = np.append(np.flip(G2), G2), H2 = np.append(np.flip(H2), H2), I2 = np.append(np.flip(I2), I2), J2 = np.append(np.flip(J2), J2), K2 = np.append(np.flip(K2), K2), L2 = np.append(np.flip(L2), L2)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) (+ kort array in, want inverse laplace grote foutenmarge)
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2) / (A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)
    y = ny(t)

    # Plotten!!!
    plt.plot(t, u, 'y', label='Trilling Laser')
    plt.plot(t, y, 'b', label='Hoogte wegdek')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y



#Hoogte: imulatie funtie definieren
def simuleerSnelheidHoogte(u, dt=0.005, vh=np.array([None]), N=3000, autocor = False):
    #Maak u langer voor betere accuraatheid
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A1 -= (l2+d)*(m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u)))
    B1 -= (l2+d)*((((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2)
    C1 -= (l2+d)*((l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2)))
    D1 = D1 - (l2+d)*((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 -= (l2+d)*(2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 -= (l2+d)*(2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 -= (l2+d)*(120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2)
    H1 -= (l2+d)*(120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2)
    I1 -= (l2+d)*(np.zeros(len(u)))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) + kort u terug in tot originele lengte
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2*(A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1) / (A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2) - 2*np.exp(-(l1+l2+d)*l/vh)
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)[:len(u)]
    y = ny(t)

    # Plotten!!!
    plt.plot(t, u, 'r', label='Hoogte wegdek')
    plt.plot(t, y*(np.sqrt(-np.arange(len(u))+len(u))/np.sqrt(len(u))*autocor+(1-autocor)), 'y', label='Gemeten Hoogte')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Hoogte (m)")
    plt.title('Trillingshoogte van de wagen bij bepaald wegdekprofiel')
    plt.legend()
    plt.show()
    return y*(np.sqrt(-np.arange(len(u))+len(u))/np.sqrt(len(u))*autocor+(1-autocor))


def invsimuleerSnelheidHoogte(u, dt=0.005, vh=np.array([None]), N=3000, yf=0):
    #Maak u langer voor betere accuraatheid
    if vh.any()==None:
        vh = 20*np.ones(len(u))
    vh = np.append(vh[:len(u)], np.mean(vh) * np.ones(N))
    u = np.append(u, u[-1]*np.ones(N))

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

    #Defineer TransferFunctie
    A1 = -I*(l1+l2)**3*(c2*kt2*m1-c1*kt1*m2)*np.ones(len(u))
    B1 = (l1+l2)**2*(-c2*kt2*m1*c1*l1**3+c2*l2*c1*(kt1*m2-2*m1*kt2)*l1**2+(2*c1*(kt1*m2-1/2*kt2*m1)*c2*l2**2+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I)*l1+c2*kt1*l2**3*m2*c1+((c2*c1+m2*k1)*kt1-kt2*(c2*c1+k2*m1))*I*l2+12*I*vh*(kt1*m2*c1+c2*m1*kt2))
    C1 = (l1+l2)*(-kt2*m1*(k1*c2+k2*c1)*l1**4+((kt1*m2-3*m1*kt2)*(c1*k2+c2*k1)*l2+12*c1*c2*m1*vh*kt2)*l1**3+(3*(kt1*m2-m1*kt2)*(k1*c2+c1*k2)*l2**2+12*c1*c2*vh*(kt1*m2+2*m1*kt2)*l2+(((c1-c2)*kt2+c2*k1+c1*k2)*kt1-kt2*(c1*k2+c2*k1))*I)*l1**2+((c1*k2+c2*k1)*(3*kt1*m2-m1*kt2)*l2**3+12*(2*kt1*m2+m1*kt2)*c1*c2*vh*l2**2+2*(((c1-c2)*kt2+k1*c2+k2*c1)*kt1-kt2*(k1*c2+c1*k2))*I*l2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+m1*k2))*vh*I)*l1+kt1*m2*(c1*k2+c2*k1)*l2**4+12*c1*c2*kt1*vh*m2*l2**3+(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+k1*c2))*I*l2**2+12*((c1*c2+m2*k1)*kt1+kt2*(c1*c2+k2*m1))*vh*I*l2-60*I*vh**2*(-kt1*m2*c1+c2*m1*kt2))
    D1 = (-kt2*(kt1*c1*c2+k1*k2*m1)*l1**5+(((-3*kt2*c1*c2+k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2+12*kt2*m1*vh*(c1*k2+c2*k1))*l1**4+(((-2*kt2*c1*c2+4*k1*k2*m2)*kt1-6*k1*k2*kt2*m1)*l2**2+12*vh*(kt1*m2+3*kt2*m1)*(c1*k2+c2*k1)*l2+I*((k1-k2)*kt2+k1*k2)*kt1-kt2*(60*c1*c2*m1*vh**2))*l1**3+(((2*kt2*c1*c2+6*k1*k2*m2)*kt1-4*k1*k2*kt2*m1)*l2**3+36*vh*(kt2*m1+kt1*m2)*(c1*k2+c2*k1)*l2**2+((3*I*(k1-k2)*kt2+60*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(40*c1*c2*m1*vh**2+I*k1*k2))*l2+12*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+k1*c2))*I)*l1**2+(((3*kt2*c1*c2+4*k1*k2*m2)*kt1-k1*k2*kt2*m1)*l2**4+12*(3*kt1*m2+m1*kt2)*(c1*k2+c2*k1)*vh*l2**3+((3*I*(k1-k2)*kt2+120*c1*c2*m2*vh**2+3*I*k1*k2)*kt1-3*kt2*(20*c1*c2*m1*vh**2+I*k1*k2))*l2**2+24*vh*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I*l2+60*((c1*c2+m2*k1)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I)*l1+kt1*(kt2*c1*c2+k1*k2*m2)*l2**5+12*kt1*m2*vh*(c1*k2+k1*c2)*l2**4+((I*(k1-k2)*kt2+60*c1*c2*m2*vh**2)*kt1-I*k1*k2*kt2)*l2**3+12*vh*(((c1+c2)*kt2+k1*c2+k2*c1)*kt1+kt2*(c1*k2+k1*c2))*I*l2**2+60*((c1*c2+k1*m2)*kt1-kt2*(c1*c2+k2*m1))*vh**2*I*l2+120*I*vh**3*(kt1*c1*m2+c2*kt2*m1))
    E1 = (-kt1*kt2*(c1*k2+c2*k1)*l1**5-3*kt2*(kt1*(c1*k2+k1*c2)*l2-4*vh*(kt1*c1*c2+k1*k2*m1))*l1**4+(-2*kt1*kt2*(c1*k2+c2*k1)*l2**2+12*vh*((4*kt2*c1*c2+k1*k2*m2)*kt1+3*k1*k2*kt2*m1)*l2-60*vh**2*m1*kt2*(c1*k2+c2*k1))*l1**3+(2*kt1*kt2*(c1*k2+c2*k1)*l2**3+36*((2*kt2*c1*c2+k1*k2*m2)*kt1+k1*k2*kt2*m1)*vh*l2**2+60*vh**2*(kt1*m2-2*m1*kt2)*(c1*k2+c2*k1)*l2+12*(((k1+k2)*kt2+k1*k2)*I*kt1+kt2*(10*c1*c2*m1*vh**2+I*k1*k2))*vh)*l1**2+(3*kt1*kt2*(c1*k2+c2*k1)*l2**4+12*vh*((4*kt2*c1*c2+3*m2*k1*k2)*kt1+k1*k2*m1*kt2)*l2**3+60*(c1*k2+c2*k1)*(2*m2*kt1-m1*kt2)*vh**2*l2**2+24*((I*(k1+k2)*kt2+5*c1*c2*m2*vh**2+I*k1*k2)*kt1+kt2*(5*c1*c2*m1*vh**2+I*k1*k2))*vh*l2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I)*l1+kt1*kt2*(k1*c2+k2*c1)*l2**5+12*kt1*vh*(kt2*c1*c2+m2*k1*k2)*l2**4+60*kt1*m2*vh**2*(c1*k2+c2*k1)*l2**3+12*((I*(k1+k2)*kt2+10*c1*c2*m2*vh**2+I*k1*k2)*kt1+I*k1*k2*kt2)*vh*l2**2+60*(((c1-c2)*kt2+c1*k2+c2*k1)*kt1-kt2*(c1*k2+c2*k1))*vh**2*I*l2+120*((c1*c2+k1*m2)*kt1+kt2*(c1*c2+k2*m1))*vh**3*I)
    F1 = (-k1*k2*kt1*kt2*l1**5+3*kt1*kt2*(4*vh*(c1*k2+c2*k1)-k1*k2*l2)*l1**4+2*kt2*(-k1*k2*kt1*l2**2+24*kt1*vh*(c1*k2+k1*c2)*l2-30*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**3+(k1*k2*kt1*kt2*l2**3+72*kt1*kt2*vh*(c1*k2+c2*k1)*l2**2-60*((kt2*c1*c2-m2*k1*k2)*kt1+2*k1*k2*kt2*m1)*vh**2*l2+120*kt2*m1*vh**3*(c1*k2+c2*k1))*l1**2+(3*k1*k2*kt1*kt2*l2**4+48*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+60*((kt2*c1*c2+2*m2*k1*k2)*kt1-k1*k2*kt2*m1)*vh**2*l2**2+120*vh**3*(kt2*m1+m2*kt1)*(c1*k2+c2*k1)*l2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I)*l1+k1*k2*kt1*kt2*l2**5+12*kt1*kt2*vh*(c1*k2+k1*c2)*l2**4+60*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**3+120*kt1*m2*vh**3*(c1*k2+c2*k1)*l2**2+60*(((k1-k2)*kt2+k1*k2)*kt1-k1*k2*kt2)*vh**2*I*l2+120*vh**3*(((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*I)
    G1 = 12*(k1*k2*kt1*kt2*l1**4-kt2*(5*vh*(c1*k2+c2*k1)-4*k1*k2*l2)*kt1*l1**3-kt2*(5*kt1*vh*(c1*k2+c2*k1)*l2-6*k1*k2*kt1*l2**2-10*vh**2*(kt1*c1*c2+m1*k1*k2))*l1**2+(4*k1*k2*kt1*kt2*l2**2+5*kt1*kt2*vh*(c1*k2+k1*c2)*l2+10*((2*kt2*c1*c2+m2*k1*k2)*kt1+k1*k2*kt2*m1)*vh**2)*l2*l1+k1*k2*kt1*kt2*l2**4+5*kt1*kt2*vh*(c1*k2+c2*k1)*l2**3+10*kt1*vh**2*(kt2*c1*c2+m2*k1*k2)*l2**2+10*vh**2*(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*vh
    H1 = 60*(l1+l2)**2*kt1*kt2*vh**2*(k1*k2*(l2-l1)+2*vh*(c1*k2+k1*c2))
    I1 = 120*k1*k2*kt1*kt2*vh**3*(l1+l2)**2
    A1 -= (l2+d)*(m*(l1+l2)**3*(c1*kt1*l1*m2+c2*kt2*l2*m1)*np.ones(len(u)))
    B1 -= (l2+d)*((((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*l1**2+((((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+m1*kt2))*l2+12*m2*vh*kt1*m*c1)*l1+(((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*l2-12*m1*vh*kt2*m*c2)*l2)*(l1+l2)**2)
    C1 -= (l2+d)*((l1+l2)*((kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l1**3+(((((2*c1+c2)*kt2+2*c1*k2+2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2-12*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(m1*kt2-m2*kt1))*vh)*l1**2+(((((c1+2*c2)*kt2+c1*k2+c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*(((-c1*c2-m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(-m2*kt1+kt2*m1))*vh*l2+60*m2*vh**2*kt1*m*c1)*l1+l2*((kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*l2**2-12*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh*l2+60*m1*vh**2*kt2*m*c2)))
    D1 = D1 - (l2+d)*((kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l1**4+(((((3*k1+k2)*kt2+3*k1*k2)*kt1+kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2-12*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+k1*c2)*(kt2*m1-kt1*m2))*vh)*l1**3+(((((3*k1+3*k2)*kt2+3*k1*k2)*kt1+3*kt2*k1*k2)*m+(12*c1*c2*kt2+6*m2*k1*k2)*kt1+6*m1*kt2*k1*k2)*l2**2-12*((((c2-2*c1)*kt2-2*c1*k2-2*c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*vh*l2+60*((c1*c2+m2*k1)*kt1*m+c1*c2*(m2*kt1+kt2*m1))*vh**2)*l1**2+(((((k1+3*k2)*kt2+k1*k2)*kt1+3*kt2*k1*k2)*m+(8*c1*c2*kt2+4*m2*k1*k2)*kt1+4*m1*kt2*k1*k2)*l2**3-12*vh*((((2*c2-c1)*kt2-c1*k2-c2*k1)*kt1+2*kt2*(c1*k2+c2*k1))*m+3*(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*vh**2*(((c1*c2+m2*k1)*kt1+(c1*c2+m1*k2)*kt2)*m+2*c1*c2*(m2*kt1+kt2*m1))*l2+120*m2*vh**3*kt1*m*c1)*l1+l2*((kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*l2**3-12*vh*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-kt1*m2))*l2**2+60*((c1*c2+m1*k2)*kt2*m+c1*c2*(m2*kt1+kt2*m1))*vh**2*l2-120*m1*vh**3*kt2*m*c2))
    E1 -= (l2+d)*(2*kt1*kt2*(c1*k2+c2*k1)*l1**4+(8*kt1*kt2*(c1*k2+c2*k1)*l2-12*k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-m2*kt1))*vh)*l1**3+(12*kt1*kt2*(c1*k2+c2*k1)*l2**2-12*((((k2-2*k1)*kt2-2*k1*k2)*kt1+kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2+60*(kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2)*l1**2+(8*kt1*kt2*(c1*k2+c2*k1)*l2**3-12*((((2*k2-k1)*kt2-k1*k2)*kt1+2*kt2*k1*k2)*m+3*k1*k2*(kt2*m1-kt1*m2))*vh*l2**2+60*((((c1+c2)*kt2+c1*k2+c2*k1)*kt1+kt2*(c1*k2+c2*k1))*m+2*(c1*k2+c2*k1)*(m2*kt1+m1*kt2))*vh**2*l2-120*(-(c1*c2+m2*k1)*kt1*m+c1*c2*(kt2*m1-kt1*m2))*vh**3)*l1+2*l2*(kt1*kt2*(c1*k2+c2*k1)*l2**3-6*(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2*l2**2+30*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(m2*kt1+kt2*m1))*vh**2*l2-60*((c1*c2+m1*k2)*kt2*m+c1*c2*(kt2*m1-m2*kt1))*vh**3))
    F1 -= (l2+d)*(2*kt1*kt2*k1*k2*l1**4+8*kt1*kt2*k1*k2*l1**3*l2+(12*k1*k2*kt1*kt2*l2**2+60*(kt1*k1*(k2+kt2)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2)*l1**2+(8*kt1*kt2*k1*k2*l2**3+60*((((k1+k2)*kt2+k1*k2)*kt1+kt2*k1*k2)*m+2*(2*c1*c2*kt2+m2*k1*k2)*kt1+2*m1*kt2*k1*k2)*vh**2*l2-120*(-kt1*(c1*k2+c1*kt2+c2*k1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1))*vh**3)*l1+2*(kt1*kt2*k1*k2*l2**3+30*(kt2*k2*(k1+kt1)*m+(2*c1*c2*kt2+m2*k1*k2)*kt1+m1*kt2*k1*k2)*vh**2*l2-60*vh**3*(kt2*(c1*k2+c2*k1+c2*kt1)*m+(c1*k2+c2*k1)*(kt2*m1-m2*kt1)))*l2)
    G1 -= (l2+d)*(120*(kt1*kt2*(c1*k2+c2*k1)*l1**2+(2*kt1*kt2*(c1*k2+c2*k1)*l2-k1*(-kt1*(k2+kt2)*m+k2*(kt2*m1-kt1*m2))*vh)*l1+(kt1*kt2*(c1*k2+c2*k1)*l2-(kt2*(k1+kt1)*m+k1*(kt2*m1-kt1*m2))*vh*k2)*l2)*vh**2)
    H1 -= (l2+d)*(120*kt1*kt2*(l1+l2)**2*vh**2*k1*k2)
    I1 -= (l2+d)*(np.zeros(len(u)))
    A2 = (I*m*m1*m2)*((l1+l2)**3)*np.ones(len(u))
    B2 = (I*m*m1*m2)*(12*vh*(l1+l2)**2) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*((l1+l2)**3)
    C2 = (I*m*m1*m2)*(60*vh**2*(l1+l2)) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(12*vh*(l1+l2)**2) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*((l1+l2)**3)
    D2 = (I*m*m1*m2)*(120*vh**3) + (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(60*vh**2*(l1+l2)) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(12*vh*(l1+l2)**2) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    E2 = (m*l1**2*m1*m2*c1+m*l2**2*m1*m2*c2+(m2*(m+m1)*c1+m1*c2*(m+m2))*I)*(120*vh**3) + ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(60*vh**2*(l1+l2)) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*((l1+l2)**3)
    F2 = ((m*m2*k1+c2*(m+m2)*c1)*m1*l1**2+2*l2*l1*m1*m2*c1*c2+(c2*(m+m1)*c1+k2*m*m1)*m2*l2**2+(m*m2*kt1+m*m1*kt2+m2*(m+m1)*k1+c2*(m+m1+m2)*c1+m1*(m+m2)*k2)*I)*(120*vh**3) + ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(12*vh*(l1+l2)**2) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*((l1+l2)**3)
    G2 = ((m*m2*c1*kt1+m1*(m*c1*kt2+(m+m2)*(c1*k2+c2*k1)))*l1**2+2*l2*l1*m1*m2*(c1*k2+c2*k1)+(kt1*m*m2*c2+m*m1*kt2*c2+m2*(m+m1)*(c1*k2+c2*k1))*l2**2+((m2*c1+c2*(m+m2))*kt1+((m+m1)*c1+m1*c2)*kt2+(m+m1+m2)*(c1*k2+c2*k1))*I)*(120*vh**3) + (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(60*vh**2*(l1+l2)) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(12*vh*(l1+l2)**2) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*((l1+l2)**3)
    H2 = (((m*m2*k1+c1*c2*(m+m2))*kt1+((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*m1)*l1**2+2*(c1*c2*kt1*m2+m1*(kt2*c1*c2+m2*k1*k2))*l1*l2+(m2*(c1*c2+k2*m)*kt1+kt2*(c1*c2*(m+m1)+k2*m*m1)+k1*k2*m2*(m+m1))*l2**2+((kt2*m+m2*k1+c1*c2+(m+m2)*k2)*kt1+((m+m1)*k1+c1*c2+k2*m1)*kt2+k1*k2*(m+m1+m2))*I)*(120*vh**3) + (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(60*vh**2*(l1+l2)) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(12*vh*(l1+l2)**2) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*((l1+l2)**3)
    I2 = (((m*c1*kt2+(m+m2)*(c1*k2+c2*k1))*kt1+kt2*m1*(c2*k1+k2*c1))*l1**2+2*l2*(kt1*m2+m1*kt2)*(c1*k2+k1*c1)*l1+((c2*kt2*m+m2*(c2*k1+k2*c1))*kt1+kt2*(m+m1)*(c1*k2+c2*k1))*l2**2+(((c1+c2)*kt2+c2*k1+c1*k2)*kt1+kt2*(c1*k2+c2*k1))*I)*(120*vh**3) + ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(60*vh**2*(l1+l2)) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(12*vh*(l1+l2)**2) + (k1*k2*kt1*kt2*(l1+l2)**2)*((l1+l2)**3)
    J2 = ((((c1*c2+k1*m)*kt2+k1*k2*(m+m2))*kt1+k1*k2*kt2*m1)*l1**2+2*(kt1*(kt2*c1*c2+k1*k2*m2)+k1*k2*kt2*m1)*l1*l2+(((c1*c2+k2*m)*kt2+k1*k2*m2)*kt1+k1*k2*kt2*(m+m1))*l2**2+(((k1+k2)*kt2+k1*k2)*kt1+k1*k2*kt2)*I)*(120*vh**3) + (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(60*vh**2*(l1+l2)) + (k1*k2*kt1*kt2*(l1+l2)**2)*(12*vh*(l1+l2)**2)
    K2 = (kt1*kt2*(l1+l2)**2*(c1*k2+c2*k1))*(120*vh**3) + (k1*k2*kt1*kt2*(l1+l2)**2)*(60*vh**2*(l1+l2))
    L2 = (k1*k2*kt1*kt2*(l1+l2)**2)*(120*vh**3)

    # Laplace transformeer u
    lu = np.arange(0, len(u) * dt, dt)[:len(u)]
    u2 = lambda s: np.trapz(u * np.exp(-lu * s) * dt)

    # Invers laplace transformeer TF*L(u) + kort u terug in tot originele lengte
    l = np.arange(0, len(u) * dt, dt)[:len(u)] * 1j + 0.001
    nu = np.vectorize(u2)
    nsfl = 2/((A1*l**8 + B1*l**7 + C1*l**6 + D1*l**5 + E1*l**4 + F1*l**3 + G1*l**2 + H1*l + I1) / (A2*l**11 + B2*l**10 + C2*l**9 + D2*l**8 + E2*l**7 + F2*l**6 + G2*l**5 + H2*l**4 + I2*l**3 + J2*l**2 + K2*l + L2) - np.exp(-(l1+l2+d)*l/vh))
    nul = nu(l)
    y1 = lambda t: np.real(np.trapz(nul * nsfl * np.exp(l * t) * dt) / (2 * np.pi))
    ny = np.vectorize(y1)
    u = u[:u.size-N]
    t = np.arange(0, len(u) * dt, dt)[:len(u)]
    y = ny(t)
    a = np.linspace(y[0], y[-1]-yf, len(y))
    y = y - a

    # Plotten!!!
    plt.plot(t, u, 'b', label='Gemeten Hoogte')
    plt.plot(t, y, 'g', label='Gesimuleerde Hoogte wegdek')
    plt.xlabel("Tijd (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()
    return y
  
  
#TEST WEGPROFIELEN
  
#print(simuleerSnelheidHoogte(np.append(np.append(np.zeros(100), 0.1*np.ones(500)), np.zeros(1000))))#, vh=30+20*np.sin(np.arange(5000)/250)))
#print(simuleerSnelheidHoogte(np.append(np.append(np.zeros(100), np.vectorize(lambda t: -0.05*np.sin(t/50))(np.arange(157))), np.zeros(750)), vh=30*np.ones(5000)))#+20*np.cos(np.arange(5000)/250)))
#print(simuleerSnelheidHoogte(np.append(np.append(np.append(np.zeros(100), -0.001*np.arange(200)), 0.001*np.arange(500)-0.2), 0.3*np.ones(1000)), vh=30+20*np.sin(np.arange(5000)/250)))
#print(simuleerSnelheidHoogte(np.append(np.append(np.append(np.append(np.zeros(100), 0.02*np.sin(np.arange(100)/64)), 0.01/50*np.arange(50)+0.02), 0.03*np.cos(np.arange(200)/42)), np.zeros(900)), vh=30+20*np.sin(np.arange(5000)/250)))

#print(invsimuleerSnelheidHoogte(simuleerSnelheidHoogte(np.append(np.append(np.zeros(100), 0.1*np.ones(500)), np.zeros(1000)))))
#print(invsimuleerSnelheidHoogte(simuleerSnelheidHoogte(np.append(np.append(np.zeros(100), np.vectorize(lambda t: -0.05*np.sin(t/50))(np.arange(157))), np.zeros(1100)), vh=30+20*np.sin(np.arange(5000)/250), autocor=False), vh=30+20*np.sin(np.arange(5000)/250)))
#print(invsimuleerSnelheidHoogte(simuleerSnelheidHoogte(np.append(np.append(np.append(np.zeros(100), -0.001*np.arange(200)), 0.001*np.arange(500)-0.2), 0.3*np.ones(1000)), vh=30+20*np.sin(np.arange(5000)/250), autocor=False), yf=0.3, vh=30+20*np.sin(np.arange(5000)/250)))
#print(invsimuleerSnelheidHoogte(simuleerSnelheidHoogte(np.append(np.append(np.append(np.append(np.zeros(100), 0.02*np.sin(np.arange(100)/64)), 0.01/50*np.arange(50)+0.02), 0.03*np.cos(np.arange(200)/42)), np.zeros(1000)), vh=30+20*np.sin(np.arange(5000)/250), autocor=False), vh=30+20*np.sin(np.arange(5000)/250)))
#print(invsimuleerSnelheidHoogte(simuleerSnelheidHoogte(np.append(np.append(np.zeros(100), 0.05*np.sin(np.arange(23562)/100)), np.zeros(1000)), vh=30+20*np.sin(np.arange(125000)/250)), vh=30+20*np.sin(np.arange(125000)/250)))
#print(invsimuleerSnelheidHoogte(simuleer(np.append(np.append(np.zeros(100), 0.1*np.ones(500)), np.zeros(750)), vh=30+20*np.sin(np.arange(5000)/25)), vh=30+20*np.sin(np.arange(5000)/25)))
