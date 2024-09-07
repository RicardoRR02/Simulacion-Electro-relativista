import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, proton_mass, elementary_charge
from mpl_toolkits.mplot3d import Axes3D
import math as ma
from decimal import Decimal ,getcontext
import time
from matplotlib.ticker import ScalarFormatter
e=elementary_charge
med=proton_mass
masa=med*c**2 *207
q = e *c *82
B0=20*10**7
#B0=1*10**5
#condiciones iniciales y parametros de la simulacion
x0 = np.array([0, 5, 0])
v0 = np.array([0*c, 0.2*c, 0.7*c])
t0 = 0
tf = 100
dt_init = 0.0001
tol = 1000
getcontext().prec=50
#definir la exactitud de los decimales usado en las operaciones
def d(x):
    n=Decimal(x)
    return n
# Funcion que calcula la aceleracion de la partícula cargada con efetos relativistas
def acceleration(x, v, t):
    # Calcular la aceleracion de la particula cargada utilizando la ecuacion de movimiento de Lorentz relativista
    #print(d((d((np.linalg.norm(v) / c)))**2), np.linalg.norm(v))
    gamma =d( 1 / d(ma.sqrt(d(1 - d((d((np.linalg.norm(v) / c)))**2)))))
    gamma=float(gamma)
    beta=v/c
    betacua=np.dot(beta,beta)
    G=(2*gamma+1)/(gamma+1)
    frecuencia= q*B0/(masa*gamma)
    #definir vectores comines
    x , y , z = x[0], x[1], x[2]
    r= ma.sqrt(x**2+y**2+z**2)
    s=ma.sqrt(x**2+y**2)
    #proteger el codigo si tenemos r o s que sean cero, por ejemplo campo coulombmico
    if r==0 or s ==0:
        r=10**-4
        s=10**-4
    #vectores unitarios esfericos, para eficiencia solo quitar como comentario el que se vaya a usar
    #teta=ma.atan(s/z)
    #phi=ma.atan(y/x)
    #runi=np.array([x/r,y/r,z/r])
    #tethauni=np.array([ma.cos(teta)*cos(phi),ma.cos(teta)*ma.sin(phi),-ma.sin(teta)])
    #phiuni=np.array([-ma.sin(phi),ma.cos(phi),0])
    #vectores unitarios cilindricos, para eficiencia solo quitar como comentario el que se vaya a usar
    suni=np.array([x/s,y/s,0])
    #phiunicil=np.array([-y/s,x/s,0])
    # Definir las componentes del campo electromagnetico E y B
    #E = np.array([0,B0,0])
    E = 0*suni 
    B =np.array([0,0,B0])

    #ecuacion de la aceleracion de la particula
    a = (masa/q)*( (1+betacua)*E + 2*np.cross(beta,B) - G*np.dot(beta,E)*beta )
    #time.sleep(0.1)
    return a

# Metodo de Runge-Kutta de cuarto orden 
def rk4_step(x, v, dt, t):
    k1v = dt * acceleration(x, v, t)
    k1x = dt * v
    k2v = dt * acceleration(x + 0.5 * k1x, v + 0.5 * k1v, t + 0.5 * dt)
    k2x = dt * (v + 0.5 * k1v)
    k3v = dt * acceleration(x + 0.5 * k2x, v + 0.5 * k2v, t + 0.5 * dt)
    k3x = dt * (v + 0.5 * k2v)
    k4v = dt * acceleration(x + k3x, v + k3v, t + dt)
    k4x = dt * (v + k3v)

    new_v = v + (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
    new_x = x + (1/6) * (k1x + 2*k2x + 2*k3x + k4x)

    return new_x, new_v
#metodo de Runge-Kutta de tercer orden
def rk3_step(x, v, dt, t):
    k1v = dt * acceleration(x, v, t)
    k1x = dt * v
    k2v = dt * acceleration(x + 0.5 * k1x, v + 0.5 * k1v, t + 0.5 * dt)
    k2x = dt * (v + 0.5 * k1v)
    k3v = dt * acceleration(x + k2x - k1x, v + k2v - k1v, t + dt)
    k3x = dt * (v + k2v - k1v)

    new_v = v + (1/4) * (k1v + 3*k3v)
    new_x = x + (1/4) * (k1x + 3*k3x)

    return new_x, new_v
#metodo de Runge-kutta con paso adaptativo
def runge_kutta_adaptive(x0, v0, t0, tf, dt_init, tol):
    t = [t0]
    x = [x0]
    v = [v0]

    t_current = t0
    x_current = x0
    v_current = v0
    dt = dt_init

    while t_current < tf:
        if np.linalg.norm(v0)>=c:
            print("Velocidad  mayor que la velocidad de la luz...")
            break
        while True:
            try:
                # Paso con Runge-Kutta de 4 orden
                x_rk4, v_rk4 = rk4_step(x_current, v_current, dt, t_current)
                    
                # Paso con Runge-Kutta de 3 orden
                x_rk3, v_rk3 = rk3_step(x_current, v_current, dt, t_current)
                break
            except (ValueError, ZeroDivisionError):
                dt *=0.5
        # Estimación del error
        error = np.linalg.norm(x_rk4 - x_rk3)
        #print(np.linalg.norm(v_rk4)/c)
        if error < tol and np.linalg.norm(v_rk4) < c:
            # Aceptar el paso
            t_current += dt
            x_current = x_rk4
            v_current = v_rk4
            t.append(t_current)
            x.append(x_current)
            v.append(v_current)
            # Incrementar el paso si el error es menor que una fracción de la tolerancia
            if error < tol / 10:
                dt *= 2
        else:
            # Reducir el tamaño del paso
            dt *= 0.5

    return t, x, v
# Resolver las ecuaciones de movimiento utilizando Runge-Kutta de cuarto orden
t, x, v = runge_kutta_adaptive(x0, v0, t0, tf, dt_init, tol)
#Ejecutar el metodo para resolver las ecuaciones, solo si v0 es menor a c
if np.linalg.norm(v0)<c:
    #Se define la lista de las posiciones como vectores columna
    xx=[]
    for elemento in x:
        xx.append(np.array(elemento)[:, np.newaxis])
    #Se define la matriz inversa de transformacion de Lorentz
    def matinv(v):
        gamma =d( 1 / d(ma.sqrt(d(1 - d((d((np.linalg.norm(v) / c)))**2)))))
        gamma=float(gamma)
        v_norm = np.linalg.norm(v)
        #Componente diagonal de la matriz inversa
        def A(a):
            A=1+(gamma-1)*v[a]**2/v_norm**2
            return A
        #Componentes de la matriz que no son de la diagonal
        def B(a,b):
            B=(gamma-1)*v[a]*v[b]/v_norm**2
            return B
        mat=np.array([[A(0),  B(0,1),B(0,2)],
                      [B(1,0),A(1),  B(1,2)],
                      [B(2,0),B(2,1),A(2) ]])
        return mat
    #Se define la parte temporal de la transformacion de Lorentz
    def vect(v):
        gamma = 1 / np.sqrt(1 - np.linalg.norm(v)**2 / c**2)
        vt=np.array([[gamma*v[0]/c],
                     [gamma*v[1]/c],
                     [gamma*v[2]/c]])
        return vt
    #Se realiza la transformacion
    x_prima=[]
    for i in range (0,len(x)):
        x_prima.append(np.dot(matinv(v[i]),xx[i])+vect(v[i])*t[i])
    x_primaa = [(v[0, 0], v[1, 0], v[2, 0]) for v in x_prima]
    xob = [p[0] for p in x_primaa]
    yob = [p[1] for p in x_primaa]
    zob = [p[2] for p in x_primaa]
    xpa = [p[0] for p in x]
    ypa = [p[1] for p in x]
    zpa = [p[2] for p in x]
    

    fig = plt.figure(figsize=(16, 8))
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.05, wspace=0)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(xpa, ypa, zpa, label='Trayectoria de la Partícula', color='blue', linewidth=1)
    ax1.set_title("Sistema de la Partícula", fontsize=18)
    ax1.set_xlabel('X (m)', fontsize=14)
    ax1.set_ylabel('Y (m)', fontsize=14)
    ax1.set_zlabel('Z (m)', fontsize=14)
    ax1.view_init(elev=30, azim=45)
    ax1.legend()
    ax1.set_box_aspect([1, 1, 1]) 
    formatter = ScalarFormatter(useMathText=True)  
    formatter.set_scientific(True)  
    formatter.set_powerlimits((-2, 2))  
    ax1.xaxis.set_major_formatter(formatter)
    ax1.yaxis.set_major_formatter(formatter)
    ax1.zaxis.set_major_formatter(formatter)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(xob, yob, zob, label='Trayectoria de la Partícula', color='red', linewidth=1)
    ax2.set_title("Sistema del Observador", fontsize=18)
    ax2.set_xlabel('X (m)', fontsize=14)
    ax2.set_ylabel('Y (m)', fontsize=14)
    ax2.set_zlabel('Z (m)', fontsize=14)
    ax2.view_init(elev=30, azim=45)
    ax2.legend()
    ax2.set_box_aspect([1, 1, 1]) 
    formatter = ScalarFormatter(useMathText=True) 
    formatter.set_scientific(True)  
    formatter.set_powerlimits((-2, 2))  
    ax2.xaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    ax2.zaxis.set_major_formatter(formatter)
    plt.show()
    #graficas de los plamos 
    #XY
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(xpa, ypa, color='blue')
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('y (m)')
    axs[0].grid(True)
    axs[0].set_title('Sistema partícula \n Plano XY')
    
    axs[1].plot(xob, yob, color='red')
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel('y (m)')
    axs[1].grid(True)
    axs[1].set_title('Sistema observador \n Plano XY')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  
    for ax in axs:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
    #XZ
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(xpa, zpa, color='blue')
    axs[0].set_xlabel('x (m)')
    axs[0].set_ylabel('z (m)')
    axs[0].grid(True)
    axs[0].set_title('Sistema partícula \n Plano XZ')
    
    axs[1].plot(xob, zob, color='red')
    axs[1].set_xlabel('x (m)')
    axs[1].set_ylabel('z (m)')
    axs[1].grid(True)
    axs[1].set_title('Sistema observador \n Plano XZ')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  
    for ax in axs:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()
    #YZ
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(ypa, zpa, color='blue')
    axs[0].set_xlabel('y (m)')
    axs[0].set_ylabel('z (m)')
    axs[0].grid(True)
    axs[0].set_title('Sistema partícula \n Plano XZ')
    
    axs[1].plot(yob, zob, color='red')
    axs[1].set_xlabel('y (m)')
    axs[1].set_ylabel('z (m)')
    axs[1].grid(True)
    axs[1].set_title('Sistema observador \n Plano XZ')
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))  
    for ax in axs:
        ax.xaxis.set_major_formatter(formatter)
        ax.yaxis.set_major_formatter(formatter)
    plt.tight_layout()
    plt.show()       



