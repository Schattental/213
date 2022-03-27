# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 23:12:14 2022

@author: Benedikt Gregor 20215194

Solving ODEs with Runge-Kutta and other methods, analyzing results and delving into Lorenz systems
"""
#%% Question 1 a)

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18}) # keep those graph fonts readable!
plt.rcParams['figure.dpi'] = 300 # plot resolution

# similar to Matlab's fval function - allows one to pass a function
def feval(funcName, *args):
    return eval(funcName)(*args)

# vectorized forward Euler with 1d numpy arrays
def euler(f, y0, t, h): # Vectorized forward Euler (so no need to loop) 
    k1 = h*f(y0, t)                     
    return y0 + k1

''' backup plan without odestepper
def euler_back(f, y0, t, h):
    n = len(t)
    y = np.zeros(n)
    y[0] = y0
    for i in range(0, n-1):
        y[i + 1] = y[i] + h*f(y[i], t[i])/(1 + h*10)
    return y
'''
def euler_back(f, y0, t, h):
    k1 = h*f(y0, t)/(1 + h*10) # not general
    return y0 + k1

# stepper function for integrating ODE solver over some time array
def odestepper(odesolver, deriv, y0, t):
    y0 = np.asarray(y0)
    y = np.zeros((t.size, y0.size))
    y[0,:] = y0
    h = t[1]-t[0]
    y_next = y0 # initial conditions 

    for i in range(1, len(t)):
        y_next = feval(odesolver, deriv, y_next, t[i-1], h)
        y[i,:] = y_next
    return y

def funexact(t):
    return np.exp(-10*t)

# -10y for some reason as in the announcements, supposedly the derivative
def fun(y,t):
    return -10*y

def plotme():
    plt.plot(ts, y2, '-r', label='Exact', linewidth=3)
    plt.plot(ts, y1, 'gs', label='F-Euler $n={}$'.format(n), markersize=8)
    plt.xlabel('$t$')     
    plt.ylabel('$y$')
    plt.xlim(0, b)
    plt.plot(ts, y3, 'b.', markersize=16, label='B-Euler $n={}$'.format(n))
    plt.legend(loc='best')
    
# defining plot format
plt.figure(figsize=(10,4))
plt.xticks([0, 0.2, 0.4, 0.6])
# first plot for n = 10
plt.subplot(1, 2, 1)
a, b, n, y0 = 0., 0.6, 10, 1.
ts=a+np.arange(n)/(n-1)*b
y1 = odestepper('euler', fun, y0, ts)
y2 = funexact(ts)
y3 = odestepper('euler_back', fun, y0, ts)
plotme()

# changing n to 20
plt.subplot(1, 2, 2)
a, b, n, y0 = 0., 0.6, 20, 1.
ts=a+np.arange(n)/(n-1)*b
y1 = odestepper('euler', fun, y0, ts)
y2 = funexact(ts)
y3 = odestepper('euler_back', fun, y0, ts)
plotme()


#%% Question 1 b)
def rk4(f, y0, t, h):                
    k0 = h*f(y0, t)                           
    yd = y0 + k0/2. 
    k1 = h*f(yd, t+h/2.) 
    yd = y0 + k1/2. 
    k2 = h*f(yd, t+h/2.)
    yd = y0 + k2 
    k3 = h*f(yd, t+h) 
    y = y0 + (k0 + 2.*(k1 + k2) + k3)/6.
    return y

T0 = 2*np.pi
T10 = T0*10
m = 1; k = 1
w0 = np.sqrt(k/m)
h = 0.01
tr = np.arange(0, T10, h)
y0 = np.array([0, 1])
# original function
def limoncello(t):
    return np.cos(w0*t)
# derivative of limoncello not used here
def lemonade(y, t):
    return -w0*np.sin(w0*t)
# odes as array
def limo(y, t):
    return np.array(y[1], -y[0])

# plotting all (I cant get this to plot properly I tried for ages)
yl = limoncello(tr)
yr = odestepper('rk4', limo, y0, tr)
yfe = odestepper('euler', limo, y0, tr)
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title("h = " + str(h))
plt.plot(yr[:, 1], yr[:, 0], label="RK4")
plt.plot(yfe[:, 1], yfe[:, 0], 'r--',label="F-Euler")
plt.ylabel("x")
plt.xlabel("v")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("h = " + str(h))
plt.plot(tr/T0, yr[:, 0], 'b', label="RK4")
plt.plot(tr/T0, yfe[:, 0], 'r,', label="F-Euler")
plt.plot(tr/T0, yl, 'g--', label="Exact")
plt.xlabel("t($T_0$)")
plt.legend()
plt.show()

# changing step size
h = 0.005
tr = np.arange(0, T10, h)
# plotting all
yl = limoncello(tr)
yr = odestepper('rk4', lemonade, 1, tr)
yfe = odestepper('euler', lemonade, 1, tr)
#plt.xticks([0, 5, 10])
tv = lemonade(0, tr)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.title("h = " + str(h))
plt.plot(tv, yr, label="RK4")
plt.plot(tv, yfe, 'r--',label="F-Euler")
plt.ylabel("x")
plt.xlabel("v")
plt.legend()

plt.subplot(1, 2, 2)
plt.title("h = " + str(h))
plt.plot(tr/T0, yr, label="RK4", color="blue")
plt.plot(tr/T0, yfe, 'r,',label="F-Euler")
plt.plot(tr/T0, yl, 'g--', label="Exact")
plt.xlabel("t($T_0$)")
plt.legend()
plt.show()

#%% Question 2 a)
# setting variables
alpha=0.
beta=1.
gamma=0.04
w=1.

# defining duffing odes
def duffing(y, t):
    return np.array([y[1], -0.08*y[1] - y[0]**3 + 0.2*np.cos(t)])
# found force to be 0.2
F = 0.2
# new initial conditions and time interval
y0 = np.array([-0.1, 0.1])
td = np.arange(0, 4*T10, 0.01)

y = odestepper('rk4', duffing, y0, td)
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.title(r'$\alpha, F, \omega$ = {},{},{}'.format(alpha, F, w))
plt.plot(y[round(len(td)/4):, 1], y[round(len(td)/4):, 0], 'r-', label = "RK4")
plt.plot(y[0, 1], y[0, 0], 'bo', markersize=16)
plt.plot(y[-1, -1], y[-1, 0], 'go', markersize=16)
plt.xlabel("v"); plt.ylabel("x")
plt.subplot(1, 2, 2)
plt.plot(td/2/np.pi, y[:, 0], 'b-', label = "RK4")
plt.xlabel("t")


#%% Question 2 b)
# new alpha
alpha = 0.1
# updated odes
def duffing2(y, t):
    return np.array([y[1], -0.08*y[1] - 0.1*y[0] - y[0]**3 + 7.5*np.cos(t)])
# new force
F = 7.5
y = odestepper('rk4', duffing2, y0, td)
plt.figure(figsize=(10,4))
plt.subplot(1, 2, 1)
plt.title(r'$\alpha, F, \omega$ = {},{},{}'.format(alpha,F, w))
plt.plot(y[round(len(td)/4):, 1], y[round(len(td)/4):, 0], 'r-', label = "RK4")
plt.plot(y[0, 1], y[0, 0], 'bo', markersize=16)
plt.plot(y[-1, -1], y[-1, 0], 'go', markersize=16)
plt.xlabel("v"); plt.ylabel("x")
plt.subplot(1, 2, 2)
plt.plot(td/2/np.pi, y[:, 0], 'b-', label = "RK4")
plt.xlabel("t")


#%% Question 3 a)

# variables for lorenz
sig = 10; r = 28;  b = 8./3.
h = 0.01
# lorenz odes as arrays
def lor(y, t):
    return np.array([sig*(y[1]-y[0]), r*y[0]-y[1]-y[0]*y[2], y[0]*y[1]-b*y[2]])


y0 = np.array([1, 1, 1])
tlo = np.arange(0, 4*T0, h)
ylo = odestepper('rk4', lor, y0, tlo)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(azim=20, elev=29)
ax.plot3D(ylo[:, 0], ylo[:, 1], ylo[:, 2], label="y0=[1,1,1]", linewidth=1)
y0 = np.array([1, 1, 1.001])
ylo2 = odestepper('rk4', lor, y0, tlo)
ax.plot3D(ylo2[:, 0], ylo2[:, 1], ylo2[:, 2], label="y0=[1,1,1.001]", linewidth=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z");
plt.legend()
plt.show()


#%% Question 3 b)
from matplotlib import animation


# making animation
# setting axes
x11 = np.array(ylo[:, 0]) 
y11 = np.array(ylo[:, 1])
z11 = np.array(ylo[:, 2])
x22 = np.array(ylo2[:, 0]) 
y22 = np.array(ylo2[:, 1])
z22 = np.array(ylo2[:, 2])


time_vals = tlo
plt.rcParams.update({'font.size': 18})
fig = plt.figure(dpi=180) # nice and sharp!
ax = fig.add_axes([0.1, 0.1, 0.85, 0.85], projection='3d')
line, = ax.plot3D(x11, y11, z11, 'r-', linewidth=0.8)
line2, = ax.plot3D(x22, y22, z22, 'b-', linewidth=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

def init():
    line.set_data(np.array([]), np.array([]))
    line.set_3d_properties([])
    line.axes.axis([-30, 30, -30, 30])
    line2.set_data(np.array([]), np.array([]))
    line2.set_3d_properties([])
    line2.axes.axis([-30, 30, -30, 30])   
    return line,line2

def update(num):
    line.set_data(x11[:num], y11[:num])
    line.set_3d_properties(z11[:num])
    line2.set_data(x22[:num], y22[:num])
    line2.set_3d_properties(z22[:num])
    fig.canvas.draw()
    return line,line2
 
ani = animation.FuncAnimation(fig, update, init_func=init, interval=1, frames=len(time_vals), blit=True, repeat=True)
plt.show()

