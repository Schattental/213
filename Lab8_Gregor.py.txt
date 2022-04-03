# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 00:33:57 2022

@author: Benedikt Gregor 20215194
Solving PDEs in 1D and 2D using jacobi iteration
"""
#%% Question 1
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300 # plot resolution

# initial conditions at x,t = 0
def heat_initial(x):
    return 20 + 30*np.exp(-100*(x-0.5)**2)

# length of rod in [m]
L = 1
# diffusion coefficient alpha in [m^2/s]
alpha = 2.3e-4
# time steps and number of grod points
dt = 0.1; n = 100
# defining kappa used in approximation
dx = L/n
kappa = (alpha*dt)/dx**2
# time to iterate over
t = np.arange(0, 61, dt)
x = np.arange(0, 1, dx)

def heat_iterator(t, dt):
    # checking caurant stability
    if (kappa > 0.5):
        raise Exception("Caurant stability unsatisfied")
    # matrix with time stepping through rows and space in col
    m = np.zeros((len(t), len(x))) 
    m[:, 0] = 20 # setting boundary on first column
    m[:, -1] = 20 # boundary on last column
    m[0, :] = heat_initial(x) # initial conditions
    u = m[0] # template for array
    for j in range(1, len(t)):
        u1 = m[j-1] # u1 is the spacial array at a time step
        u[1:-2] = u1[1:-2] + kappa*(u1[0:-3] - 2*u1[1:-2] + u1[2:-1])
        m[j] = u
    m[0] = heat_initial(x) # setting inital again because it was altered
    return m

res = heat_iterator(t, dt)
# time steps where to plot
steps = [0, 50, 100, 200, 300, 600]
# subroutine to plot the time stamps
def plott(time):
    for i in range(len(time)):
        plt.plot(x, res[time[i]], label="time = " + str(time[i]/10), lw=2)
    plt.xlabel("Rod length [m]"); plt.ylabel("Temperature [C]")
    plt.legend()
    plt.show()
    
plott(steps)


#%% Question 2
from numpy import linalg as lg

x = np.linspace(0, 2, 100)
y = np.linspace(0, 1, 50)

# souce equation
kombucha = lambda x, y: np.cos(10.*x) - np.sin(5.*y - np.pi/4.)

#x,y = np.ix_(x,y) was working on a version with this but meshgrid seemed easier

# jacobi iteration
def jacobi(x, y, fun):
    # step size
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    s = np.zeros((len(y), len(x))) # making a matrix as an output
    x, y = np.meshgrid(x, y)
    f = fun(x, y) # initial values from source
    s[0] = 0; s[-1] = 0; s[:,0] = 0; s[:,-1] = 0 # boundaries along all sides of the matrix
    nm = lg.norm(s, ord=1)
    while True:
        s[1:-1,1:-1] = (hy**2 * (s[2:,1:-1] + s[:-2,1:-1]) + hx**2 * (s[1:-1,2:] + s[1:-1,:-2]) - (hy**2 * hx**2)*f[1:-1, 1:-1])/(2*(hx**2 + hy**2))
        nm_new = lg.norm(s, ord=1)
        if(nm_new - nm)/nm_new > 1e-5:
            nm = nm_new
        else:
            break
    return s

''' Older attempt at this leaving it here for future reference
    for i in range(1, len(x)-1):
        ux = s[i]
        for j in range(1, len(y)-1):
            uy = s[:,j]
            #f = (ux[0:-3] - 2*ux[1:-2] + ux[2:-1])/hx**2 + (uy[0:-3] - 2*uy[1:-2] + uy[2:-1])/hy**2
            u[1:-2] = (hy**2 * (ux[0:-3] + ux[2:-1]) + hx**2 * (uy[0:-3] + uy[2:-1]) - (hy**2 * hx**2)*f)/2*(hx**2 + hy**2)
        s[i] = u
    return s
'''
sol = jacobi(x, y, kombucha)
# using inferno cmap because it looks cooler ;) 
plt.imshow(sol, cmap = "inferno", origin = "lower", extent=[0,2,0,1])
cb = plt.colorbar()
cb.solids.set_edgecolor("face")
cb.set_label("$\phi(x, y)$")
plt.xlabel("x"); plt.ylabel("y")
plt.draw()
plt.show()

#%% Question 3
import time # used for measuring runtime
# ülker makes cheap soda drinks
ülker = lambda x, y: np.cos(3*x + 4*y) - np.cos(5*x - 2*y)
n = 800
pi2 = 2*np.pi
# using fourier transforms to solve for solutions
def jacobifft(n, fun):
    x = np.linspace(0, pi2, n); y = x # x and y are the same to make a square
    h = x[1] - x[0] # h is the same for y 
    x, y = np.meshgrid(x, y)
    f = np.fft.fft2(fun(x, y)) # transforming to fourier space
    s = np.zeros((n,n)) # blank matrix for solutions
    k = np.arange(0, n); l = k # k and l are the same
    k, l = np.meshgrid(k, l) # also making a meshgrid here to avoid broadcasting problems
    # calculating using given formula and slicing to avoid for loops
    s[1:,:] = np.real(0.5*((h**2 * f[1:,:])/((np.cos(pi2*k/n) + np.cos(pi2*l/n) -2)[1:,:])))
    s[0,1:] = np.real(0.5*((h**2 * f[0,1:])/((np.cos(pi2*k/n) + np.cos(pi2*l/n) -2)[0,1:])))
    return np.fft.ifft2(s)
# measuring time to execute jacobifft
start_time = time.time()
sol = jacobifft(n, ülker)
print("Runtime for jacobifft = %s s" % (time.time() - start_time))
# plotting results as cmap
plt.imshow(np.real(sol), cmap= "magma", extent= [0,2,0,2])
cb = plt.colorbar()
cb.solids.set_edgecolor("face")
cb.set_label("$\phi(x, y)$")
plt.xlabel("x($\pi$)"); plt.ylabel("y($\pi$)")
plt.draw()
plt.show()
    
    
