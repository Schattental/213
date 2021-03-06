# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:45:59 2022

@author: Benenedikt Gregor 20215194
"""
#%% Question 1 a)
#importing some important libraries
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # setting a higher resolution because nobody deserves blurry plots
import numpy as np
import math as m
import sympy as sm
from sympy.abc import x
import matplotlib.pyplot as plt

# making it easy to get the differentials
la1 = lambda s, m : sm.diff(sm.exp(sm.sin(s*x)), x, m)

# saving the orders from 0 to 3
f0 = la1(2,0)
f1 = la1(2,1)
f2 = la1(2,2)
f3 = la1(2,3)

# printing the equations of the differntials to console
print(f1)
print(f2)
print(f3)

# making lambda functions out of the functions for later use
f0l = sm.lambdify(x, f0)
f1l = sm.lambdify(x, f1)
f2l = sm.lambdify(x, f2)
f3l = sm.lambdify(x, f3)

#ff0 = np.vectorize(f0l)

# defining points on x axis
xs = np.linspace(0, 2*m.pi, 200)

# creating line2D objects or artists for easy labeling and plotting
line0, = plt.plot(xs, f0l(xs), 'r'); line0.set_label("f")
line1, = plt.plot(xs, f1l(xs), 'g'); line1.set_label("f '")
line2, = plt.plot(xs, f2l(xs), 'b'); line2.set_label("f ''")
line3, = plt.plot(xs, f3l(xs), 'y'); line3.set_label("f '''")
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.show()

#%% Question 1 b)
def calc_fd(f,x,h): # forward difference
    fd = (f(x+h) - f(x))/h
    return fd

def calc_cd(f,x,h): # central difference
    cd = (f(x+h/2) - f(x-h/2))/h
    return cd

h = 0.15
# plotting forward and central difference for h = 0.15
fd, = plt.plot(xs, calc_fd(f0l,xs,h)); fd.set_label("fd 0.15")
cd, = plt.plot(xs, calc_cd(f0l,xs,h)); cd.set_label("cd 0.15")

h = 0.5
# plotting forward and central difference for h = 0.5
fd2, = plt.plot(xs, calc_fd(f0l,xs,h)); fd2.set_label("fd 0.5")
cd2, = plt.plot(xs, calc_cd(f0l,xs,h)); cd2.set_label("cd 0.5")
anal, = plt.plot(xs, f1l(xs), '--'); anal.set_label("analytical")
# labeling x and y
plt.xlabel('x'); plt.ylabel('y')
plt.legend(loc = 'lower right')
plt.show()

#%% Question 1 c)
# stepping through 10e-16 to 1 in 16 steps to go through one order of magnitude every time
hs = np.linspace(10e-16, 1, 16)
r_err = np.finfo(np.float64).eps #rounding error as defined in lecture slides

# absolute errors
absfd, = plt.plot(hs, abs((calc_fd(f0l,1,hs)-f1l(1)))); absfd.set_label("Absolute error f-d")
abscd, = plt.plot(hs, abs((calc_cd(f0l,1,hs)-f1l(1)))); abscd.set_label("Absolute error c-d")

# analytical errors using formulae given in the lectures and using rounding error here
anafd, = plt.plot(hs, abs((h/2)*f2l(hs)+2*f0l(hs) * (r_err/h))); anafd.set_label("Analytical error f-d")
anacd, = plt.plot(hs, abs((h**2/24)*f3l(hs)+2 * f0l(hs)*(r_err/h))); anacd.set_label("Analytical error c-d")

plt.legend()
plt.show()
#%% Question 1 d)
# Richardson error 
rfd = [abs((2*calc_fd(f0l, 1, h/2) - calc_fd(f0l, 1, h))-f1l(1)) for h in hs]
rcd = [abs((4*calc_cd(f0l, 1, h/2) - calc_cd(f0l, 1, h))/3-f1l(1)) for h in hs]
richfd, = plt.plot(hs, rfd); richfd.set_label("Richardson f-d")
richcd, = plt.plot(hs, rcd); richcd.set_label("Richardson c-d")

# analytical error as before
anafd, = plt.plot(hs, abs((h/2)*f2l(hs)+2*f0l(hs) * (r_err/h))); anafd.set_label("Analytical error f-d")
anacd, = plt.plot(hs, abs((h**2/24)*f3l(hs)+2 * f0l(hs)*(r_err/h))); anacd.set_label("Analytical error c-d")

plt.legend()
plt.show()
#%% Question 2 a)
h = 0.01

# using code given in assignment instructions
def calc_cd_1(f, n, x, h):
    cd = (f(n, x+h/2) - f(n, x-h/2))/h
    return cd

def calc_cd_2(f, n, x, h):
    cd = (calc_cd_1(f, n, x+h/2, h) - calc_cd_1(f, n, x-h/2, h))/h
    return cd

def calc_cd_3(f, n, x, h):
    cd = (calc_cd_2(f, n, x+h/2, h) - calc_cd_2(f, n, x-h/2, h))/h
    return cd

def calc_cd_4(f, n, x, h):
    cd = (calc_cd_3(f, n, x+h/2, h) - calc_cd_3(f, n, x-h/2, h))/h
    return cd

def LPs(n, x, h):
    def f(n, x): return (pow(x, 2)-1)**n # sadly there is no switch / case in this python verison :(
    fn = 1;
    if n == 1:
        fn = calc_cd_1(f, n, x, h)
    elif n == 2:
        fn = calc_cd_2(f, n, x, h)
    elif n == 3:
        fn = calc_cd_3(f, n, x, h)
    elif n == 4:
        fn = calc_cd_4(f, n, x, h)
    return 1/((2**n)*m.factorial(n))*fn
    
# using legendre equation from legendre.py which was provided
def legendre(n,x):
    if n==0: # P
        val2 = 1. # P0
        dval2 = 0. # P0'
    elif n==1: # derivatives
        val2 = x # P1'
        dval2 = 1. # P1'
    else:
        val0 = 1.; val1 = x # sep P0 and P2 to start recurrence relation
        for j in range(1,n):
            val2 = ((2*j+1)*x*val1 - j*val0)/(j+1) 
            # P_j+1=(2j+1)xP_j(x)-jP_j-1(x)  / (j+1), starts from P2
            val0, val1 = val1, val2
        dval2 = n*(val0-x*val1)/(1.-x**2) # derivative
    return val2, dval2

xs = np.linspace(0, 1, 200) #200 steps like in demo

# using plotting function from call_legendre.py which was provided
def plotcomp(nsteps):

    xs = [i/nsteps for i in range (-nsteps+1,nsteps)]
    for n in range(1,5): # this will compute P1 to P4
        ys = [legendre(n,x)[0] for x in xs]
        plt.plot(xs, ys, 'k-', label='recurrence n={0}'.format(n), linewidth=3) # .format() for automatic labeling
        plt.xlabel('$x$', fontsize=20)
        plt.ylabel("$P_n(x)$", fontsize=20)
        ys = [LPs(n,x,h) for x in xs]
        plt.plot(xs, ys, 'r--', label='Rodrigues n={0}'.format(n), linewidth=3)
        plt.legend(loc="best")
        plt.show() # create a new graph for each n
        
nsteps = 200 # number of x points
plotcomp(nsteps)

#%%Question 2 b)
def calc_cd_n(f, n, x, h, order): # recursive function 
    if order == 1:
        return calc_cd_1(f, n, x, h) #using previous cd function for convenience
    return (calc_cd_n(f, n, x+h/2, h, order-1) - calc_cd_n(f, n, x-h/2, h, order-1))/h # calling recursively

def LPss(n, x, h): # new LP function to use the recursive cd
    def f(n, x): return (pow(x,2)-1)**n
    fn = calc_cd_n(f, n, x, h, n)
    return 1/((2**n)*m.factorial(n))*fn

def plotcomp2(nsteps): #slightly modified to plot to P8

    xs = [i/nsteps for i in range (-nsteps+1,nsteps)]
    for n in range(1,9): # this will compute P1 to P8
        ys = [legendre(n,x)[0] for x in xs]
        plt.plot(xs, ys, 'k-', label='recurrence n={0}'.format(n), linewidth=3)
        plt.xlabel('$x$', fontsize=20)
        plt.ylabel("$P_n(x)$", fontsize=20)
        ys = [LPss(n,x,h) for x in xs] # new LP function used here
        plt.plot(xs, ys, 'r--', label='Rodrigues n={0}'.format(n), linewidth=3)
        plt.legend(loc="best")
        plt.show() # create a new graph for each n
        
nsteps = 200 # number of x points
plotcomp2(nsteps)

