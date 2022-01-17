# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 22:45:59 2022

@author: Beni
"""
#%% Question 1 a)
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
import numpy as np
import math as m
import sympy as sm
from sympy.abc import x
import matplotlib.pyplot as plt

la1 = lambda s, m : sm.diff(sm.exp(sm.sin(s*x)), x, m)

f0 = la1(2,0)
f1 = la1(2,1)
f2 = la1(2,2)
f3 = la1(2,3)

print(f1)
print(f2)
print(f3)

f0l = sm.lambdify(x, f0)
f1l = sm.lambdify(x, f1)
f2l = sm.lambdify(x, f2)
f3l = sm.lambdify(x, f3)

#ff0 = np.vectorize(f0l)


xs = np.linspace(0, 2*m.pi, 200)
#print(xs)
#print(len(xs),xs[0],xs[len(xs)-1 ] - 2 * m.pi)

line0, = plt.plot(xs, f0l(xs), 'r'); line0.set_label("f")
line1, = plt.plot(xs, f1l(xs), 'g'); line1.set_label("f '")
line2, = plt.plot(xs, f2l(xs), 'b'); line2.set_label("f ''")
line3, = plt.plot(xs, f3l(xs), 'y'); line3.set_label("f '''")
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
#plt.plot(xs, f0l(xs), f1l(xs), f2l(xs))

#sm.plot(expr0, expr1, expr2, expr3, (x,0,2*m.pi), title = "Derivatives", legend = True)
#sm.plot(xs, 2*np.exp(np.sin(2*xs))*np.cos(2*xs), title = "Plots", legend = True)

#sm.plot(sm.exp(sm.sin(2*x)), (x, 0, 2*m.pi))
    

#%% Question 1 b)
def calc_fd(f,x,h): # forward difference
    fd = (f(x+h) - f(x))/h
    return fd
def calc_cd(f,x,h): # central difference
    cd = (f(x+h/2) - f(x-h/2))/h
    return cd
h = 0.15
fd, = plt.plot(xs, calc_fd(f0l,xs,h)); fd.set_label("fd 0.15")
cd, = plt.plot(xs, calc_cd(f0l,xs,h)); cd.set_label("cd 0.15")
h = 0.5
fd2, = plt.plot(xs, calc_fd(f0l,xs,h)); fd2.set_label("fd 0.5")
cd2, = plt.plot(xs, calc_cd(f0l,xs,h)); cd2.set_label("cd 0.5")
anal, = plt.plot(xs, f1l(xs), '--'); anal.set_label("analytical")

plt.xlabel('x'); plt.ylabel('y')
plt.legend(loc = 'lower right')

#%% Question 1 c)
hs = np.linspace(10e-16, 1, 16)
absfd, = plt.plot(hs, ((f0l(1 + hs)-f0l(1))/hs)-f1l(1)); absfd.set_label("Absolute error f-d")
abscd, = plt.plot(hs, ((f0l(1 + hs/2)-f0l(1 - hs/2))/hs)-f1l(1)); abscd.set_label("Absolute error c-d")
plt.legend()

