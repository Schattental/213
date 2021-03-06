# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 05:51:13 2022

@author: Benedikt Gregor 20215194
"""
#%% Question 1 a)
import numpy as np
import math as m
import sympy as sm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # making plots more readable with higher resolution
import matplotlib.pyplot as plt
import math as m
import mpmath as mp
from scipy.integrate import dblquad

# Rectangle rule for integration as in the lecture slides
n = 100
a = 0.
b = 1.
h = (b-a)/n # the mistake here was n-1 from the lecture slides

int = 0. # Initialize the integral variable
for i in range(0,n):
    xi = a + h*i # Determine the xi value for the loop
    int = int + (2/np.pi**0.5)*np.exp(-xi **2)*h
    
#print(int)

# defining lambda function of erf to easily plug into the integral approx functions
erf = lambda x: (2/np.pi**0.5)*np.exp(-x**2)

# Rectangle rule method as a function
def rect(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a,b,n) # using an np array here to avoid a for loop (vectorization)
    return np.sum(f(x))*h

#print(rect(erf, 0, 1, 100))


    
#%% Question 1 b)
# Trapezoid method of integral approximation
def trap(f, a, b, n):
    h = (b-a)/n
    x = np.linspace(a,b,n+1)
    return (np.sum(f(x)) - 0.5*f(a) - 0.5*f(b))*h
    
#print(trap(erf, 0, 1, 100))
    
# Simpson's rule for approximation 
def simp(f, a, b, n):
    if n % 2 == 0:
        print("Use uneven n for more accurate approx.")
    h = (b-a)/(n-1)
    x = np.linspace(a+h, b-h, n-2)
    alt = np.empty((n-2,),int) # creating an alternating array to multiply with the array of values from f(x) to treat even and odd
    alt[::2] = 4 # using slicing to my advantage, this is faster than tiling
    alt[1::2] = 2
    return (h/3)*(np.sum(f(x)*alt) + f(a) + f(b))

#print(simp(erf, 0, 1, 101))

# comparing the methods
print("Relative errors for rectangle rule")
print(((m.erf(1) - rect(erf, 0, 1, 100))/m.erf(1))*100, "%")
print(((m.erf(1) - rect(erf, 0, 1, 101))/m.erf(1))*100, "%\n")
print("Relative errors for trapezoid rule")
print(((m.erf(1) - trap(erf, 0, 1, 100))/m.erf(1))*100, "%")
print(((m.erf(1) - trap(erf, 0, 1, 101))/m.erf(1))*100, "%\n")
print("Relative errors for simpson's rule")
print(((m.erf(1) - simp(erf, 0, 1, 100))/m.erf(1))*100, "%")
print(((m.erf(1) - simp(erf, 0, 1, 101))/m.erf(1))*100, "%\n")

'''
The errors above show a clear improvement in accuracy for the rectangle and trapezoid rule when increasing n.
However, an incredible amount of precision is gained with simpson's rule by going to 101. The cause being that an uneven n (as discussed in the lectures)
is needed for simpson's rule to show its full potential.
For this reason a tip is printed to the console when the function detects even n.
'''
#%% Question 1 c)

def adaptive_step_trap(f, a, b, n):
    while(True):
        n = 2*n - 1 # as instructed in the assignment using this to increase n
        res = trap(f, a, b, n)
        rel_err = (m.erf(1) - res)/m.erf(1) # not multiplying by 100 here because the value is used for comparison in the next line
        if abs(rel_err) < 1e-13: # breaking out of loop until min error is reached
            #print(abs(rel_err))
            break
    print("Steps needed for trapezoid rule to reach 1e-13 relative error = " + str(n))
    
adaptive_step_trap(erf, 0, 1, 3)

def adaptive_step_simp(f, a, b, n):
    while(True):
        n = 2*n - 1 # as instructed in the assignment using this to increase n
        res = simp(f, a, b, n)
        rel_err = (m.erf(1) - res)/m.erf(1) # not multiplying by 100 here because the value is used for comparison in the next line
        if abs(rel_err) < 1e-13: # breaking out of loop until min error is reached
            #print(abs(rel_err))
            break
    print("Steps needed for simpson's rule to reach 1e-13 relative error = " + str(n))
    
adaptive_step_simp(erf, 0, 1, 3)

    
#%% Question 2 a)
array_in = np.loadtxt("Hysteresis-Data.csv", delimiter=',', skiprows=1) #skipping the frist roe because it only contains formatting info
vx = array_in[:,1] # taking second element of each row (second column)
vy = array_in[:,2] # taking third element of each row (third column)

plt.plot(vx, vy, 'r-'); plt.xlabel('$V_x$'); plt.ylabel('$V_y$') # plotting the data and giving it a red colour
plt.show()

#%% Question 2 b)
def integral(vx, vy):
    h = (vx[:-1]+vx[1:])/(len(vx)-2) # h is the equivalent to dx
    return np.sum((vy[1:] + vy[:-1])*h/2) # multiplying dx and y values then take the sum

print("Area between the lines = " + str(integral(vx, vy)))

#%% Question 3 a)
#2D simpson
def simp2d(f2d, a, b, c, d, n, m):

    h_i = (b-a)/(n-1)
    h_j = (d-c)/(m-1)
    
    xs1 = np.linspace(a, b, n)
    xs2 = np.linspace(c, d, m)
    
    alt_n = np.empty((n,),int) # creating an alternating array to multiply with the array of values from f2d to treat even and odd
    alt_n[::2] = 2 # using slicing to my advantage, this is faster than tiling
    alt_n[1::2] = 4 # creates alternating array length of n [1, 4, 2, 4, ..., 4, 2, 1]
    alt_n[0], alt_n[-1] = 1, 1 # setting first and last element 
    alt_m = np.empty((m,),int) # creating an alternating array to multiply with the array of values from f2d to treat even and odd
    alt_m[::2] = 2 # using slicing to my advantage, this is faster than tiling
    alt_m[1::2] = 4 # creates alternating array length of m  [1, 4, 2, 4, ..., 4, 2, 1]
    alt_m[0], alt_m[-1] = 1, 1 # setting first and last element 
    
    #applying weighting to alternating arrays
    c_i = (h_i/3)*alt_n
    c_j = (h_j/3)*alt_m
    cs2d = np.outer(c_i, c_j) # making a weight matrix
    #print(alt_n)
    X, Y = np.meshgrid(xs1, xs2, indexing = 'ij')
    return np.sum(cs2d*f2d(X, Y)) # applying weight matrix to mesh of X, Y values and taking sum

#%% Question 3 b)
f2 = np.vectorize(lambda x,y: m.sqrt(x**2+y)*m.sin(x)*m.cos(y)) # using vectorize to "flatten" the function

print("Simpson's double integral using n = 101, m = 101 : " + str(simp2d(f2, 0, m.pi, 0, m.pi/2, 101, 101)))
print("Simpson's double integral using n = 1001, m = 1001 : " + str(simp2d(f2, 0, m.pi, 0, m.pi/2, 1001, 1001)))
print("Simpson's double integral using n = 51, m = 101 : " + str(simp2d(f2, 0, m.pi, 0, m.pi/2, 51, 101)))

#%% Question 3 c)
# accurate integral using mpmath library
f2d = lambda x,y: m.sqrt(x**2+y)*m.sin(x)*m.cos(y) 
print("Double integral using mp.quad : " + str(mp.quad(f2d, [0, m.pi], [0, m.pi/2])))

#%% Question 3 d)
# accurate integral using scipy
print("Double integral using scipy dblquad : " + str(dblquad(f2d, 0, m.pi/2, 0, m.pi))) # takes in boundaries for y first and then x ? weird
# can use epsrel=1.e-10 to further reduce error
