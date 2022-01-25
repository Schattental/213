# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 05:51:13 2022

@author: Benedikt Gregor 20215194
"""
#%% Question 1 a)
import numpy as np
import math as m
import sympy as sm

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
        n = 2*n - 1
        res = trap(f, a, b, n)
        rel_err = ((m.erf(1) - res)/m.erf(1))*100
        if rel_err < 1e-13:
            break
    print("Steps needed for trapezoid rule to reach 1e-13 relative error = " + str(n))
    
adaptive_step_trap(erf, 0, 1, 3)

def adaptive_step_simp(f, a, b, n):
    while(True):
        n = 2*n - 1
        res = simp(f, a, b, n)
        rel_err = ((m.erf(1) - res)/m.erf(1))*100
        if abs(rel_err) < 1e-13:
            break
    print("Steps needed for simpson's rule to reach 1e-13 relative error = " + str(n))
    
adaptive_step_simp(erf, 0, 1, 3)

    
#%% Question 2 a)

#%% Question 2 b)

#%% Question 3 a)

#%% Question 3 b)

#%% Question 3 c)

#%% Question 3 d)