# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 16:31:03 2022

@author: Benedikt Gregor 20215194
"""
#%%Question 1 a)

# test function defined here, it does not have any roots
sprite = lambda x: 1/(x-3)

# bisection method for approximating roots
def bisect(f, x_m, x_p, cycles=500, tol=1e-8): 
    for i in range(cycles):
        x = (x_m + x_p)/2 # calculating the middle point
        if f(x)*f(x_p) > 0: # checking if the new point is positive
            x_p = x # making the new point the new positive x
        else:
            x_m = x # else making it the new negative x
    
        xn = (x_m+x_p)/2
        dx = abs(xn-x)
        
        if abs(dx/xn) < tol: # checking to see if the error is small enough
            break
        else:
            xn = None
    return xn


res = bisect(sprite, 0, 5) 
print(res)
# the solution to this is not correct. bisect subroutine returns the point at which the function isn't defined not its roots
# However, the function does not have roots so asking whether or not this result is correct is wrong itself
# this shows how robust the approach is. It does its thing and doesn't just crash even though the function is trying to trick it.

#%%Question 1 b)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

fanta = lambda x: np.exp(x-np.sqrt(x))-x
    
# finding root near 1 using window [0, 2] to avoid root near 2.5
res2 = bisect(fanta, 0, 2)
print("Approx. root position from bisect method = ", res2)

# Newton-Raphson method to approximate roots
def raph(f, xi, cycles = 100, tol=1e-8): # xi = initial guess for x  
    for i in range(cycles):
        fd = (f(xi + 0.01) - f(xi))/0.01
        dx = -f(xi)/fd
        xi = xi + dx
        if abs(dx/xi) < tol:
            #print(i)
            break
    return xi

print("Approx. root position from N-R method = ", raph(fanta, 0.01))

# Newton-Raphson method visualized
# using 4 cycles here to get the same graph as in assignment
def raph2graph(f, xi, cycles = 4, tol=1e-8): # xi = initial guess for x  
    xs = np.linspace(0, 1, 100) # restricing to go from 0 to 1
    plt.plot(xs, f(xs), color='blue', lw=2)
    plt.ylabel('$f$(x)'); plt.xlabel('x')
    plt.grid(visible=True) # showing grid
    
    for i in range(cycles):
        fd = (f(xi + 0.01) - f(xi))/0.01 # numerical derivative, chose 0.01 as my perturbation cuz I felt like it was good enough
        dx = -f(xi)/fd
        
        xss = np.linspace(xi, 1, 10) # making a few x points for the tangent line
        b = f(xi)-fd*xi # getting b in y = kx + b for the tangent line
        plt.plot(xss, (fd*xss+b), 'r--') # plotting the tangent
        plt.plot([xi, xi], [f(xi), 0], 'k:') # plotting a straight line down
        plt.text(xi, 0, r'$x^{(%.f)}$' % i , fontsize='x-large') # adding the description for every x
        xi = xi + dx # updating x for next cycle
        
        if abs(dx/xi) < tol:
            #print(i)
            break
        
    plt.ylim([0, 1]) # limitting y axis to only show truncated version
    plt.title('Newton-Raphson method visualized')
    plt.show() # showing all plots together at the end of the subroutine
    
    return xi   

raph2graph(fanta, 0.01)

#%%Question 1 c)
import math as m

a = 1 # setting the value of the root to be ignored
beer = lambda x: (m.exp(x-m.sqrt(x))-x)/(x-a) # new function to exclude root at 1
# also using math library here because with np I encountered a runtime warning for an invalid value in double_scalars
# it seems numpy isn't great at handling very large values

def raph_x(f, xi, cycles = 100, tol=1e-8): # xi = initial guess for x  
    for i in range(cycles):
        fd = (f(xi + 0.01) - f(xi))/0.01
        dx = -f(xi)/fd
        xi = xi + dx
        if abs(dx/xi) < tol:
            #print(i)
            break
    return xi

# testing all inital conditions
print(raph_x(beer, 2.0))
print(raph_x(beer, 0.5))
print(raph_x(beer, 4.0))
print(raph_x(beer, 0.1))

#%%Question 1 d)
R = 1 # radius of sphere
roh_w = 1 # water density
roh_s = 0.8 # sphere density

vodka = lambda h: roh_w*(1/3)*m.pi*(3*R*m.pow(h,2)-m.pow(h,3))-(4/3)*m.pi*m.pow(R,3)*roh_s # formula to express when Volume*roh_liquid = m_sphere

soln = bisect(vodka, 0, 2) # calculating h with bisection method
print("distance submerged in water = ", soln)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"}) # declaring the plot to be 3D
ax.set_box_aspect([10,10,4.5]) # setting aspect ratio because otherwise the sphere looks squished / stretched 

XX, YY = np.mgrid[-2:2:10j, -2:2:10j] 

ZZ = np.zeros_like(XX)

ax.plot_surface(XX,YY,ZZ, alpha=0.5, zorder=0.3) # plotting surface to represent water

# creating unit sphere
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
X = np.cos(u)*np.sin(v)
Y = np.sin(u)*np.sin(v)
Z = np.cos(v)+1

# using vmin and vmax to show about where 0 is (not setting it very close to 0 because the sharp cuoff is ugly)
ax.plot_surface(X, Y, Z-soln, cmap=mpl.cm.winter, alpha=0.8,vmin=-0.3, vmax=0.01, zorder=0.5) 
ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
ax.set_title('Sphere in water with green being above surface')

plt.show()

#%%Question 2 a)
import sympy as sm

x0, x1, x2 = sm.symbols('x0 x1 x2') # defining symbols, so when another set is added the new ones have to be added here

# I decided to go with sympy functions here, thought it was convenient, but it turned out to be a bit hard to work with

def fs1():
    
    f0 = x0**2 - 2*x0+np.power(x1,4) - 2*x1**2 + x1
    f1 = x0**2 + x0 + 2*np.power(x1,3) - 2*x1**2 - 1.5*x1 - 0.05
    
    return np.array([f0, f1])


def fs2():
    
    f0 = 2*x0 - x1*sm.cos(x2) - 3
    f1 = x0**2 - 25*(x1 - 2)**2 + sm.sin(x2) - np.pi/10
    f2 = 7*x0*sm.exp(x1) - 17*x2 + 8*np.pi
    
    return np.array([f0, f1, f2])

#%%Question 2 b)
import numpy.linalg as la 

# numerical Jacobian
def jan(f, x, h=1e-4):
    n = np.zeros(len(x)**2)
    m = n.reshape(len(x), len(x)) # making an empty matrix of the correct size for jacobian
    Id = np.copy(m) # copying empty jacobian
    np.fill_diagonal(Id, 1) # making identity matrix
    b = [] # define emply list for b to use later
    
    for j in range(len(x)):
        for i in range(len(x)):
            ff = sm.lambdify(list(sm.ordered(f()[j].free_symbols)), f()[j]) # getting the function of the set
            
            m[j][i] = (ff(*(x+Id[i]*h))-ff(*x))/h # numerically evaluate the derivative and put into jacobian
            
        fn = sm.lambdify(list(sm.ordered(f()[j].free_symbols)), f()[j]) # lambdifying the sympy functions from the set
        b.append(-1*fn(*x)) # evaluating b column for later use
            
    return {"Matrix": m, "b": b} # returning dictionary

# testing 
J = jan(fs1, [3, 2])["Matrix"]

print(J, jan(fs1, [3, 2])["b"])

#%%Question 2 c)
# root approximation subroutine with an input j to choose which type jacobian to use
def nr_ja(f, x, j, cycles=10, tol=1e-8):
    
    xroots = la.solve(j(f, x)["Matrix"], j(f, x)["b"]) # getting first list of roots
    
    for k in range(cycles):
        hroots = la.solve(j(f, xroots)["Matrix"], j(f, xroots)["b"])
        new_roots = xroots + hroots
        dr = new_roots - xroots
        if(np.sum(abs(dr/new_roots)) < tol):
            break
        xroots = new_roots
    return xroots
            
# defining intervals as per assignment instructions
x_1 = np.array([1.0, 1.0])
x_2 = np.array([1.0, 1.0, 1.0])

r1 = nr_ja(fs1, x_1, jan)
print("Roots for fs1:", r1)

r2 = nr_ja(fs2, x_2, jan)
print("Roots for fs2:", r2)
print()

# test subroutine to check if roots are actually close to 0 by subbing them in
def testroots(fs, x):
    res = []
    for j in range(len(x)):
        ff = sm.lambdify(list(sm.ordered(fs()[j].free_symbols)), fs()[j])
        res.append(ff(*x))
    return res

check1 = testroots(fs1, r1)
print("Checking roots to see if they are close to 0 for fs1")
print(check1) # should be close to 0
print()
print("Checking roots to see if they are close to 0 for fs2")
check2 = testroots(fs2, r2)
print(check2) # should be close to 0

#%%Question 2 d)

# analytical jacobian
# took me eons to write this, but I learned a lot about lambda and sympy functions
def jaa(fs, x):
    b = []
    n = np.zeros(len(x)**2)
    m = n.reshape(len(x), len(x)) # making an empty matrix of the correct size
    for j in range(len(x)):
        for i in range(len(x)):
            var = list(sm.ordered(fs()[j].free_symbols))[i] # getting the variable to take the correct partial derivative
            fd = sm.diff(fs()[j], var) # taking the derivative
            fd_l = sm.lambdify(list(sm.ordered(fs()[j].free_symbols)), fd) # lambdifying the derivative to evaluate with all variables

            m[j][i] = fd_l(*x) # using asterisk to pass parameters in form of array to lambda function and putting value into matrix

        fs_l = sm.lambdify(list(sm.ordered(fs()[j].free_symbols)), fs()[j]) # lambdifiying the sympy function from the set
        b.append(-1*fs_l(*x)) # evaluatint the function at x to get b
        
    return {"Matrix": m, "b": b} # returning dictionary

r1_a = nr_ja(fs1, x_1, jaa)
print("Roots for fs1:", r1_a)

r2_a = nr_ja(fs2, x_2, jaa)
print("Roots for fs2:", r2_a)

check1_a = testroots(fs1, r1_a)
print("Checking roots to see if they are close to 0 for fs1")
print(check1_a) # should be close to 0
print()
print("Checking roots to see if they are close to 0 for fs2")
check2_a = testroots(fs2, r2_a)
print(check2_a) # should be close to 0

# COMMENT ABOUT H
# Testing the step size h, a noticeable difference is seen when h is bigger or equal to 1e-2



