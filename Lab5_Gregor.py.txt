# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 02:57:43 2022

@author: Benedikt Gregor 20215194

Creating and displaying interpolation methods (legendre, monomial, spline, trig) and how they change with varying number of data points
"""
#%% Question 1 a)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # making plots higher res to save my eyes
import scipy.linalg as sp

# defining Runge's function ԅ(≖‿≖ԅ)
fine_wine = lambda x: 1/(1+25*x**2)

# function to generate data
def datagen(n, f, spacing):
    if spacing == "cheb":
        k = np.arange(1,n+1,1)
        xk_1 = np.cos(((2*k-1)/(2*n))*np.pi) # taken from wikipedia
        xk = np.flip(xk_1) # flipping to also go from negative to positive values like the equidistant points
    else:
        xk = np.linspace(-1,1,n) # making equidistant points
    yk = fine_wine(xk)
    return  {"xk": xk, "yk": yk} # returning dictionary
    

# adapted code from  Morgan & Claypool (2006) found from oregon state uni lecture on interpolation topic
def legendrepol(x, start, finish, spacing, n):
    # generating data points
    xin = datagen(n, fine_wine, spacing)["xk"]
    yin = datagen(n, fine_wine, spacing)["yk"]
    y = 0.0;
    for i in range(start, finish+1): 
       l = 1.0
       for j in range(start, finish+1):
           if i != j:
              l *= ((x - xin[j-1])/(xin[i-1] - xin[j-1]))
       y += (yin[i-1] * l)
    return y

xs = np.linspace(-1, 1, 100) # making the 100 equidistant x points for plotting

def monomialpol(n, spacing):
    # generating data points
    x = datagen(n, fine_wine, spacing)["xk"]
    y = datagen(n, fine_wine, spacing)["yk"]
    ls = np.zeros(n**2)
    m = ls.reshape(n,n) # making empty matrix of n*n shape
    # filling first column with 1s which is not necessary (will be to the power of 0 anyways) but wanted to be sure it happens
    for i in range(n):
        m[i][0] = 1
        for k in range(n-1):
            m[i][k+1] = x[i]**(k+1) # filling matrix
    sol = sp.solve(m, y)
    
    coeffs = np.flip(sol) # flipping array because poly1d makes the first element the highest order x
    p = np.poly1d(coeffs) # making a polynomial from all the coeffs
    ys = p(xs)
    return ys


def plotter(spacing, n):
    # plotting points for interpolation
    xx = datagen(n, fine_wine, spacing)["xk"]
    yy = datagen(n, fine_wine, spacing)["yk"]
    plt.plot(xx, yy, '.', color = "purple", label = "data")

    # monomial interpolation
    yss = monomialpol(n, spacing)

    # legendre interpolation
    ys = legendrepol(xs, 1, len(datagen(n, fine_wine, "does not matter here")["xk"]), spacing, n)

    # legendre plot
    plt.plot(xs, ys, 'r', label="legendre")
    # monomial plot
    plt.plot(xs, yss, 'g--', label = "monomial")
    plt.title(spacing + " n = " + str(n))
    plt.legend()

# n = 15
# plotting with chebyshev points
plt.subplot(1, 2, 1)
plotter("cheb", 15)

# plotting with equidistant points
plt.subplot(1, 2, 2)
plotter("equi", 15)
plt.show()

#%% Question 1 b)
# n = 91
# plotting with chebyshev points
plt.subplot(1, 2, 1)
plotter("cheb", 91)

# plotting with equidistant points
plt.subplot(1, 2, 2)
plotter("equi", 91)
plt.show()

# n = 101
# plotting with chebyshev points
plt.subplot(1, 2, 1)
plotter("cheb", 101)

# plotting with equidistant points
plt.subplot(1, 2, 2)
plotter("equi", 101)
plt.show()

# comments on the interpolations
print("The interpolatioon using equidistant points gets fruther away from the real function with an increase in n.")
print("Which is becasue the oscillations caused by higher order polynomials are boosted whith this increase in n.")
print("It can also be seen that with an increase in n of 10 the peak of the oscillations increase by one order of magnitude.")
print("In contrast the interpolation with chebyshev points gets smoother the bigger n is with diminishing returns once a certain number is reached.")

#%% Question 2
# this spline function is NOT all mine
# adapted code from RH Landau, MJ Paez, and CC Bordeianu for spline interpolation
def spline(n, spacing):
    # defining lists to append to later
    xx = []
    yy = []
    # making array of x values for interpolation to be plotted with
    xs = np.linspace(-1, 1, 100)
    # generating data points to interpolate between
    x = datagen(n, fine_wine, spacing)["xk"]
    y = datagen(n, fine_wine, spacing)["yk"]
    # creating initial arrays y and u
    y2 = np.zeros((n), float);    u = np.zeros((n), float)

    # calculating the first derivative
    for i in range(0, n):                  
        yp1 = (y[1]-y[0])/(x[1]-x[0])-(y[2]-y[1])/(x[2]-x[1])+(y[2]-y[0])/(x[2]-x[0])
    
    # calculating the nth derivative
    ypn = (y[n-1] - y[n-2])/(x[n-1] - x[n-2]) - (y[n-2] - y[n-3])/(x[n-2] - x[n-3]) + (y[n-1] - y[n-3])/(x[n - 1] - x[n - 3])
    if(yp1 > 0.99e30):
        y2[0] = 0.
        u[0] = 0.
    else:
        y2[0] = - 0.5
        u[0] = (3./(x[1] - x[0]) )*( (y[1] - y[0])/(x[1] - x[0]) - yp1)
    
    # decomposition
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1])/(x[i + 1] - x[i - 1]) 
        p = sig*y2[i - 1] + 2. 
        y2[i] = (sig - 1.)/p 
        u[i] = (y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i]-y[i-1])/(x[i]-x[i-1])
        u[i] = (6.*u[i]/(x[i + 1] - x[i - 1]) - sig*u[i - 1])/p
    
    # testing for natural functions
    if (ypn > 0.99e30):  qn = un = 0.
    else:
        qn = 0.5;
        un = (3/(x[n-1] - x[n-2]) )*(ypn - (y[n-1] - y[n-2])/(x[n-1]-x[n-2]))
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.)

    for k in range(n-2, 1, - 1):  y2[k] = y2[k]*y2[k + 1] + u[k]

    # making the fit
    for i in range(1, len(xs) + 2):
        xout = x[0] + (x[n - 1] - x[0])*(i - 1)/(len(xs)) 
        klo = 0;    khi = n - 1
        while (khi - klo >1):
            k = (khi + klo) >> 1
            if (x[k] > xout): khi  = k
            else: klo = k
        h = x[khi] - x[klo] 
        if (x[k] > xout):  khi = k
        else: klo = k 
        h = x[khi] - x[klo]
        a = (x[khi] - xout)/h 
        b = (xout - x[klo])/h 
        yout = a*y[klo]+b*y[khi] +((a*a*a-a)*y2[klo]+(b*b*b-b)*y2[khi])*h*h/6
        xx.append(xout)
        yy.append(yout)
    return {"xk": xx, "yk": yy} # returning dictionary for easy referencing

def plotter2(n):
    # using subplots for side by side plots
    # generating x and y for spline and the data that is being interpolated with chebyshev points
    x1 = spline(n, "cheb")["xk"]
    y1 = spline(n, "cheb")["yk"]
    x0 = datagen(n, fine_wine, "cheb")["xk"]
    y0 = datagen(n, fine_wine, "cheb")["yk"]
    # plotting data points and spline of those
    plt.subplot(1, 2, 1)
    plt.plot(x0, y0, '.', color = "purple", label = "data")
    plt.plot(x1, y1, label= "spline")
    plt.title("cheb n = " + str(n))
    plt.legend()
    # generating x and y for spline and the data that is being interpolated with equidistant points
    plt.subplot(1, 2, 2)
    x2 = spline(n, "equi")["xk"]
    y2 = spline(n, "equi")["yk"]
    x00 = datagen(n, fine_wine, "equi")["xk"]
    y00 = datagen(n, fine_wine, "equi")["yk"]
    # plotting data points and spline of those
    plt.plot(x00, y00, '.', color = "purple", label = "data")
    plt.plot(x2, y2, label= "spline")
    plt.title("equi n = " + str(n))
    plt.legend()
    plt.show()

# plotting both for n = 7 and n = 15
plotter2(7)
plotter2(15)

#%% Question 3 a)
import math as m

schnaps = lambda x: m.exp(m.sin(2*x)) # defining periodic function

# function for trigonometric interpolation
def trigpol(n):
    jaeger = np.vectorize(schnaps) # vectorizing to convert down to size 1
    m = int((n-1)/2)
    j = np.arange(0, (n), 1)
    xj = (2*np.pi*j)/n # making equal distance points between 0 and 2*pi
    
    yj = jaeger(xj) # making y values with vectorized version og periodic function
    # defining lists to append to because I dont get how np.append works... ¯\_(ツ)_/¯
    ax = []
    bx = []
    # calcualting a and b for the formula
    for k in range(m): # looks like having the same range for k here does not matter ò_ô
        ax.append((1/m)*np.sum(yj*np.cos(k*xj)))
        bx.append((1/m)*np.sum(yj*np.sin(k*xj)))
    
    # making the lists to numpy arrays and defining the x points for plotting the interpolation
    a = np.array(ax); b = np.array(bx)
    xs = np.linspace(0, 2*np.pi, 500) # making 500 points for plotting because why not
    
    ys = 0.5*a[0] + sum([(a[k]*np.cos(k*xs) + b[k]*np.sin(k*xs)) for k in range(1, m, 1)])
    
    # plotting raw data and interpolation
    plt.plot(xj, yj, '.', color = "magenta", label="data")
    plt.plot(xs, ys, color = "orange", label= "trig")
    plt.title("trig n = " + str(n))
    plt.legend()
    
# plotting for both n = 11 and n = 51
plt.subplot(1, 2, 1)
trigpol(11)
plt.subplot(1, 2, 2)
trigpol(51)

