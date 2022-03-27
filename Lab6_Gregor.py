# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 03:57:19 2022

@author: Benedikt Gregor 20215194

Discrete Fourier transformations and the inverse
"""
#%% Question 1 a)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300 # my eyes are pleased by this

pi2 = np.pi*2 # pi twice

# defining function to generate data
chardonnay = lambda a0, a1, a2, w0, w1, w2, t: a0*np.sin(w0*t) + a1*np.sin(w1*t) + a2*np.sin(w2*t)

# making the equally spaced time intervals with n steps
t30 = np.linspace(0, pi2, 30)
t60 = np.linspace(0, pi2, 60)
signal30y = chardonnay(3., 1., 0.5, 1., 4., 7., t30)
signal60y = chardonnay(3., 1., 0.5, 1., 4., 7., t60)

def dft(f, N, t):
    if N % 2 != 0:
        raise Exception("Only even N accepted") # throwing a nice excpetion
    k = np.linspace(0, N-1, N) # k and j are the same
    y = np.sum(f*np.exp(-1j*k[:, None]*k*pi2/N), axis=1) # according to equation
    return {"x": k, "y": y} # returning dictionary

def idft(ydft, N, a=0, b=pi2):
    if N % 2 != 0:
        raise Exception("Only even N accepted") # extremely clean indeed 
    k = np.linspace(0, N-1, N) # j and k are the same so using only k
    y = 1/N*np.sum(ydft*np.exp(1j*k*2*np.pi/N*k[:, None]), axis=1) # according to equation
    return {"x": (a-b)*k/N+a, "y": y} # returning dictionary
    
t60 = np.linspace(0, pi2, 60) # not sure why this is here again
signal60y = chardonnay(3., 1., 0.5, 1., 4., 7., t60) # generating the data with the given parameters

# plotting everything for both n
plt.subplot(1, 2, 1)
plt.plot(t60, signal60y, color = "blue", label = "n = 60")
plt.plot(t30, signal30y, 'r--', label = "n = 30")
plt.xlabel("t"); plt.ylabel("y")
plt.legend()
plt.subplot(1, 2, 2)
# plotting dft of the data generated in line 41 for both n
plt.stem(dft(signal60y, 60, t60)["x"], abs(dft(signal60y, 60, t60)["y"]), 'r', markerfmt=" ", basefmt="-r", label= "n = 60")
plt.stem(dft(signal30y, 30, t30)["x"], abs(dft(signal30y, 30, t30)["y"]), 'b', markerfmt=" ", basefmt="-b", label= "n = 30")
plt.xlabel("$\omega$");
plt.legend()
plt.show()

    


#%% Question 1 b)
# sensitivity as far as I understand it is the difference caused by a parameter change
# here I am changing the a1 variable of the input function "chardonnay"
t60a = np.arange(0, pi2, pi2/60) # making arange for n = 60

signal60y2 = chardonnay(3., 2., 0.5, 1., 4., 7., t60) # linspace with parameter change

signal60ay = chardonnay(3., 1., 0.5, 1., 4., 7., t60a) # arange normal
signal60ay2 = chardonnay(3., 2., 0.5, 1., 4., 7., t60a) # arange with parameter change

# calculating the difference caused by the parameter change of the function
diffa = np.sum(abs(dft(signal60ay2, 60, t60a)["y"]) - abs(dft(signal60ay, 60, t60a)["y"]))
diffl = np.sum(abs(dft(signal60y2, 60, t60)["y"]) - abs(dft(signal60y, 60, t60)["y"]))
# interpreting results
print("Change in arange:", diffa, "\nChange in linspace: ", diffl)
print("The difference caused by the parameter change is bigger using np.linspace")
print("The difference of the two methods is in this case ", 100*(diffa-diffl)/diffa, "%")
#%% Question 1 c)
ydft = dft(signal60y, 60, t60)["y"]
# plotting  original unmodified data
plt.plot(t60, signal60y,'b', label="function")
yidft = idft(ydft, 60)["y"]
# plotting data that has gone through dft and back with idft
plt.plot(np.real(t60), np.real(yidft), 'r--', label="idft")
plt.title("Function to dft back with idft")
plt.legend()
plt.text(0,-2, "The functions match nicely")
plt.show()
#%% Question 2 a)
# gaussian pulse function
gaup = lambda t, sig, w: np.exp((-t**2)/sig**2)*np.cos(w*t)
# new time range with n steps
t60p = np.linspace(-np.pi, np.pi, 60)
yg = gaup(t60p, 0.5, 0) # generating datan with gaussian pulse function
ys = dft(yg, 60, t60p)["y"] # transforming

# shifting as shown in lecture slides
w_shift = np.fft.fftfreq(60, pi2/60)*2.*np.pi
w_shift = np.fft.fftshift(w_shift)
y_shift = np.fft.fftshift(ys)

# plotting shifted and unshifted data
plt.subplot(1, 2, 1)
plt.plot(t60p, yg, 'b')
plt.xlabel("t"); plt.ylabel("y")
plt.subplot(1, 2, 2)
plt.plot(dft(yg, 60, t60p)["x"], abs(dft(yg, 60, t60p)["y"]), 'b', label=  "no shift")
plt.plot(w_shift, abs(y_shift), 'r--', label= "with shift")
plt.xlabel("$\omega$")
plt.legend()
plt.show()
#%% Question 2 b)
# getting tired of the manual shifting so making a quick subroutine called shifted
def shifted(n, sig, w):
    tnp = np.linspace(-np.pi, np.pi, n)
    y1 = gaup(tnp, sig, w)
    y = dft(y1, n, tnp)["y"]
    w_shift = np.fft.fftfreq(n, pi2/n)*2.*np.pi
    w_shift = np.fft.fftshift(w_shift)
    y_shift = np.fft.fftshift(y)
    return w_shift, y_shift

# new time range with more steps
t400p = np.linspace(-np.pi, np.pi, 400)
# data with new sig and w
y10p = gaup(t400p, 1, 10)
y20p = gaup(t400p, 1, 20)
# shifting the new data
w10s, y10s = shifted(400, 1, 10)
w20s, y20s = shifted(400, 1, 20)
# plotting the shifted data with both frequencies
plt.subplot(1, 2, 1)
plt.plot(t400p, y10p, color = "purple")
plt.plot(t400p, y20p, color = "orange")
plt.xlabel("t"); plt.ylabel("y")
plt.subplot(1, 2, 2)
plt.xlim([-40, 40]) # limiting x to zoom in a bit like in instructions
plt.plot(w10s, abs(y10s), label="$\omega$ = 10", color = "purple")
plt.plot(w20s, abs(y20s), label="$\omega$ = 20", color = "orange")
plt.xlabel("$\omega$")
plt.legend()
plt.show()

#%% Question 3
# new time range now till 8*pi with n steps
t200e = np.linspace(0, pi2*4, 200)
# changing the first function and adding noise with w1
riesling = chardonnay(3., 1., 0., 1., 10., 0., t200e)
# getting transformed data generated with noisy set
rdft = dft(riesling, 200, t200e)["y"]
rxdft = dft(riesling, 200, t200e)["x"]

# RECEIVED HELP FROM FELLOW STUDENT KEEGAN KELLY WITH THE FILTER FUNCTION
def filt(t, y):
    # getting max y value
    m = abs(y).max()
    # setting unwanted y to 0 if they are approx max
    for i in range(len(t)):
        if round(abs(y[i])) != round(m):
            y[i] = 0
    return y

# filtering the transformed array of values
rdft_filt = filt(t200e, rdft)
# converting the filtered data back to a wave with idft
ridft = idft(rdft_filt, 200)["y"]

# plotting filtered and unfiltered data in both tima and frequency domains
plt.subplot(1, 2, 1)
plt.plot(t200e, riesling, 'r--',label = "unfiltered")
plt.plot(t200e, np.real(ridft), 'b',label = "filtered")
plt.xlabel("t"); plt.ylabel("y")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(rxdft/4, abs(dft(riesling, 200, t200e)["y"]), 'r--', label = "unfiltered")
plt.plot(rxdft/4, abs(rdft_filt), 'b',label = "filtered")
plt.xlabel("$\omega$")
plt.legend()
plt.show()
