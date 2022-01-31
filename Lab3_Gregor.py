# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:36:32 2022

@author: Benedikt Gregor 20215194
"""

#%% Question 1 a)
import numpy as np

def matrix(n):
    ar = np.linspace(21, 20+n**2, n**2)
    matrix = ar.reshape(n,n)
    #print(matrix)
    return matrix

def trilow(A):
    dim = len(A); size = dim**2
    zero = np.zeros(size)
    zerom = zero.reshape(dim, dim)
    for i in range(dim):
        temp = A[i:,i]
        i*np.insert(temp, 0, 0)
        zerom[i:,i] = temp
    return zerom

def triupp(A):
    dim = len(A); size = dim**2
    zero = np.zeros(size)
    zerom = zero.reshape(dim, dim)
    for i in range(dim):
        temp = A[:i+1,i]
        nulls = np.empty(dim-(i+1)); nulls.fill(0)
        temp = np.append(temp, nulls)
        zerom[:,i] = temp
    return zerom

print(trilow(matrix(4)))
print(triupp(matrix(4)))
    
    
#print(np.tril(matrix(4)))

def frob_norm(A):
    return np.sqrt(np.sum(np.abs(A)**2))

def inf_norm(A):
    return np.max(np.sum(np.abs(A), axis = 1))

print(frob_norm(matrix(3)))
print(np.linalg.norm(matrix(3), 'fro'))
print(inf_norm(matrix(3)))
print(np.linalg.norm(matrix(3),np.inf))

#%% Question 1 b)
def invdiag(n):
    zero = np.zeros(n**2)
    zero.fill(-1)
    zerom = zero.reshape(n, n)
    upp = triupp(zerom)
    np.fill_diagonal(upp, 1)
    return upp

#print(invdiag(16))
def solve_invdiag(A, pert):
    b = np.empty((len(A),),int)
    b[::2] = 1
    b[1::2] = -1
    if pert:
        A[-1:,0] = -0.001
    solv = np.linalg.solve(A, b)
    res = np.append(solv, np.linalg.cond(A))
    return res
    
diag4F = solve_invdiag(invdiag(4), False)
diag4T = solve_invdiag(invdiag(4), True)
diag16F = solve_invdiag(invdiag(16), False)
diag16T = solve_invdiag(invdiag(16), True)

print("First 3 slns for A4 no pert = " + str(diag4F[0:3]) + " cond num = " + str(diag4F[-1]))
print("First 3 slns for A4 with pert = " + str(diag4T[0:3]) + " cond num = " + str(diag4T[-1]))
print("First 3 slns for A16 no pert = " + str(diag16F[0:3]) + " cond num = " + str(diag16F[-1]))
print("First 3 slns for A16 with pert = " + str(diag16T[0:3]) + " cond num = " + str(diag16T[-1]))


#%% Question 2 a)
def backsub1 (U, bs):
    n = bs.size
    xs = np.zeros_like(bs) # assigns same dims as bs
    xs[n-1] = bs[n-1]/U[n-1 , n-1] # bb is not needed for this case
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range(i+1, n):
            bb += U[i, j]*xs[j]
            xs[i] = (bs[i] - bb)/U[i, i]
    return xs

U = matrix(5000)
bs = U[0:1]

from timeit import default_timer
#from pymoso.chnutils import testsolve
timer_start = default_timer()
#testsolve(backsub1(U , bs))
timer_end = default_timer()
time1 = timer_end - timer_start
print('time1 :', time1 )
#%% Question 2 b)

#%% Question 2 c)

#%% Question 2 d)