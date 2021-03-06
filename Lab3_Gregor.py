# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 17:36:32 2022

@author: Benedikt Gregor 20215194
"""

#%% Question 1 a)
import numpy as np

def matrix(n):
    # creating an array of length n**2 with values from 21 to n
    ar = np.linspace(21, 20+n**2, n**2)
    # shaping the array to a matrix
    matrix = ar.reshape(n,n)
    #print(matrix)
    return matrix

def trilow(A):
    dim = len(A); size = dim**2 # getting the size of the matrix to know what to deal with
    zero = np.zeros(size) # making an array of zeros
    zerom = zero.reshape(dim, dim) # making a matrix of the zeros array
    for i in range(dim):
        temp = A[i:,i] # getting the first col and then with every loop get one less row of each next col
        i*np.insert(temp, 0, 0) # inserting the columns into the zero matrix
        zerom[i:,i] = temp
    return zerom

def triupp(A):
    dim = len(A); size = dim**2
    zero = np.zeros(size)
    zerom = zero.reshape(dim, dim) # making zero matrix as in trilow
    for i in range(dim):
        temp = A[:i+1,i] # getting rows and every time with less of the values
        nulls = np.empty(dim-(i+1)); nulls.fill(0) # filling the missing values with 0
        temp = np.append(temp, nulls) 
        zerom[:,i] = temp # adding the rows to our zero matrix which acts as a placeholder
    return zerom

print(trilow(matrix(4)))
print(triupp(matrix(4)))
    
    
#print(np.tril(matrix(4)))

def frob_norm(A):
    return np.sqrt(np.sum(np.abs(A)**2)) # literally translating the math into python and it just works

def inf_norm(A):
    return np.max(np.sum(np.abs(A), axis = 1)) # taking what is on the assignment instructions and writing in python code

print(frob_norm(matrix(3)))
print(np.linalg.norm(matrix(3), 'fro'))
print(inf_norm(matrix(3)))
print(np.linalg.norm(matrix(3),np.inf))

#%% Question 1 b)
def invdiag(n):
    zero = np.zeros(n**2)
    zero.fill(-1)
    zerom = zero.reshape(n, n) # making a matrix of -1
    upp = triupp(zerom) # using triupp to leave only upper triangle
    np.fill_diagonal(upp, 1) # replacing the diagonal with 1
    return upp # returning finished matrix

# print test invdiag with any n:
#print(invdiag(16))

def solve_invdiag(A, pert):
    b = np.empty((len(A),),int) # maing empty array
    b[::2] = 1 # slicing to get alternating array of 1 and -1
    b[1::2] = -1
    if pert: # checking if to include a perturbation
        A[-1:,0] = -0.001 # setting the perturbation 
    solv = np.linalg.solve(A, b)
    res = np.append(solv, np.linalg.cond(A))
    return res
    
diag4F = solve_invdiag(invdiag(4), False)
diag4T = solve_invdiag(invdiag(4), True)
diag16F = solve_invdiag(invdiag(16), False)
diag16T = solve_invdiag(invdiag(16), True)

# printing first three solutions of every variant and condition number
print("First 3 slns for A4 no pert = " + str(diag4F[0:3]) + " cond num = " + str(diag4F[-1]))
print("First 3 slns for A4 with pert = " + str(diag4T[0:3]) + " cond num = " + str(diag4T[-1]))
print("First 3 slns for A16 no pert = " + str(diag16F[0:3]) + " cond num = " + str(diag16F[-1]))
print("First 3 slns for A16 with pert = " + str(diag16T[0:3]) + " cond num = " + str(diag16T[-1]))


#%% Question 2 a)
# I thought I might have to use pymoso for the testsolve() which is in the example but I didn't bother
#from pymoso import testsolve

# I tried for ages to get this to work without any for loops
# because xs and bs are interdependent I think it is impossible to completely vectorize
# without changing the math so one for loop has to stay
def backsub2(U, bs):
    n = bs.size
    xs = np.zeros(n)
    xs[n-1] = bs[n-1]/U[n-1 , n-1] 
    for i in range(n-2, -1, -1):
        # vectorizing the summing step
        xs[i] = (bs[i] - np.sum(U[i, i+1:n] * xs[i+1:n])) / U[i, i]
    return xs

# original backsub code
def backsub1 (U, bs):
    n = bs.size
    xs = np.zeros(n) # assigns same dims as bs
    xs[n-1] = bs[n-1]/U[n-1 , n-1] # bb is not needed for this case
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range(i+1, n):
            bb += U[i, j]*xs[j]
            xs[i] = (bs[i] - bb)/U[i, i]
    return xs

# making a test matrix
U = matrix(5000)
bs = U[0:1].flatten()

from timeit import default_timer
#from pymoso.chnutils import testsolve

# subroutine for solutions of backsub and timing them
def mysolve(f, U, b):
    timer_start = default_timer()
    sol = f(U, b)
    timer_end = default_timer()
    time1 = timer_end - timer_start
    print("Solutions: ", sol[0], sol[1], sol[3])
    print('Took this long :', time1 )
    
mysolve(backsub1, U, bs)
mysolve(backsub2, U, bs)
#%% Question 2 b)
# basically the same code as in the slides
def gauelim(A, b):
    U = np.copy(A)
    bs = np.copy(b)
    n = len(bs)
    for k in range(n-1):
        for i in range(k+1, n):
            eff = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= eff * U[k, j]
            bs[i] -= eff*bs[k]
    xs = backsub2(U, bs)
    return xs

A = np.array([[2, 1, 1], 
              [1, 1, -2], 
              [1, 2, 1]], float)

b1 = np.array([8, -2, 2], float)
sov = gauelim(A, b1)
print(sov)

#%% Question 2 c)
# modified elimination by checking for small values and changing rows
def gauelim2(A, b):
    U = np.copy(A)
    bs = np.copy(b)
    n = len(bs)
    for k in range(n-1):
        # checking for 0 or small values that approach 0
        if np.fabs(U[k, k]) < 0.1: # not sure what value is small enough so... just used 0.1 which seems to work
            for i in range(k+1, n):
                if np.fabs(U[i, k]) > np.fabs(U[k, k]):
                    #swapping rows of U and b
                    U[[k, i]] = U[[i, k]]
                    bs[[k, i]] = bs[[i, k]]
                    break # after switching return to elimination 
        for i in range(k+1, n):
            eff = U[i, k] / U[k, k]
            for j in range(k, n):
                U[i, j] -= eff * U[k, j]
            bs[i] -= eff*bs[k]
    xs = backsub2(U, bs) # calling modified backsub
    return xs

AA = np.array([[2, 1, 1], 
              [2, 1, -4], 
              [1, 2, 1]], float)

b2 = np.array([8, -2, 2], float)
sov = gauelim2(AA, b2)
print(sov)
#%% Question 2 d)
def inverse(U):
    # coding this routine the unflexible way described in the instructions
    x = np.array([1, 0, 0], float)
    y = np.array([0, 1, 0], float)
    z = np.array([0, 0, 1], float)
    
    sol1, sol2, sol3 = gauelim2(U, x), gauelim2(U, y), gauelim2(U, z)
    # making arrays to columns
    sol11 = np.array([sol1]).T
    sol22 = np.array([sol2]).T
    sol33 = np.array([sol3]).T
    # making columns to a matrix
    A = np.concatenate((sol11, sol22, sol33), axis = 1)
    return A

C = np.array([[1, 2, 3], 
              [0, 1, 4],
              [5, 6, 0]], float)

print("Inverse matrix of C: \n",inverse(C))
print("Checking solution by multiplying with invC with C: \n", np.matmul(inverse(C), C))
print("Checking with numpy: \n", np.linalg.inv(C))
    