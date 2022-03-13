
from matplotlib import pyplot as plt
import numpy as np


A_i = np . linalg . pinv ( A )
x_m = A_i@b

# define a mesh of multipliers of the two null vectors
dm = 0.01
m1 = np . arange ( - 1.0 , 1.0 , dm )
m2 = np . arange ( - 1.0 , 1.0 , dm )
M1 , M2 = np . meshgrid ( m1 , m2 )

# compute the norms of the possible solutions
norms = np . zeros_like ( M1 )
for i , m1i in enumerate ( m1 ):
    for j , m2j in enumerate ( m2 ):
        norms [ i , j ] = np . linalg . norm ( x_m + m1i*n1 + m2j*n2 )
fig = plt . figure ( figsize= ( 5 , 5 ))
ax1 = fig . add_subplot ( 111 )
cs = ax1 . contour ( M1 , M2 , norms , 10 )
ax1 . clabel ( cs , inline=1 , fontsize=10 )
ax1 . set_title ( 'Contour plot of norms of linear system solutions' )
ax1 . set_xlabel ( '$d_1$' , fontsize=12 )
ax1 . set_ylabel ( '$d_2$' , fontsize=12 )

# add a point at (0,0)
ax1 . plot ( 0 , 0 , 'ro' )



# RIGHT INVERSE MINIMAL NORM-------------------------------------------------------------------
# come up with an example with indpt equations
A = np . array ([
[ 1 , 0 , 4 , 0 ],
[ 0 , 1 , - 2 , 0 ],
[ 0 , 0 , 0 , 1 ]])
b = np . array ([ 1 , - 2 , 3 ])
# construct the right inverse:
A_ri = A . T @ sl . inv ( A @ A . T )
pprint ( A_ri )
# print it to check its the identity (to round off error)
pprint ( A @ A_ri )
x_m = A_ri @ b
print ( 'Min norm solution: ' , x_m )
# check that this is a solution: Ax = b?
print ( np . allclose ( b , A @ x_m ))