

A = np.array ([
[ 1. , 2. , 5. , - 2. ],
[ - 1. , - 1. , - 3. , 3. ],
[ 2. , 7. , 16. , - 1 ]])

b = np . array ([ 12. , 0. , 60 ])
# use the pseudo inverse to compute a solution
A_i = np . linalg . pinv ( A )
x_m = A_i@b
pprint ( x_m )
# check that this is a solution: Ax = b?
print ( np . allclose ( b , A@x_m ))
