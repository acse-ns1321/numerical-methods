


# . Initialize x₀
# 2. Calculate r₀ = Ax₀ − b
# 3. Assign p₀ = −r₀
# 4. For k = 0, 1, 2, …:
#     * calculate αₖ = -rₖ'pₖ / pₖ'Apₖ
#     * update xₖ₊₁ = xₖ + αₖpₖ
#     * calculate rₖ₊₁ = Axₖ₊₁ - b
#     * calculate βₖ₊₁ = rₖ₊₁'Apₖ / pₖ'Apₖ
#     * update pₖ₊₁ = -rₖ₊₁ + βₖ₊₁pₖ

from matplotlib import pyplot as plt
import numpy as np


def conjugate_gradient(A, b, x0):
    r0 = b - A @ x0
    p = r0
    x = x0
    for i in range(A.shape[0]):
        Ap = A @ p
        alpha = np.dot(r0, r0) / np.dot(p, Ap)
        x = x + alpha * p
        r1 = r0 - alpha * Ap
        beta = np.dot(r1, r1)/np.dot(r0, r0)
        p = r1 + beta * p
        r0 = r1
    return x

def conjGrad(A,x,b,tol,N):
    r = b - A.dot(x)
    p = r.copy()
    for i in range(N):
        Ap = A.dot(p)
        alpha = np.dot(p,r)/np.dot(p,Ap)
        x = x + alpha*p
        r = b - A.dot(x)
        if np.sqrt(np.sum((r**2))) < tol:
            print('Itr:', i)
            break
        else:
            beta = -np.dot(r,Ap)/np.dot(p,Ap)
            p = r + beta*p
    return x 

# ORTHOGONAL DIRECTIONS IN THE NEXT ITERATIONS

plt . figure ( figsize= ( 8 , 8 ))
x = np . array ([[ 0 , 0 ], [ 2. / 3 , 2. / 3 ], [ 1 , .5 ]])
plt . plot ( x [:, 0 ], x [:, 1 ], '.k' )
plt . annotate ( r"${\bf x}^{(0)}$" , x [ 0 ], fontsize=16 ,
horizontalalignment= 'right' , verticalalignment= 'top' )
plt . annotate ( r"${\bf x}^{(1)}$" , x [ 1 ], fontsize=16 ,
horizontalalignment= 'center' , verticalalignment= 'bottom' )
plt . annotate ( r"${\bf x}^{(2)}={\bf x}^{*}$" , x [ 2 ], fontsize=16 ,
horizontalalignment= 'left' , verticalalignment= 'top' )
plt . annotate ( "" , x [ 1 ], xytext=x [ 0 ], arrowprops={ 'arrowstyle' : '->' })
plt . annotate ( "" , x [ 2 ], xytext=x [ 1 ], arrowprops={ 'arrowstyle' : '->' })
plt . annotate ( "" , x [ 1 ] + [ 1 , 1 ], xytext=x [ 1 ], arrowprops={ 'arrowstyle' : '->' , 'color' : 'red' })
plt . annotate ( "" , x [ 1 ] + [ 1. / 3 , - 1. / 3 ], xytext=x [ 1 ], arrowprops={ 'arrowstyle' : '->' , 'color' : 'red' })
plt . annotate ( r"${\bf r}^{(0)}$" , x [ 1 ] + [ .5 , .5 ], color= 'red' , fontsize=16 ,
horizontalalignment= 'left' , verticalalignment= 'top' )
plt . annotate ( r"${\bf r}^{(1)}$" , x [ 1 ] + [ 1 // 6 , - 1. / 6 ], color= 'red' , fontsize=16 )
plt . annotate ( "" , x [ 1 ] + [ 1. / 3 , - 1. / 3 ], xytext=x [ 1 ], arrowprops={ 'arrowstyle' : '->' , 'color' : 'red' })
plt . plot ([ - 0.25 , 1.75 ],[ - 0.25 , 1.75 ], '.' , color= 'white' ) # ensure these points are in view after axis equal
plt . axis ( "equal" );