from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse.linalg as spl
import scipy.linalg as sl
import scipy.sparse as sp  
# in this section we will investigate a number of stationary methods of he form x^k+1 = x^k + P(r^k)
# where P(r^k) is a linear operator which we will call the preconditioner (for reasons to become clear later)
# this P is an approximate inverse of the matrix A, and there is usually some one-off setup work to be done
# before P can be called - we will therefore use the following base structure for our "preconditioners"
class Preconditioner ( spl . LinearOperator ):
    def __init__ ( self , A ):
     super () . __init__ ( dtype=None , shape=A . shape )
    def __call__ ( self , r ):
     raise NotImplemented ( "The call method should be overloaded" )
    def _matvec ( self , r ):
     return self ( r )

class JacobiPreconditioner ( Preconditioner ):
    def __init__ ( self , A ):
        super () . __init__ ( A )
        self . diag = A . diagonal () # extract main diagonal
    def __call__ ( self , r ):
       return r / self . diag

# using these we can implemented a main loop (the iteration) that will be the same for every stationary method
def stationary_method ( A , b , P , rtol=1e-5 ):
    """Generic stationary method: x^k+1 = x^k + P(r^k)"""
    x = np . zeros_like ( b ) # start with zero initial guess
    r = b - A @ x
    r0 = sl . norm ( r ) # norm of initial residual
    it = 0
    while sl . norm ( r ) > rtol*r0 :
        x = x + P ( r )
        r = b - A @ x
        it += 1
    return x , it
nx = 20
A , b = A , b = Poisson_2D_Dirichlet_FD_sparse ( nx , nx , RHS_f , DBC )
#A, b = Poisson_2D_Dirichlet_FE(tri['vertices'], tri['triangles'], tri['vertex_markers'], DBC=DBC_simple)
x , its_sd = steepest_descent ( A , b )
P = JacobiPreconditioner ( A )
x , its_jac = stationary_method ( A , b , P )
counter = SimpleCounter ()
x , info_cg = spl . cg ( A , b , tol=1e-5 , callback=counter )
print ( "Steepest Descent: {} iterations" . format ( its_sd ))
print ( "Jacobi method: {} iterations" . format ( its_jac ))
print ( "Conjugate Gradient: {} iterations" . format ( counter . count ))

# Gauss Siedel Prcoditioner

class GaussSeidelPreconditioner ( Preconditioner ):
    """Gauss Seidel method: M=L+omega*D"""
    def __init__ ( self , A , omega=1.0 ):
        super () . __init__ ( A )
        # extract M = D + L
        self . M = sp . tril ( A ) . tocsr ()
        # for future use: if omega/=1, we use M = D + omega*L
        if not omega==1.0 :
            # do this by adding M = (D+L) + (1-omega)*L
            # tril(A) returns D+L
            # tril(A, -1) returns L only
            self.M += ( omega - 1 ) *sp.tril ( A , - 1 )
    def __call__ ( self , r ):
     return spl . spsolve_triangular ( self . M , r )
x , its_gs = stationary_method ( A , b , GaussSeidelPreconditioner ( A ))
print ( "Gauss Seidel: {} iterations" . format ( its_gs ))

# Successive Over Relaxation

omegas = np . arange ( 0 , 2 , .05 )
its_sor = []
for omega in omegas :
x , it = stationary_method ( A , b , GaussSeidelPreconditioner ( A , omega=omega ))
its_sor . append ( it )
print ( "For omegas:" , omegas )
print ( "Required SOR iterations:" , its_sor )
plt . plot ( omegas , its_sor , '.-' )
plt . xlabel ( 'omega $\omega$' )
plt . ylabel ( 'Iterations' );



class SSORPreconditioner ( Preconditioner ):
    """SSOR method"""
    def __init__ ( self , A , omega=1.0 ):
        super(). __init__ ( A )
        self . DOL = sp . tril ( A ) # DOL = D + L
        self . DOU = sp . triu ( A ) # DOU = D + U
        if not omega==1.0 :
            # add (omega-1)*L and (omega-1)*U resp.
            # see comments in GaussSeidelPreconditioner
            self . DOL += ( omega - 1.0 ) *sp . tril ( A , - 1 )
            self . DOU += ( omega - 1.0 ) *sp . triu ( A , 1 )
            # every iteration we need to compute M^-1 r
        # from the definition of M above we get
        # M^-1 = omega*(2-omega) * DOU^-1 * D * DOL^-1
        # (note the order is swapped)
        # we combine the steps of multiplying with the diagonal matrix M
        # and multiplying with the scalar omega*(2-omega) by scaling the diagonal:
        self . diag = A . diagonal () * ( omega* ( 2 - omega ))
        def __call__ ( self , r ):
            # 1) do a lower triangular solve with DOL
            # 2) multiply with self.diag = diagonal(A)*omega*(2-omega)
            # 3) do a upper triangulaer solve with DOU using spsolve_triangular(..., lower=False)
            return spl . spsolve_triangular ( self . DOU , self . diag * spl . spsolve_triangular ( self . DOL , r ), lower=False )
x , its_gs = stationary_method ( A , b , SSORPreconditioner ( A , omega=1.7 ))
print ( "SSOR: {} iterations" . format ( its_gs ))

# NOTE: omitting some smaller values of omega which give excessive iteration counts
omegas = np . arange ( 0.3 , 2 , .05 )
its_ssor = []
for omega in omegas :
    x , it = stationary_method ( A , b , SSORPreconditioner ( A , omega=omega ))
    its_ssor . append ( it )
print ( "For omegas:" , omegas )
print ( "Required SSOR iterations:" , its_ssor )
plt.plot ( omegas , its_ssor , '.-' )
plt .xlabel ( 'omega $\omega$' )
plt. ylabel ( 'Iterations' );

class ILUPreconditioner ( Preconditioner ):
    """ILU preconditioner - wraps around spilu"""
    def __init__ ( self , A , fill_factor=1.0 ):
        super () . __init__ ( A )
        self . Minv = spl . spilu ( A . tocsc(), fill_factor=fill_factor ) #, fill_factor=fill_factor, drop_tol=1e-4, o
    def __call__ ( self , r ):
      return self . Minv . solve ( r )

x , its_ilu0 = stationary_method ( A , b , ILUPreconditioner ( A ))
x , its_ilu_ff2 = stationary_method ( A , b , ILUPreconditioner ( A , fill_factor=2.0 ))
print ( "ILU(0) method: {} iterations" . format ( its_ilu0 ))
print ( "ILU fill_factor=2: {} iterations" . format ( its_ilu_ff2 ))


import pyamg
class AMGPreconditioner ( Preconditioner ):
    """AMG preconditioner - wraps around pyamg's ruge_stuben_solver"""
    def __init__ ( self , A ):
        super () . __init__ ( A )
        ml = pyamg . ruge_stuben_solver ( A . tocsr (), strength= ( 'classical' ))
        self . gamg = ml . aspreconditioner ()
    def __call__ ( self , r ):
        return self . gamg ( r )
nx = 20
amg = AMGPreconditioner ( A )
x , its_amg = stationary_method ( A , b , amg )
print ( "AMG method: {} iterations" . format ( its_amg ))