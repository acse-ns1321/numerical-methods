# Solve the systems with Steepest Descent  using a relative  tolerance of 1e-5.

import numpy as np
import scipy.linalg as sl

def steepest_descent ( A , b , rtol=1e-5 ):
    """Simple Steepest Descent method with zero initial guess and relative tolerance"""
    x = np . zeros_like ( b ) # start with zero initial guess
    r = b - A @ x
    r0 = sl . norm ( r ) # norm of initial residual
    it = 0
    while sl . norm ( r ) >rtol*r0 :
        alpha_star = np . dot ( r , r ) / np . dot ( r , A @ r )
        x = x + alpha_star*r
        r = b - A @ x
        it += 1
    return x , it