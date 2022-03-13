# create a slighty challenging matrix and random rhs
n = 100
np . random . seed ( 0 )
A = np . diagflat ( np . random . random ( n )) + 10 * np . random . random (( n , n )) / n
b = np . random . random ( n )
# for scipy's gmres we need a slightly different residual monitor (than ResidualMonitor we used for scipy's cg
# the callback is not called with the current guess x^k every iteration (as this is not always avalaible)
# instead it only provides the norm of the current residual, so we simply add it to the list
class GMRESResidualMonitor :
def __init__ ( self ):
self . residuals = []
def __call__ ( self , rk ):
self . residuals . append ( rk )
# without restarts gmres should converge in n iterations, so setting the restart value to n
# it should not have done any restarts
monitor1 = GMRESResidualMonitor ()
x , info = spl . gmres ( A , b , callback=monitor1 , tol=1e-10 , restart=n )
# now with a restart of 60
monitor2 = GMRESResidualMonitor ()
x , info = spl . gmres ( A , b , callback=monitor2 , tol=1e-10 , restart=60 )
# and the default restart of 30
monitor3 = GMRESResidualMonitor ()
x , info = spl . gmres ( A , b , callback=monitor3 , tol=1e-10 , restart=30 )
fig , ax = plt . subplots ( 1 , 1 , figsize= ( 16 , 8 ))
fig . subplots_adjust ( hspace=0.4 )
ax . semilogy ( monitor1 . residuals , label= 'GMRES - no restart' )
ax . semilogy ( monitor2 . residuals , label= 'GMRES - restart=60' )
ax . semilogy ( monitor3 . residuals , label= 'GMRES - restart=30' )
ax . set_xlabel ( 'iterations' )
ax . set_ylabel ( 'residual' )
ax . grid ()
ax . set_xticks ( np . arange ( 7 ) *60 )
ax . axis ([ 0. , 420 , 1e-10 , 1 ])
ax . legend ();