def chord_method ( F , m , x_0 , atol = 1.e-5 , maxiter=100 ):
    x_n = []
    y_n = []
    x = x_0
    # iterate until we hit break either as we hit tolerance or maximum number iterations
    # since we include the initial guess, the max. number of entries is maxiter+1
    for i in range ( maxiter+1 ):
        x_n . append ( x )
        y_n . append ( F ( x ))
    # Chord update:
    x = x - m * F ( x )
    if abs ( x - x_n [ - 1 ]) < atol :
        break
    return x_n , y_n
# note that with m=1, this gives G(x) = np.tan(x) - .1 as before
def F ( x ):
    return x - np . tan ( x ) + .1
fig , ( ax1 , ax2 ) = plt . subplots ( 1 , 2 , figsize= ( 16 , 6 ))
fig . subplots_adjust ( wspace=0.3 )
ax1 . axis ( 'equal' )
x0 = 0.6
m = - 2.5
x_n , y_n = chord_method ( F , m , x0 , atol=1.e-2 , maxiter=4 )
plot_iteration ( ax1 , x_n , y_n , F )
plot_triangle ( ax1 , x_n , y_n , m )
ax1 . set_title ( "Chord method iteration" , fontsize=20 )
print ( 'The slope marker contains the value 1/m = ' , 1. / m )
m = - 1.
ax2 . axis ( 'equal' )
x_n , y_n = chord_method ( F , m , x0 , atol=1.e-3 , maxiter=4 )
plot_iteration ( ax2 , x_n , y_n , F )
plot_triangle ( ax2 , x_n , y_n , m )
ax2 . set_title ( "Chord method iteration" , fontsize=20 )
print ( 'The slope marker contains the value 1/m = ' , 1. / m )