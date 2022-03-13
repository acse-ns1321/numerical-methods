def newton_method(F, jac, x_0, atol = 1.e-5, maxiter=100):
    x_n = []
    y_n = []
    x = x_0
    
    # iterate until we hit break either as we hit tolerance or maximum number iterations
    # since we include the initial guess, the max. number of entries is maxiter+1
    for i in range(maxiter+1):
        x_n.append(x)
        Fx = F(x)
        y_n.append(Fx)
        # Newton update:
        x = x - Fx/jac(x)
        if abs(x - x_n[-1]) < atol:
            break
    
    return x_n, y_n


def F(x):
    return np.arctan(x)


def dFdx(x):
    return 1/(1+x**2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
x0 = 1.0
x_n, y_n = newton_method(F, dFdx, x0, atol=1.e-2, maxiter=4)
plot_iteration(ax1, x_n, y_n, F)
ax1.set_title("Newton method iteration", fontsize=20)

x0 = 2.0
x_n, y_n = newton_method(F, dFdx, x0, atol=1.e-2, maxiter=2)
plot_iteration(ax2, x_n, y_n, F)
ax2.set_title("Newton method iteration", fontsize=20)

# plot the exact same Newton iteration along with the primitive of F
# i.e. df(x)/dx = F(x) - so that we treat it as a minimisation of f
def f(x):
    return x*np.arctan(x) - np.log(x**2 + 1)/2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
x0 = 1.0
x_n, y_n = newton_method(F, dFdx, x0, atol=1.e-2, maxiter=4)
f_n = [f(x) for x in x_n]
plot_iteration(ax1, x_n, f_n, f, include_chords=False)
ax1.set_ylabel('$y=f(x)$')
ax1.set_title("Newton method iteration", fontsize=20)

x0 = 2.0
x_n, y_n = newton_method(F, dFdx, x0, atol=1.e-2, maxiter=2)
f_n = [f(x) for x in x_n]
plot_iteration(ax2, x_n, f_n, f, include_chords=False)
ax2.set_ylabel('$y=f(x)$')
ax2.set_title("Newton method iteration", fontsize=20)