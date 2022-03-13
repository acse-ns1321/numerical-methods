#  𝑉.𝑇𝑉𝒂=𝑉.𝑇𝒚 


# Some random data
X = np.array([0.5, 2.0, 4.0, 5.0, 7.0, 9.0])
y = np.array([0.5, 0.4, 0.3, 0.1, 0.9, 0.8])

# Consider a polynomial of degree 3 - so not high enough to go through all the data
# (unless we're in the unlikely case where all the data happens to lie on a cubic!)
N = 3

# Use a numpy function to construct the Vandermonde matrix
V = np.vander(X, N+1, increasing=True)

# Form the matrix A by transposing V and multiplying by V:
A = V.T @ V  # same as A = np.transpose(V) @ V

# Use a function from SciPy's linalg sub-package to find the inverse:
invA = sl.inv(A)

# Form the RHS vector:
rhs = V.T @ y

# Multipy through by the inverse matrix to find a:
a = invA @ rhs
print('a = \n', a)

# Compare against the coefficient that numpy's polyfit gives us
poly_coeffs = np.polyfit(X, y, N)
print('\npoly_coeffs = \n', poly_coeffs)
# they're the same, we just set up our algorithm to return the coefficient in the 
# opposite order to polyfit - we need to remember this when we evaluate the polynomial

print('\nOur a vector = output from np.polyfit (as long as we flip the order)? ', 
      np.allclose(np.flip(a), poly_coeffs))
# set up figure
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111)
ax1.margins(0.1)

XX = np.linspace(0.4, 9.1, 100)
yy = a[0] + a[1] * XX + a[2] * XX**2 + a[3] * XX**3

ax1.plot(XX, yy, 'b', label='Least squares fit (cubic)')

# Overlay raw data
ax1.plot(X, y, 'ko', label='Raw data')
ax1.set_xlabel('$X$', fontsize=16)
ax1.set_ylabel('$y$', fontsize=16)
ax1.set_title('Least squares approximation of a cubic to multiple data points', fontsize=16)
ax1.grid(True)
ax1.legend(loc='best', fontsize=14);