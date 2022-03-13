import numpy as np
import scipy.optimisation
# UK:
years = np.array([
    1700, 1730, 1750, 1790, 1800, 1810, 1820, 1821, 1822, 1823, 1824, 1825, 1826, 
    1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835, 1836, 1837, 1838, 1839,
    1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851, 1852,
    1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865,
    1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878,
    1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891,
    1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904,
    1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917,
    1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930,
    1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943,
    1944, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956,
    1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
    1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
    1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
    1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
])
population = np.array([
    7200607, 7547572, 8221188, 10955459, 12327466, 13983890, 16186397, 16427955,
    16661282, 16897898, 17139513, 17381418, 17629610, 17858803, 18093284, 18329053,
    18569823, 18811881, 19028016, 19246151, 19468285, 19691420, 19918844, 20136267,
    20354979, 20578690, 20804402, 21027114, 21284460, 21504941, 21751903, 21994095,
    22231076, 22279863, 22246880, 22266109, 22287069, 22310067, 22567009, 22837200,
    23117738, 23417123, 23728007, 23974430, 24230353, 24488987, 24746160, 25009641,
    25291179, 25561448, 25828236, 26107967, 26385197, 26667813, 26955755, 27249851,
    27544197, 27844139, 28050928, 28385947, 28730581, 29082812, 29444523, 29814831,
    30173677, 30557062, 30920888, 31273446, 31576465, 31875253, 32184118, 32500733,
    32822214, 33142079, 33464925, 33793233, 34127982, 34472021, 34836751, 35211963,
    35593770, 35976405, 36367212, 36763751, 37166289, 37569674, 37975154, 38373808,
    38737346, 39102443, 39474558, 39850962, 40231673, 40614789, 41003923, 41398500,
    41795923, 42150769, 42318019, 42556673, 42965211, 43296057, 43473615, 43573615,
    43529634, 43437404, 43718000, 44072000, 44372000, 44596000, 44915000, 45059000,
    45232000, 45389000, 45578000, 45672000, 45866000, 46074000, 46335000, 46520000,
    46666000, 46868000, 47081000, 47289000, 47494000, 47991000, 48226000, 48216000,
    48400000, 48789000, 49016000, 49182000, 49217000, 49519000, 50014000, 50312000,
    50616016, 50621416, 50686056, 50797272, 50945400, 51123708, 51328660, 51559672,
    51818620, 52108972, 52433156, 52789816, 53171324, 53562804, 53945020, 54303104,
    54630988, 54928528, 55194456, 55429644, 55634932, 55809752, 55953320, 56066828,
    56152180, 56211944, 56247408, 56262020, 56263660, 56262128, 56265476, 56276316,
    56296240, 56330880, 56386228, 56466132, 56574280, 56709208, 56862896, 57023748,
    57183332, 57339444, 57494540, 57650472, 57810768, 57978320, 58156748, 58346672,
    58544936, 58746728, 58950848, 59149344, 59348952, 59580224, 59884128, 60286752,
    60802800, 61414660, 62076224, 62722604, 63306844, 63811884, 64250328, 64641108,
    65015688, 65397080, 65788574,
])
# use data defined in previous cell (rescaled)
X = (years-years[0])/(years[-1]-years[0])
Y = population/population[0]

def R(x):  # the residual function
    C1, C2 = x
    return C1*np.exp(C2*X) - Y

def dRdx(x):  # the first derivative of the residual
    C1, C2 = x
    # [:, np.newaxis] turns a flat rank-1 array of lenght m into a mx1 column array
    # with np.hstack we can stack those horizontally
    return np.hstack((np.exp(C2*X)[:,np.newaxis], (C1*X*np.exp(C2*X))[:,np.newaxis]))

def dRdx2(x):  # the second derivative of the residual
    C1, C2 = x
    m = len(X)
    # first we collect the 2x2=4 entries for each of the m data points
    # into a m x 4 array
    ans = np.hstack((
         np.zeros((m,1)),
         (X*np.exp(C2*X))[:,np.newaxis],
         (X*np.exp(C2*X))[:,np.newaxis],
         (C1*X**2*np.exp(C2*X))[:,np.newaxis]
    ))
    # then we reshape it to m x 2 x 2
    return ans.reshape((m, 2, 2))


def f(x):  # the least squares fit that we're trying to minimize
    return 0.5 * np.sum(R(x)**2)


def F(x):  # this is the 1st derivative of f: F(x) = f'(x)
    return np.dot(R(x), dRdx(x))


def full_hessian(x):  # 2nd derivative (Hessian) of f
    Rprime = dRdx(x)
    return np.tensordot(Rprime, Rprime, axes=(0,0)) + np.tensordot(
            R(x), dRdx2(x), axes=(0,0))


def gn_hessian(x):  # approximation to the Hessian according to the Gauss-Newton method
    Rprime = dRdx(x)
    return np.tensordot(Rprime, Rprime, axes=(0,0))


x0 = [0, 0]
result = sop.minimize(f, x0, jac=F, hess=full_hessian, method='trust-ncg')
print("Newton method")
print("""Succesful: {success}
Values for C1, C2: {x}
Number of iterations: {nit}
Number of function, Jacobian and Hessian evaluations: {nfev}, {njev}, {nhev}
""".format(**result))

result = sop.minimize(f, x0, jac=F, hess=gn_hessian, method='trust-ncg')
print("Gauss-Newton method")
print("""Succesful: {success}
Values for C1, C2: {x}
Number of iterations: {nit}
Number of function, Jacobian and Hessian evaluations: {nfev}, {njev}, {nhev}
""".format(**result))

plt.figure(figsize=(12,6))
plt.plot(years, Y, '.', label='Population UK (indexed at 1700)')
C1, C2 = result['x']
plt.plot(years, C1*np.exp(C2*X), label='Exponential Fit')
plt.yticks(np.arange(11))
plt.legend();


# For both cases we have here used a trust-region method, 
# as neither the Hessian nor the approximate Gauss-Newton Hessian are guaranteed SPD,
#  as we can see by computing the eigenvalues:

# compute eigenvalues of the full and Gauss_Newton Hessians
# if these are SPD then their eigenvalues should be strictly positive

w,v = sl.eigh(full_hessian(x0))
print("Full Hessian eigenvalue at x0:", w)

w,v = sl.eigh(gn_hessian(x0))
print("Gauss-Newton Hessian eigenvalue at x0:", w)

# In scipy.optimisation the least_squares function provides various methods, 
# where the lm method refers to Levenberg-Marquard (Gauss-Newton with trust region approach, see next section). 
# Here we only need to provide the residual function  𝑅(𝐱)  and its derivative  𝑅′(𝐱) .

x0 = [0,0]
result = sop.least_squares(R, x0, jac=dRdx, method='lm')
print("Levenberg-Marquard (Trust Region Gauss-Newton)")
print("""Succesful: {success}
Values for C1, C2: {x}
Number of function and Jacobian evaluations: {nfev}, {njev}
""".format(**result))