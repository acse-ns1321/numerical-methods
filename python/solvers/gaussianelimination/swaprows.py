# This function swaps rows in matrix A
# (and remember that we need to do likewise for the vector b 
# we are performing the same operations on)

def swap_row(A, b, i, j):
    """ Swap rows i and j of the matrix A and the vector b.
    """ 
    if i == j:
        return
    print('swapping rows', i,'and', j)
    # If we are swapping two values, we need to take a copy of one of them first otherwise
    # we will lose it when we make the first swap and not be able to use it for the second.
    # We need to make sure it is a real copy - not just a copy of a reference to the data!
    # use np.copy to do this. 
    iA = np.copy(A[i, :])
    ib = np.copy(b[i])

    A[i, :] = A[j, :]
    b[i] = b[j]

    A[j, :] = iA
    b[j] = ib