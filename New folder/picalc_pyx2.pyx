import time
from math import sqrt

def main( int nmax ):

    cdef:
        double pibyfour = 0.0
        double dx = 1.0 / nmax
        int i

    initial = time.time()
    
    for i in range(nmax):
        pibyfour += sqrt(1-(i*dx)**2)
    
    final = time.time()
    
    print("Elapsed time: {:8.6f} s".format(final-initial))

    pi = 4.0 * pibyfour * dx
    print("Pi = {:18.16f}".format(pi))
    
    return 0

