from cython.parallel cimport parallel
cimport openmp

def main():

    cdef:
        int thread

    threads = 8
    print("Threads set to {:2d}".format(threads))

    with nogil, parallel(num_threads=threads):
        thread = openmp.omp_get_thread_num()
        with gil:
            print("Hello from thread {:2d}".format(thread))

    return 0

