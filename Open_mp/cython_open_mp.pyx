import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
cimport numpy as np
from libc.math cimport cos, sin, exp, M_PI
from cython.parallel cimport prange, parallel
from libc.stdlib cimport rand, RAND_MAX, malloc, free

cdef extern from *:
    """
    int omp_get_max_threads(void) {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }
    
    int omp_get_thread_num(void) {
        #ifdef _OPENMP
        return omp_get_thread_num();
        #else
        return 0;
        #endif
    }
    """
    int omp_get_max_threads()
    int omp_get_thread_num()

def initdat(nmax):
    return np.random.random_sample((nmax, nmax)) * 2.0 * np.pi

def plotdat(arr, pflag, nmax):
    if pflag == 0:
        return

    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax, nmax))

    if pflag == 1:
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i, j] = one_energy_py(arr, i, j, nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:
        mpl.rc('image', cmap='hsv')
        cols = arr % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

def savedat(arr, nsteps, Ts, runtime, ratio, energy, order, nmax):
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    
    FileOut = open(filename, "w")
    print("#=====================================================", file=FileOut)
    print("# File created:        {:s}".format(current_datetime), file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax, nmax), file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps), file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts), file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime), file=FileOut)
    print("#=====================================================", file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:", file=FileOut)
    print("#=====================================================", file=FileOut)

    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f}".format(i, ratio[i], energy[i], order[i]), file=FileOut)
    FileOut.close()

cdef double one_energy_c(double* arr, int ix, int iy, int nmax) nogil:
    cdef double en = 0.0
    cdef int ixp = (ix+1) % nmax
    cdef int ixm = (ix-1) % nmax
    cdef int iyp = (iy+1) % nmax
    cdef int iym = (iy-1) % nmax
    cdef double ang, cos_ang
    
    ang = arr[ix * nmax + iy] - arr[ixp * nmax + iy]
    cos_ang = cos(ang)
    en += 0.5 * (1.0 - 3.0 * cos_ang * cos_ang)
    
    ang = arr[ix * nmax + iy] - arr[ixm * nmax + iy]
    cos_ang = cos(ang)
    en += 0.5 * (1.0 - 3.0 * cos_ang * cos_ang)
    
    ang = arr[ix * nmax + iy] - arr[ix * nmax + iyp]
    cos_ang = cos(ang)
    en += 0.5 * (1.0 - 3.0 * cos_ang * cos_ang)
    
    ang = arr[ix * nmax + iy] - arr[ix * nmax + iym]
    cos_ang = cos(ang)
    en += 0.5 * (1.0 - 3.0 * cos_ang * cos_ang)
    
    return en

def one_energy_py(arr, ix, iy, nmax):
    return one_energy_c(<double*>np.PyArray_DATA(arr), ix, iy, nmax)

def all_energy(arr, nmax):
    cdef double enall = 0.0
    cdef int i, j
    cdef double* arr_ptr = <double*>np.PyArray_DATA(arr)
    cdef int N = nmax
    
    with nogil:
        for i in prange(N, schedule='guided'): #shared work fixed data segments
            for j in range(N):
                enall += one_energy_c(arr_ptr, i, j, N)
    
    return enall

def get_order(arr, nmax):
    cdef int i, j, a, b
    cdef double[:, :] Qab = np.zeros((3, 3))
    cdef double[:, :] delta = np.eye(3)
    cdef double* arr_ptr = <double*>np.PyArray_DATA(arr)
    cdef int N = nmax
    
    cdef double* lab = <double*>malloc(3 * N * N * sizeof(double))
    if lab == NULL:
        raise MemoryError()
    
    try:
        with nogil:
            for i in prange(N, schedule='static'): #Shared datasegements across the loop
                for j in range(N):
                    lab[0*N*N + i*N + j] = cos(arr_ptr[i*N + j])
                    lab[1*N*N + i*N + j] = sin(arr_ptr[i*N + j])
                    lab[2*N*N + i*N + j] = 0.0
        
        for a in range(3):
            for b in range(3):
                Qab[a, b] = 0.0
        
        for i in range(N):
            for j in range(N):
                for a in range(3):
                    for b in range(3):
                        Qab[a, b] += 3 * lab[a*N*N + i*N + j] * lab[b*N*N + i*N + j] - delta[a, b]
        
        for a in range(3):
            for b in range(3):
                Qab[a, b] = Qab[a, b] / (2.0 * N * N)
        
        Q_np = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                Q_np[a, b] = Qab[a, b]
        
        eigenvalues, eigenvectors = np.linalg.eig(Q_np)
        return eigenvalues.max()
    
    finally:
        free(lab)

cdef int mc_step_site_c(double* arr, int ix, int iy, double ang, double Ts, int nmax) nogil:
    cdef double en0, en1, boltz, diff
    
    en0 = one_energy_c(arr, ix, iy, nmax)
    arr[ix * nmax + iy] += ang
    en1 = one_energy_c(arr, ix, iy, nmax)
    
    diff = en1 - en0
    
    if diff <= 0.0:
        return 1
    else:
        boltz = exp(-diff / Ts)
        if boltz >= (rand() / (<double>RAND_MAX)):
            return 1
        else:
            arr[ix * nmax + iy] -= ang
            return 0

def MC_step(arr, Ts, nmax):
    cdef double scale = 0.1 + Ts
    cdef int total_accept = 0
    cdef int i, j
    cdef double* arr_ptr = <double*>np.PyArray_DATA(arr)
    cdef int N = nmax
    cdef int* xran_ptr
    cdef int* yran_ptr
    cdef double* aran_ptr
    cdef double Ts_c = Ts
    
    xran_np = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    yran_np = np.random.randint(0, nmax, size=(nmax, nmax), dtype=np.int32)
    aran_np = np.random.normal(scale=scale, size=(nmax, nmax)).astype(np.float64)
    
    xran_ptr = <int*>np.PyArray_DATA(xran_np)
    yran_ptr = <int*>np.PyArray_DATA(yran_np)
    aran_ptr = <double*>np.PyArray_DATA(aran_np)
    
    with nogil: #Not using python
        for i in prange(N, schedule='guided'): # guided large chunks and then it gets smaller chunks after time.
            for j in range(N):
                total_accept += mc_step_site_c(arr_ptr, 
                                               xran_ptr[i*N + j], #vectorised form
                                               yran_ptr[i*N + j], 
                                               aran_ptr[i*N + j], 
                                               Ts_c, N)
    
    return total_accept / (N * N)

def main(program, nsteps, nmax, temp, pflag):
    print(f"Starting simulation...")
    print(f"Number of threads: {omp_get_max_threads()}")
    
    lattice = initdat(nmax)

    if pflag > 0:
        plotdat(lattice, pflag, nmax)
    
    energy = np.zeros(nsteps + 1, dtype=np.float64)
    ratio = np.zeros(nsteps + 1, dtype=np.float64)
    order = np.zeros(nsteps + 1, dtype=np.float64)
    
    energy[0] = all_energy(lattice, nmax)
    ratio[0] = 0.5
    order[0] = get_order(lattice, nmax)
    
    initial = time.time()

    for it in range(1, nsteps + 1):
        ratio[it] = MC_step(lattice, temp, nmax)
        energy[it] = all_energy(lattice, nmax)
        order[it] = get_order(lattice, nmax)
        
        if it % max(1, nsteps // 10) == 0:
            print(f"Step {it}/{nsteps} complete")

    final = time.time()
    runtime = final - initial
    
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)

    if pflag > 0:
        plotdat(lattice, pflag, nmax)
    
    print(f"{program}: Size: {nmax}, Steps: {nsteps}, T*: {temp:5.3f}, "
          f"Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")

if __name__ == '__main__':
    if len(sys.argv) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print(f"Usage: python {sys.argv[0]} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>")