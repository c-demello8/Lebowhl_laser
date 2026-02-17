import sys
from Cython_1 import main

if int(len(sys.argv)) == 5:
    PROGNAME = sys.argv[0]
    ITERATIONS = int(sys.argv[1])
    SIZE = int(sys.argv[2])
    TEMPERATURE = float(sys.argv[3])
    PLOTFLAG = int(sys.argv[4])
    main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG) #Have to use the parameters from the run in here to be able to input values
else:
    print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))


