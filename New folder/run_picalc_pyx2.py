import sys
from picalc_pyx2 import main

if int(len(sys.argv)) == 2:
    main(int(sys.argv[1]))
else:
    print("Usage: {} <ITERATIONS>".format(sys.argv[0]))

