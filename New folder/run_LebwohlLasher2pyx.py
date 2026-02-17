import sys
from LebwohlLasher2 import main

if int(len(sys.argv)) == 4:
    main(int(sys.argv[1]))
else:
    print("Usage: {} <ITERATIONS>".format(sys.argv[0]))