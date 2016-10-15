import sys
import random

f = open(sys.argv[1], 'r')
for line in f:
    br = line.strip().split('\t')
    label = br[0]
    n = random.random();
    #if label == '1':
    #    print line.strip()
    #    continue
    if n < 0.1:
        print line.strip()

