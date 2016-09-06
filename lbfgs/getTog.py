#!/usr/bin/env python

import sys

def getT(filename):
    fl = open('label.txt', 'r')
    fp = open(filename, 'r')
    for line in fl:
        line = line.strip()
        line2 = fp.readline().strip()
        print '%s,%s' % (line2, line)


if __name__ == "__main__":
    filename = sys.argv[1]
    getT(filename)
