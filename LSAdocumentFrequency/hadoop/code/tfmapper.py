#!/usr/bin/env python
import sys
import os

def tfmapper():
  for line in sys.stdin:
    words = line.strip().split()
    for word in words:
       print "%s\t%s\t1" % (word,os.getenv('mapreduce_map_input_file','noname'))
#      print "%s\t%s\t1" % (word,"1")

if __name__=='__main__':
  tfmapper()
