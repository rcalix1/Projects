#!/usr/bin/env python
import sys
import os
def dfmapper():
  for line in sys.stdin:
    print "%s\t1" % line.strip()
if __name__ == '__main__':
  dfmapper()


