#!/usr/bin/env python
import sys
def tfreducer():
  curprefix = None
  curcount = None
  for line in sys.stdin:
    word,filename,count = line.strip().split('\t')
    prefix = '%s\t%s' % (word,filename)
    if curprefix == None:
      curprefix = prefix
      curcount = eval(count)
    elif curprefix == prefix:
      curcount += eval(count)
    else:
      print "%s\t%s" % (curprefix,curcount)
      curprefix = prefix
      curcount = eval(count)
  print "%s\t%s" % (curprefix,curcount)

if __name__=='__main__':
  tfreducer()
