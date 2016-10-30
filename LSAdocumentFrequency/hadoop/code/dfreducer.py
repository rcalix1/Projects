#!/usr/bin/env python
import sys
def dfreducer():
  curword = None
  curcount = None
  space = []
  for line in sys.stdin:
    word,filename,wordcount,count = line.strip().split()
    prefix = "%s\t%s\t%s" %(word,filename,wordcount)
    if word == None:
      curword = word
      curcount = eval(count)
      space.append(prefix)
    elif curword == word:
      curcount += eval(count)
      space.append(prefix)
    else:
      for item in space:
        print "%s\t%d" % (item,curcount)
      curword = word
      curcount = eval(count)
      space = [prefix]
  for item in space:
    print "%s\t%d" % (item,curcount)
if __name__=='__main__':
  dfreducer()


