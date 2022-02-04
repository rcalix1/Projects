######################################################
## 2021

import numpy as np
import nltk
import re



######################################################

f_in  = open("in/PulmCorpus.txt", "r")
f_out = open("out/PulmCorpus.txt", "w")


ii = 0
for line in f_in.readlines():
    line = line.replace("\n","")
    if line == "":
        continue
    if len(line) < 3:
        continue
    temp = line.split(" ")
    if len(temp) == 0:
        continue
    sent = ' '.join(temp)
    string = sent.lower()

    #string = re.sub(r'[^\w\s]', '', string)   ## remove punctuation characters
    #string = re.sub(r'\b[0-9]+\b\s*', '', string)    ## remove numbers
    
    f_out.write(string + "\n")
  
    ii = ii + 1


f_in.close()
f_out.close()

######################################################

print("<<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>>")
