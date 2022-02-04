import os
## import gensim
import json

######################################
''' 
# Opening JSON file
f = open('json_files.jan.2022/PMC1866182.json')


# returns JSON object as
# a dictionary
data = json.load(f)

dict_data = data[0]

print(dict_data)

f.close()

print(  dict_data['text']   )
print(  dict_data.keys()   )

'''

######################################

f_write = open('output/all_text_combined.txt', 'a')





######################################

path = 'json_files.jan.2022/'

i = 0
errors = 0

for filename in os.listdir(path):
    pathname = os.path.join(path, filename)
    print(pathname)
    print(i)
    i = i + 1

    f = open(pathname)

    # returns JSON object as
    # a dictionary
    try:
        data = json.load(f)
        dict_data = data[0]
        print(  dict_data['text']   )
        f_write.write(dict_data['text'])
    except:
        print("errors", errors)
        errors = errors + 1

    f.close()


print(i)
print("errors ", errors)

########################################

f_write.close()

########################################


print("<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>")
