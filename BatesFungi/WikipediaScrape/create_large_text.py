import os
import json

######################################

f_write = open('output/all_wikipedia_text_combined.txt', 'a')


######################################

path = 'wikipedia_json_files/'

i = 0
errors = 0

for filename in os.listdir(path):
    pathname = os.path.join(path, filename)
    #print(pathname)
    #print(i)
    i = i + 1

    f = open(pathname)

    # returns JSON object as
    # a dictionary
    try:
        data = json.load(f)
        dict_data = data[0]
        #print(  dict_data['text']   )
        f_write.write(dict_data['text'])
    except:
        print("errors", errors)
        print(pathname)
        errors = errors + 1

    f.close()


print("json files processed ", i)
print("errors ", errors)

########################################

f_write.close()

########################################


print("<<<<<<<<<<<<<<DONE>>>>>>>>>>>>>>>")
