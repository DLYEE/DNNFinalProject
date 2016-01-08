import time
import re

def readGlove(f, lines):
    count = 0
    start = time.time()
    dictionary = {}
    file = open(f,"r+")
    # To generate smaller data
    # wfile = open("small_glove_10000.txt","w")
    # wordfile = open("index_10000.txt","w")
    # change open(f) to range N
    for line in open(f):
        if count > lines:
            break
        count += 1
        if count % 50000 == 0:
            print "reading Glove",count,"th lines..."
        line = file.readline()
        # wfile.write(line)
        s = re.split(" |\n",line)
        s.pop()
        s[1:] = [float(x) for x in s[1:]]  
        dictionary[s[0]] = s[1:]
        # wordfile.write(s[0] + '\n')
    end = time.time()
    print "Cost ", (end - start), "second"
    return dictionary

def specialword(vec, dictionary, word):
    if(word[-1] == '.' or word[-1] == '?' or word[-1] == ',' or 
       word[-1] == '\'' or word[-1] == ':'):
        tail = word[-1]
        word = word[:-1]
        value = dictionary.get(word)
        if value != None:
            vec.append(dictionary[word])
        # else:
            # print "Cannot find ",word, " in dictionary."
        vec.append(dictionary[tail])
    elif(word[-2:] == "\'s"):
        tail = word[-2:]
        value = dictionary.get(word[:-2])
        if value != None:
            vec.append(dictionary[word[:-2]])
        # else:
            # print "Cannot find ",word, " in dictionary."
        vec.append(dictionary[word[-2:]])
    elif(word[-3:] == "n\'t" or word[-3:] == "\'re" or word[-3:] == "\'ve"):
        tail = word[-3:]
        value = dictionary.get(word[:-3])
        if value != None:
            vec.append(dictionary[word[:-3]])
        # else:
            # print "Cannot find ",word, " in dictionary."
        vec.append(dictionary[word[-3:]])
    else:
        return
        # print "Cannot find ",word, " in dictionary."

