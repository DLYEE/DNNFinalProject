import time
import re
import random

def readGlove(f, lines):
    count = 0
    start = time.time()
    dictionary = {}
    file = open(f,"r+")
    # To generate smaller data
    # wfile = open("small_glove_10000.txt","w")
    # wordfile = open("index_10000.txt","w")
    # change open(f) to range N
    count = 0
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
        # if count < 10:
            # print dictionary[s[0]]
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
        random_list = random.sample(range(1000000), 300)
        # print random_list
        for i in range(300):
            random_list[i] = float(random_list[i] - 500000) / float(500000)
        # print random_list
        vec.append(random_list)
        # print "Cannot find ",word, " in dictionary."

def read_answer_txt(file):
    print "Reading answer_txt..."
    t_start = time.time()
    with open(file,'r+') as answers_file:
        answer_txt = []
        for line in answers_file:
            answer_info = re.split(" |\n", line)
            answer_txt.append(answer_info)
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return answer_txt

def read_choices_txt(file):
    print "Reading choices_txt..."
    t_start = time.time()
    with open(file,'r+') as choices_file:
        choices_txt = []
        for line in choices_file:
            choices_info = re.split(" |\n", line)
            choices_txt.append(choices_info)
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return choices_txt

