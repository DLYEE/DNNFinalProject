import re
from num2words import num2words
'''
def number(string, word):
    string.remove(word)
    for i in range(word):
        if (word[i].isnumeric()):
            s = chr(int(word[i])+)
'''
def checkWords(string, i):
    word = string[i]
    if len(word) > 1:
        if(word[-1] == '\"' or word[-1] == '\'' ):
            string.remove(word)
            word = word[:-1]
            string.insert(i, word)
    if len(word) > 1:
        if(word[0] == '\"' or word[-1] == '\'' ):
            string.remove(word)
            word = word[1:]
            string.insert(i, word)
    if len(word) > 2:
        if (word[-2] == '\"'):
            string.remove(word)
            word = word[:-2]
            string.insert(i, word)

    for j in range(len(word)):
        if (word[j] == "-"):
            first = word[:j]
            second = word[j+1:]
            string.remove(word)
            string.insert(i,second)
            string.insert(i,first)
            break
    
def stopWord (f, wf):
    dictionary = {}
    for line in open("stop_word.txt", "r+"):
        s = re.split(" |\n",line)
        dictionary[s[0]] = 1 
    writeFile = open(wf,'w+')
    for line in open(f,"r+"):
        line = line.lower()
        s = re.split(" |\n",line)
        for i in range(3):
            if s[-1] == "":
                s.pop()
        i = 0
        s[-1] = s[-1][:-1]
        while (i < (len(s[2:]))):
            checkWords(s, i+2)
            if (dictionary.get(s[i+2]) != None):
                s.remove(s[i+2])
            else:
                i += 1
        for i in range(len(s)):
            writeFile.write(s[i])
            if i != len(s)-1:
                writeFile.write(" ")
        writeFile.write("\n")


# stopWord ("sequense.txt", "seq.txt")
stopWord ("data/vec_100/question_word.test", "question_word.test")

