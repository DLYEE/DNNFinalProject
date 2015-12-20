#put all constants in here

question_types = []
with open('./question_types.txt','r+') as question_types_file:
    for line in question_types_file:
        question_types.append(line.rstrip())
        #line.rstrip(): delete all the whitespaces including newline
