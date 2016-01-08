from constants import question_types  
import shlex
import time

def read_imageid(file):
    t_start = time.time()
    print "Reading qid..."
    keyOrder = []
    with open(file,'r+') as questions_file:
        for line in questions_file:
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            question_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if question_info[0].isdigit():
                #save the question type index and the question in the question_dict, which is a dictionary
                #question_info[2][1:-1] => avoid double quotes 
                keyOrder.append(question_info[0])
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return keyOrder

def read_qid(file):
    t_start = time.time()
    print "Reading qid..."
    keyOrder = []
    with open(file,'r+') as questions_file:
        for line in questions_file:
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            question_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if question_info[0].isdigit():
                #save the question type index and the question in the question_dict, which is a dictionary
                #question_info[2][1:-1] => avoid double quotes 
                keyOrder.append(question_info[1])
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return keyOrder

def read_question(file):
    question_dict = {}
    keyOrder = []
    with open(file,'r+') as questions_file:
        for line in questions_file:
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            question_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if question_info[0].isdigit():
                #save the question type index and the question in the question_dict, which is a dictionary
                #question_info[2][1:-1] => avoid double quotes 
                question_dict[question_info[1]] = [find_question_type(question_info[2]), question_info[2][1:-1] ]
                keyOrder.append(question_info[1])
    return question_dict, keyOrder

def read_choices(file):
    choice_dict = {}
    with open(file, 'r+') as choices_file:
        for line in choices_file:
            #choice_info[0]: img_id
            #choice_info[1]: q_id
            #choice_info[>1]: choices
            choice_info = []
            choice_info.append(line.split()[0])
            choice_info.append(line.split()[1])
            choice_info.append(line[line.find('(A)'):line.find('(B)')])
            choice_info.append(line[line.find('(B)'):line.find('(C)')])
            choice_info.append(line[line.find('(C)'):line.find('(D)')])
            choice_info.append(line[line.find('(D)'):line.find('(E)')])
            choice_info.append(line[line.find('(E)'):line.find('\n')])
            #ignore the first line of the file
            if choice_info[0].isdigit():
                #record the choices and save into the choice_dict
                choices = []
                #choices[0] for A, choices[1] for B.....
                #ex: choices[0] = "(A)....", choices[1] = "(B)...."
                for i in range(2,len(choice_info)):
                    #ignore (A), (B)...
                    choices.append(choice_info[i][3:])
                choice_dict[choice_info[1]] = choices
                print choice_dict[choice_info[1]]

    return choice_dict

def read_answer(file):
    answer_dict = {}
    with open(file, 'r+') as answer_file:
        for line in answer_file:
            #answer_info[0]: img_id
            #answer_info[1]: q_id
            #answer_info[>1]: answer
            #parsing exception conflict with shlex: \"
            if "\\\"" in line:
                answer_info = shlex.split(line.replace("\\\"", "@@exception"), posix=False)
            else:
                answer_info = shlex.split(line, posix=False)
            if answer_info[0].isdigit():
                if "@@exception" in answer_info[2]:
                    answer_dict[answer_info[1]] = answer_info[2].replace("@@exception", "\"")[1:-1]
                else:
                    answer_dict[answer_info[1]] = answer_info[2][1:-1]
    return answer_dict

def find_question_type(question):
    type_index = -1
    for i in range(0, len(question_types)):
        #return the type if the substring, question type, included in the question
        if question_types[i] in question.lower():
            #using if, else if, for preventing the situation of question_types[-1]
            if type_index == -1 :
                type_index = i
            elif len(question_types[i]) >= len(question_types[type_index]):
                #compare to before, choose the longer type
                type_index = i
            
    return type_index

def read_question_vec(file):
    print "Reading question_vec..."
    t_start = time.time()
    with open(file,'r+') as questions_file:
        question_vec = []
        for line in questions_file:
            #question_info[0]: sent_id
            #question_info[1:]: feature_vec
            question_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if not question_info[0].isdigit():
                for i in range(len(question_info[1:])):
                    question_info[i+1] = float(question_info[i+1])
                question_vec.append(question_info[1:])
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return question_vec

def read_answer_vec(file):
    print "Reading answer_vec..."
    t_start = time.time()
    with open(file,'r+') as answers_file:
        answer_vec = []
        for line in answers_file:
            #answer_info[0]: sent_id
            #answer_info[1:]: feature_vec
            answer_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if not answer_info[0].isdigit():
                for i in range(len(answer_info[1:])):
                    answer_info[i+1] = float(answer_info[i+1])
                answer_vec.append(answer_info[1:])
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return answer_vec

def read_choices_vec(file):
    print "Reading choices_vec..."
    t_start = time.time()
    #dimension of choices_vec: (# of questions) * 5 * (size of feature vec)
    with open(file,'r+') as choices_file:
        choices_vec = []
        choices_of_a_question = []
        count = 0
        for line in choices_file:
            #choices_info[0]: sent_id
            #choices_info[1:]: feature_vec
            choices_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if not choices_info[0].isdigit():
                for i in range(len(choices_info[1:])):
                    choices_info[i+1] = float(choices_info[i+1])
                choices_of_a_question.append(choices_info[1:])
                count += 1
            if count != 0 and not (count % 5):
                choices_vec.append(choices_of_a_question)
                choices_of_a_question = []
    t_end = time.time()
    print "time cost = ", t_end - t_start
    return choices_vec

def write_file(file, answer, keyOrder):
    print "Writing file..."
    t_start = time.time()
    with open(file, 'w') as f:
        f.write('q_id,ans')
        for i in range(len(keyOrder)):
            f.write('\n')
            f.write(keyOrder[i])
            f.write(',')
            if answer[i] == 0:
                f.write('A')
            elif answer[i] == 1:
                f.write('B')
            elif answer[i] == 2:
                f.write('C')
            elif answer[i] == 3:
                f.write('D')
            elif answer[i] == 4:
                f.write('E')
    t_end = time.time()
    print "time cost = ", t_end - t_start

