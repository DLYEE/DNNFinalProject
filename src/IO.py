from constants import question_types  
import shlex

def read_questions(file):
    question_list = {}
    with open(file,'r+') as questions_file:
        for line in questions_file:
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            question_info  = shlex.split(line, posix=False)
            #ignore the first line of the file
            if question_info[0].isdigit():
                #save the question type index and the question in the question_list, which is a dictionary
                #question_info[2][1:-1] => avoid double quotes 
                question_list[question_info[1]] = [find_question_type(question_info[2]), question_info[2][1:-1] ]
    return question_list

def read_choices(file):
    choice_list = {}
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
                #record the choices and save into the choice_list
                choices = []
                #choices[0] for A, choices[1] for B.....
                #ex: choices[0] = "(A)....", choices[1] = "(B)...."
                for i in range(2,len(choice_info)):
                    # ignore (A), (B)..., and whitespace
                    choices.append(choice_info[i][3:-2])
                choice_list[choice_info[1]] = choices
    return choice_list

def read_answers(file):
    answer_list = {}
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
                    answer_list[answer_info[1]] = answer_info[2].replace("@@exception", "\"")[1:-1]
                else:
                    answer_list[answer_info[1]] = answer_info[2][1:-1]
    return answer_list

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
