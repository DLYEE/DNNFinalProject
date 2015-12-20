from constants import question_types  
import shlex

def read_questions(file):
    question_list = {}
    with open(file,'r+') as questions_file:
        for line in questions_file:
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            question_info  = shlex.split(line)
            #ignore the first line of the file
            if question_info[0].isdigit():
                question_list[question_info[1]] = [find_question_type(question_info[2]), question_info[2] ]
                print question_list[question_info[1]]
                #save the question type index and the question in the question_list, which is a dictionary
    return question_list

def read_choices(file):
    choice_list = {}
    with open(file, 'r+') as choices_file:
        for line in choices_file:
            #choice_info[0]: img_id
            #choice_info[1]: q_id
            #choice_info[>1]: choices
            choice_info = shlex.split(line, posix=False)
            #ignore the first line of the file
            if choice_info[0].isdigit():
                #record the choices and save into the choice_list
                choices = []
                #choices[0] for A, choices[1] for B.....
                #ex: choices[0] = "(A)....", choices[1] = "(B)...."
                for i in range(2,len(choice_info)):
                    choices.append(choice_info[i])
                choice_list[choice_info[1]] = choices

    return choice_list

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

read_choices('./ex.txt')
