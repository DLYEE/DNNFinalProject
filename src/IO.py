from constants import question_types  
import shlex

def read_questions(file):
    question_list = {}
    with open(file,'r+') as questions:
    #open the question file
        for question in questions:
            question_info  = shlex.split(question)
            #question_info[0]: img_id
            #question_info[1]: q_id
            #question_info[2]: question
            if question_info[0].isdigit():
            #ignore the first line of the file
                question_list[question_info[1]] = [find_question_type(question_info[2]), question_info[2] ]
                print question_list[question_info[1]]
                #save the question type index and the question in the question_list, which is a dictionary
    return question_list

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

