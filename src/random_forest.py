from kerasLSTM import make_batch
import numpy as np
import lstmIO
import shlex
from sklearn.ensemble import RandomForestClassifier

# ommiting lines after line 300 in kerasLSTM.py before running this code
# write file is not done yet

def count_num():
    word_file = open('data/vec/question_word.train', 'r+')
    num_lines = sum(1 for line in word_file)
    word_file.close()
    return num_lines
    

def y_label_generate(answer_file):
    
    # generate label y from answer_file
    
    y = []
    with open(answer_file,'r+') as file:
        for line in file:
            ans = shlex.split(line, posix=False)
            if ans[2][0] == 'A':
                y.append(0)
            elif ans[2][0] == 'B':
                y.append(1)
            elif ans[2][0] == 'C':
                y.append(2)
            elif ans[2][0] == 'D':
                y.append(3)
            elif ans[2][0] == 'E':
                y.append(4)
    return y


def x_feature_generate(question, choice, dictionary):
    
    # generate training feature
    # question type : string (lines of question)
    # choice type : string (lines of choices)
    
    x_feature = []
    question_vec = make_batch(question, dictionary)
    for i in range(len(question_vec)):
        x_feature_i = np.zeros(300)
        for j in range(len(question_vec[i])):
            x_feature_i = x_feature_i + question_vec[i][j]

        x_feature_i = x_feature_i / len(question_vec[i])
        x_feature.append(x_feature_i)
        print 'x_feature [', i, '] of question : '#, x_feature[i]
    print 'x_feature [', 0, '] of question : ', x_feature[0]
    
    choice_vec = make_batch(choice, dictionary)
    for i in range(len(choice_vec)/5):
        x_feature_i0 = np.zeros(300)
        for j in range(len(choice_vec[5*i])):
            x_feature_i0 = x_feature_i0 + choice_vec[5*i][j]
            x_feature_i0 = x_feature_i0 / len(choice_vec[5*i])
        x_feature_i1 = np.zeros(300)
        for j in range(len(choice_vec[5*i+1])):
            x_feature_i1 = x_feature_i1 + choice_vec[5*i+1][j]
            x_feature_i1 = x_feature_i1 / len(choice_vec[5*i+1])
        x_feature_i2 = np.zeros(300)
        for j in range(len(choice_vec[5*i+2])):
            x_feature_i2 = x_feature_i2 + choice_vec[5*i+2][j]
            x_feature_i2 = x_feature_i2 / len(choice_vec[5*i+2])
        x_feature_i3 = np.zeros(300)
        for j in range(len(choice_vec[5*i+3])):
            x_feature_i3 = x_feature_i3 + choice_vec[5*i+3][j]
            x_feature_i3 = x_feature_i3 / len(choice_vec[5*i+3])
        x_feature_i4 = np.zeros(300)
        for j in range(len(choice_vec[5*i+4])):
            x_feature_i4 = x_feature_i4 + choice_vec[5*i+4][j]
            x_feature_i4 = x_feature_i4 / len(choice_vec[5*i+4])
            
        # concatenate x_feature[i] & x_feature_i* together 
        #   => shape = (6, 300)
        # then ravel to 1D vector 
        #   => shape = (1, 1800)
        #   => 0~299    : x_feature[i]
        #   => 300~599  : x_feature_i0
        #   => ... ...
        x_feature[i] = np.ravel(np.concatenate((x_feature[i], x_feature_i0, x_feature_i1, x_feature_i2, x_feature_i3, x_feature_i4)))
        print 'x_feature[', i, '] : '#, x_feature[i]
    print 'x_feature [', 0, '] : ', x_feature[0]
    return x_feature


def sub_clf(question, choices, Y, dictionary):

    # make a random forest classifier for each batch
    # merge all forest together at the end
    
    batch_size = 10000
    clfs = None
    ques_file = open(question, 'r+')
    choi_file = open(choices, 'r+')
    count = 0
    ques_lines = ""
    choi_lines = ""
    current_lines = 0
    num_lines = count_num()
    for ques_line in ques_file:
        if current_lines == num_lines - 1:
            current_lines += 1
            ques_lines += ques_line
            for ch in range(5):
                choi_lines += choi_file.readline()
            x = x_feature_generate(ques_lines, choi_lines, dictionary)
            y = np.asarray(Y[num_lines - num_lines % batch_size:])
            clf = ""
            clf = RandomForestClassifier(n_estimators = 10)
            clf = clf.fit(x,y)
            if clfs == None:
                clfs = clf
            else:
                clfs.estimators_.extend(clf.estimators_)
                clfs.n_estimators += clf.n_estimators
            count += 1
            ques_lines = ""
            choi_lines = ""
        elif current_lines % batch_size == batch_size - 1: 
            current_lines += 1
            ques_lines += ques_line
            for ch in range(5):
                choi_lines += choi_file.readline()
            x = x_feature_generate(ques_lines, choi_lines, dictionary)
            y = np.asarray(Y[current_lines - batch_size : current_lines])
            clf = ""
            clf = RandomForestClassifier(n_estimators = 10)
            clf = clf.fit(x,y)
            if clfs == None:
                clfs = clf
            else:
                clfs.estimators_.extend(clf.estimators_)
                clfs.n_estimators += clf.n_estimators
            count += 1
            ques_lines = ""
            choi_lines = ""
        else:
            current_lines += 1
            ques_lines += ques_line
            for ch in range(5):
                choi_lines += choi_file.readline()
    ques_file.close()
    choi_file.close()
    return clfs


Y = y_label_generate('data/final_project_pack/answer.train_sol')
dictionary = lstmIO.readGlove("data/small_glove_200000.txt", 200000)
clf = sub_clf('data/vec/question_word.train', 'data/vec/choices_word.train', Y, dictionary)

qt_lines = open('data/vec/question_word.test', 'r+').read()
ct_lines = open('data/vec/choices_word.test', 'r+').read()
Xt = x_feature_generate(qt_lines, ct_lines, dictionary)
Yt = clf.predict(Xt)
# write file not done
print Yt
