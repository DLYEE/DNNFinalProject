from kerasLSTM import make_batch
import numpy as np
import lstmIO

def x_feature_generate(question, choice, dictionary):
    x_feature = []
    with open(question,'r+') as question_lines:
        question_vec = make_batch(question_lines, dictionary)
        for i in range(len(question_vec)):
            x_feature_i = np.zeros(300)
            for j in range(len(question_vec[i])):
                x_feature_i = x_feature_i + question_vec[i][j]

            x_feature_i = x_feature_i / len(question_vec[i])
            x_feature.append(x_feature_i)
            print 'x_feature [', i, '] of question : ', x_feature[i]
    
    with open(choice, 'r+') as choice_lines:
        choice_vec = make_batch(choice_lines, dictionary)
        for i in range(len(choice_vec)):
            x_feature_i = np.zeros(300)
            for j in range(len(choice_vec[i])):
                x_feature_i = x_feature_i + choice_vec[i][j]
                
            x_feature_i = x_feature_i / len(choice_vec[i])
            print 'x_feature [', i, '] of choice : ', x_feature_i

            x_feature[i] = x_feature[i] + x_feature_i
            print 'x_feature[', i, '] : ', x_feature[i]


dictionary = lstmIO.readGlove("data/glove.840B.300d.txt", 200000)
x_feature_generate('data/final_project_pack/question_word.train','',dictionary)
