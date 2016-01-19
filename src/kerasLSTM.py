import numpy as np
import time
import math
import re
import IO
import lstmIO
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Reshape
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.optimizers import RMSprop
from keras.layers.embeddings import Embedding
from keras import backend as K

def construct_model():
    # Dense(300) is a fully-connected layer with 300 hidden units.
    # in the first layer, you must specify the expected input data shape:
    model.add(LSTM(300, input_dim = 300, return_sequences = False, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    # model.add(Dense(100,1))
    # model.add(Activation('sigmoid'))
    # here, 100-dimensional vectors.
    model.add(Dense(1024, input_dim = 300, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    for i in range(2):
        model.add(Dense(1024, init='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    model.add(Dense(300, init='uniform'))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(300, init='uniform'))
    # model.add(Activation('softmax'))

def word2vec(word, dictionary, vec):
    # print word
    value = dictionary.get(word)
    if (value != None):
        vec.append(value)
    elif (len(word) > 0):
        lstmIO.specialword(vec, dictionary, word)
        # print "len of vec in word2vec =", len(vec)

def make_batch(lines, dictionary):
    line = lines.splitlines()
    wordvec_list = []
    max_len = 0
    for index in range(len(line)):
        linevec = []
        sentence = re.split(" |\n",line[index])
        for i in range(len(sentence)):
            word2vec(sentence[i], dictionary, linevec)
        # if index == 0:
            # print len(linevec)
        wordvec_list.append(linevec)
        length = len(linevec)
        if length > max_len:
            max_len = length
    # print "Max_len =", max_len
    wordvec = np.empty([len(line), max_len, 300])
    for index in range(len(line)):
        linevec = np.asarray(wordvec_list[index])
        linevec = np.vstack((np.zeros((max_len - np.size(linevec,0),300)),linevec))
        wordvec[index] = linevec
    return wordvec

def train(dictionary):

    # cosine similarity as symbolic cost function for training
    # y_true: true labels (theano tensor)
    # y_pred: predictions (theano tensor)
    def cost_function(y_true, y_pred):
        len1 = K.sqrt(K.sum(K.square(y_true)))
        len2 = K.sqrt(K.sum(K.square(y_pred)))
        return - K.sum(y_true * y_pred) / (len1 * len2)
    
    def count_num_lines():
        wordfile = open("data/stop_word/question_word.train", 'r+')
        num_lines = sum(1 for line in wordfile)
        print "There are ", num_lines, "lines in train file."
        wordfile.close()
        return num_lines

    def train_on_batch(num_lines, y_train):
        wordfile = open("data/stop_word/question_word.train", 'r+')
        # if num_lines != len(y_train):
            # print "Input and label have different size. I =", num_lines,", L =",len(y_train)
        current_lines = 0
        batch_size = 128
        # mini_batch_size = 32
        lines = ""
        linevec = []
        loss = 0
        count = 0
        for line in wordfile:
            if current_lines % 5000 == 0 and current_lines != 0:
                print "Training", current_lines, "lines..."
            if current_lines == num_lines -1:
                lines += line
                current_lines += 1
                x_batch = make_batch(lines, dictionary)
                y_batch = np.asarray(y_train[num_lines - num_lines % batch_size:])
                # model.fit(x_batch, y_batch, mini_batch_size, 1)
                loss += model.train_on_batch(x_batch, y_batch)[0]
                count += 1
                lines = ""
            if current_lines % batch_size == batch_size -1:
                lines += line
                current_lines += 1
                x_batch = make_batch(lines, dictionary)
                y_batch = np.asarray(y_train[current_lines-batch_size:current_lines])
                # model.fit(x_batch, y_batch, mini_batch_size, 1)
                loss += model.train_on_batch(x_batch, y_batch)[0]
                count += 1
                lines = "" 
            else:
                lines += line
                current_lines += 1
        print "Loss = ", loss / count
        wordfile.close()
    
    # trainKeyOrder = IO.read_qid('data/final_project_pack/question.train')
    # x_train = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))
    LR = 1E-3
    print "learning rate = ", LR
    # sgd = SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
    # ada = Adagrad(lr=0.01, epsilon=1e-06)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss=cost_function, optimizer=rms)

    print "Training begins..."
    t_start = time.time()
    # y_train = np.asarray(IO.read_answer_vec('data/vec/answer_word.train.vec'))
    y_train_txt = lstmIO.read_answer_txt('data/stop_word/answer_word.train')
    y_train = []
    for i in range(len(y_train_txt)):
        linevec = []
        for j in range(len(y_train_txt[i])):
            word2vec(y_train_txt[i][j], dictionary, linevec)
            # value = dictionary.get(y_train_txt[i][j])
            # if (value != None):
                # linevec.append(value)
            # elif (len(y_train_txt[i][j]) > 0):
                # lstmIO.specialword(linevec, dictionary, y_train_txt[i][j])
                # print "len of vec in word2vec =", len(linevec)
            # if i==0:
                # print j, len(linevec)
        y_train.append(np.sum(linevec, axis=0) / len(y_train_txt[i]))
    y_train = np.asarray(y_train)
    
    # train in batches
    num_lines = count_num_lines()
    num_epoch = 1
    for i in range(num_epoch):
        print "It's the ", i+1, " epoch."
        t_s = time.time()
        train_on_batch(num_lines, y_train)
        t_e = time.time()
        print "Time cost = ", t_e - t_s

    print "Training ends."
    t_end = time.time()
    print "Total time cost = ", t_end - t_start

def test(dictionary):
    
    def similarity(sent_vec1, sent_vec2):
        # for i in range(len(sent_vec1)):
            # sent_vec1[i] = float(sent_vec1[i])
            # sent_vec2[i] = float(sent_vec2[i])
        len1 = math.sqrt(sum(x * x for x in sent_vec1))
        len2 = math.sqrt(sum(x * x for x in sent_vec2))
        return np.dot(sent_vec1, sent_vec2) / (len1 * len2)
    
    def count_num_lines():
        wordfile = open("data/stop_word/question_word.test", 'r+')
        num_lines = sum(1 for line in wordfile)
        print "There are ", num_lines, "lines in test file."
        wordfile.close()
        return num_lines

    def test_on_batch(num_lines):
        wordfile = open("data/stop_word/question_word.test", 'r+')
        current_lines = 0
        batch_size = 128
        lines = ""
        linevec = []
        predict = np.empty([0, 100])
        for line in wordfile:
            if current_lines % 10000 == 0 and current_lines != 0:
                print "Testing", current_lines, "lines..."
            if current_lines == num_lines -1:
                lines += line
                x_batch = make_batch(lines, dictionary)
                if predict.size == 0:
                    # predict = model.predict(x_batch, 50)
                    predict = model.predict_on_batch(x_batch)[0]
                    # print len(predict)
                else:
                    # predict = np.vstack((predict, model.predict(x_batch, 50)))
                    # print predict.shape
                    # print model.predict_on_batch(x_batch)[0].shape
                    predict = np.vstack((predict, model.predict_on_batch(x_batch)[0]))
                    # print len(predict)
                lines = ""
                current_lines += 1
            if current_lines % batch_size == batch_size -1:
                lines += line
                x_batch = make_batch(lines, dictionary)
                if predict.size == 0:
                    # predict = model.predict(x_batch, 50)
                    predict = model.predict_on_batch(x_batch)[0]
                    # print len(predict)
                else:
                    # predict = np.vstack((predict, model.predict(x_batch, 50)))
                    # print predict.shape
                    # print model.predict_on_batch(x_batch)[0].shape
                    predict = np.vstack((predict, model.predict_on_batch(x_batch)[0]))
                    # print len(predict)
                lines = "" 
                current_lines += 1
            else:
                lines += line
                current_lines += 1
        wordfile.close()
        # print "size of predict = ", len(predict)
        # print "predict = ", predict
        return predict

    def generate_answer(predict, choices, answer):
        cos_similarity = -1000000000
        answer_of_a_question = -1
        for i in range(5):
            sim = similarity(choices[i], predict)
            if i == 0:
                cos_similarity = sim
                answer_of_a_question = i
            if sim > cos_similarity:
                cos_similarity = sim
                answer_of_a_question = i
        answer.append(answer_of_a_question)

    testKeyOrder = np.asarray(IO.read_qid('data/final_project_pack/question.test'))
    print "Testing begins..."
    t_start = time.time()
    # test in batches
    num_lines = count_num_lines()
    predict = test_on_batch(num_lines)
    print "Testing ends."
    t_end = time.time()
    print "time cost = ", t_end - t_start

    # choices = np.asarray(IO.read_choices_vec('data/vec/choices_word.test.vec'))
    choices_txt = lstmIO.read_choices_txt('data/stop_word/choices_word.test')
    # print "len of choices_txt =", len(choices_txt)
    # pick the answer among 5 choices
    print "Generating answer..."
    t_start = time.time()
    answer = []
    count = 0
    choices_of_a_question = []
    # time_b = time.time()
    time_count = 0
    for i in range(len(choices_txt)):
        linevec = []
        # print len(choices_txt[i])
        time_begin = time.time()
        for j in range(len(choices_txt[i]) - 1):
            word2vec(choices_txt[i][j], dictionary, linevec)
            # value = dictionary.get(choices_txt[i][j])
            # if (value != None):
                # linevec.append(value)
            # elif (len(choices_txt[i][j]) > 0):
                # lstmIO.specialword(linevec, dictionary, choices_txt[i][j])
        time_end = time.time()
        time_count += (time_end - time_begin)
        choices_of_a_question.append(np.sum(linevec, axis = 0) / (len(choices_txt[i]) - 1))
        # print "choices_of_a_question after appending: ",choices_of_a_question
        # print choices_of_a_question
        count += 1
        if count % 5 == 0:
            # choices_of_a_question = np.asarray(choices_of_a_question)
            generate_answer(predict[count / 5 - 1], choices_of_a_question, answer)
            choices_of_a_question = []
        if count % 25000 == 0:
            print "Generating",len(answer),"lines..."
    t_end = time.time()
    print "time cost = ", t_end - t_start

    # write file
    IO.write_file('predict.csv', answer, testKeyOrder)

###
### main code
###

model = Sequential()
construct_model()
dictionary = lstmIO.readGlove("data/WordEmbedding/glove.840B.300d.txt", 200000)
train(dictionary)
test(dictionary)

###
###end main
###

