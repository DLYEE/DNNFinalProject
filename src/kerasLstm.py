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
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    model.add(LSTM(300, input_dim = 300, return_sequences = False, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Dropout(0.5))
    # model.add(Dense(100,1))
    # model.add(Activation('sigmoid'))
    # here, 100-dimensional vectors.
    model.add(Dense(1024, input_dim = 300, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    for i in range(1):
        model.add(Dense(1024, init='uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
    # model.add(Dense(100, init='uniform'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('softmax'))

def make_batch(lines, dictionary):
    line = lines.splitlines()
    wordvec_list = []
    max_len = 0
    for index in range(len(line)):
        linevec = []
        sentence = re.split(" |\n",line[index])
        for i in range(len(sentence)):
            value = dictionary.get(sentence[i])
            if (None != value):
                linevec.append(value)
            elif (len(sentence[i]) > 0):
                lstmIO.specialword(linevec, dictionary, sentence[i])
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

    # trainKeyOrder = IO.read_qid('data/final_project_pack/question.train')
    # x_train = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))
    LR = 1E-1
    print "learning rate = ", LR
    sgd = SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
    ada = Adagrad(lr=0.01, epsilon=1e-06)
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss=cost_function, optimizer=sgd)

    print "Training begins..."
    t_start = time.time()
    y_train = np.asarray(IO.read_answer_vec('data/vec_100/small_answer.txt'))
    
    # train in batches
    wordfile = open("data/vec_100/question_word.train", 'r+')
    num_lines = sum(1 for line in wordfile)
    print "Have", num_lines, "lines in train file."
    wordfile = open("data/vec_100/question_word.train", 'r+')
    if num_lines != len(y_train):
        print "Input and label have different size. I =", num_lines,", L =",len(y_train)
    current_lines = 0
    batch_size = 1000
    lines = ""
    linevec = []
    for line in wordfile:
        if current_lines % 5000 == 0 and current_lines != 0:
            print "Reading train", current_lines, "lines..."
        if current_lines == num_lines -1:
            lines += line
            x_batch = make_batch(lines, dictionary)
            y_batch = np.asarray(y_train[num_lines - num_lines % batch_size:])
            model.fit(x_batch, y_batch, 32, 1)
            lines = ""
            current_lines += 1
        if current_lines % batch_size == batch_size -1:
            lines += line
            x_batch = make_batch(lines, dictionary)
            y_batch = np.asarray(y_train[current_lines-batch_size+1:current_lines])
            model.fit(x_batch, y_batch, 32, 1)
            lines = "" 
            current_lines += 1
        else:
            lines += line
            current_lines += 1
    
    print "Training ends."
    t_end = time.time()
    print "time cost = ", t_end - t_start

def test(dictionary):
    
    def similarity(sent_vec1, sent_vec2):
        # for i in range(len(sent_vec1)):
            # sent_vec1[i] = float(sent_vec1[i])
            # sent_vec2[i] = float(sent_vec2[i])
        len1 = math.sqrt(sum(x * x for x in sent_vec1))
        len2 = math.sqrt(sum(x * x for x in sent_vec2))
        return np.dot(sent_vec1, sent_vec2) / (len1 * len2)

    testKeyOrder = np.asarray(IO.read_qid('data/final_project_pack/question.test'))
    print "Testing begins..."
    t_start = time.time()
    # test in batches
    wordfile = open("data/vec_100/question_word.test", 'r+')
    num_lines = sum(1 for line in wordfile)
    print "Have", num_lines, "lines in test file."
    wordfile = open("data/vec_100/question_word.test", 'r+')
    current_lines = 0
    batch_size = 1000
    lines = ""
    linevec = []
    predict = np.array([[]])
    for line in wordfile:
        if current_lines % 10000 == 0 and current_lines != 0:
            print "Reading test", current_lines, "lines..."
        if current_lines == num_lines -1:
            lines += line
            x_batch = make_batch(lines, dictionary)
            if predict.size == 0:
                predict = model.predict(x_batch, 50)
            else:
                predict = np.vstack((predict, model.predict(x_batch, 50)))
            lines = ""
            current_lines += 1
        if current_lines % batch_size == batch_size -1:
            lines += line
            x_batch = make_batch(lines, dictionary)
            if predict.size == 0:
                predict = model.predict(x_batch, 50)
            else:
                predict = np.vstack((predict, model.predict(x_batch, 50)))
            lines = "" 
            current_lines += 1
        else:
            lines += line
            current_lines += 1

    print "Testing ends."
    t_end = time.time()
    print "time cost = ", t_end - t_start

    choices = np.asarray(IO.read_choices_vec('data/vec_100/choices_word.test.vec'))

    # pick the answer among 5 choices
    print "Generating answer..."
    t_start = time.time()
    answer = []
    for index in range(len(testKeyOrder)):
        cos_similarity = -100000000
        answer_of_a_question = -1
        for i in range(5):
            sim = similarity(choices[index][i], predict[index])
            if sim > cos_similarity:
                cos_similarity = sim
                answer_of_a_question = i
        answer.append(answer_of_a_question)
    t_end = time.time()
    print "time cost = ", t_end - t_start

    # write file
    IO.write_file('predict.csv', answer, testKeyOrder)

###
### main code
###

model = Sequential()
construct_model()
dictionary = lstmIO.readGlove("data/WordEmbedding/glove.840B.300d.txt", 300000)
train(dictionary)
test(dictionary)

###
###end main
###

