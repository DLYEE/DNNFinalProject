import numpy as np
import time
import math
import IO
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def construct_model():
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 100-dimensional vectors.
    model.add(Dense(1024, input_dim=100, init='uniform'))
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

def train():

    # cosine similarity as symbolic cost function for training
    # y_true: true labels (theano tensor)
    # y_pred: predictions (theano tensor)
    def cost_function(y_true, y_pred):
        len1 = K.sqrt(K.sum(K.square(y_true)))
        len2 = K.sqrt(K.sum(K.square(y_pred)))
        return - K.sum(y_true * y_pred) / (len1 * len2)

    # trainKeyOrder = IO.read_qid('data/final_project_pack/question.train')
    x_train = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))
    y_train = np.asarray(IO.read_answer_vec('data/vec/answer_word.train.vec'))

    LR = 1E-1
    print "learning rate = ", LR
    sgd = SGD(lr=LR, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss=cost_function, optimizer=sgd)

    print "Training begins..."
    t_start = time.time()

    model.fit(x_train, y_train, nb_epoch=50, batch_size=100, show_accuracy = False)

    print "Training ends."
    t_end = time.time()
    print "time cost = ", t_end - t_start

def test():

    def similarity(sent_vec1, sent_vec2):
        # for i in range(len(sent_vec1)):
            # sent_vec1[i] = float(sent_vec1[i])
            # sent_vec2[i] = float(sent_vec2[i])
        len1 = math.sqrt(sum(x * x for x in sent_vec1))
        len2 = math.sqrt(sum(x * x for x in sent_vec2))
        return np.dot(sent_vec1, sent_vec2) / (len1 * len2)

    testKeyOrder = np.asarray(IO.read_qid('data/final_project_pack/question.train'))
    x_test = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))

    print "Testing begins..."
    t_start = time.time()

    predict = model.predict(x_test, batch_size=100)

    print "Testing ends."
    t_end = time.time()
    print "time cost = ", t_end - t_start

    choices = np.asarray(IO.read_choices_vec('data/vec/choices_word.train.vec'))

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
train()
test()

###
###end main
###


