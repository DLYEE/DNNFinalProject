import numpy as np
import IO
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def construct_model():
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 100-dimensional vectors.
    model.add(Dense(64, input_dim=100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(400, init='uniform'))
    # model.add(Activation('softmax'))

def train():
    # trainKeyOrder = IO.read_qid('data/final_project_pack/question.train')
    x_train = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))
    y_train = np.asarray(IO.read_answer_vec('data/vec/answer_word.train.vec'))

    sgd = SGD(lr=1E-3, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    print "Training begins..."
    model.fit(x_train, y_train, nb_epoch=20, batch_size=100, show_accuracy = True)
    print "Training ends."

def test():

    def similarity(sent_vec1, sent_vec2):
        # for i in range(len(sent_vec1)):
            # sent_vec1[i] = float(sent_vec1[i])
            # sent_vec2[i] = float(sent_vec2[i])
        return np.dot(sent_vec1, sent_vec2)

    testKeyOrder = np.asarray(IO.read_qid('data/final_project_pack/question.train'))
    x_test = np.asarray(IO.read_question_vec('data/vec/question_word.train.vec'))

    print "Testing begins..."
    predict = model.predict(x_test, batch_size=100)
    print "Testing ends."

    choices = np.asarray(IO.read_choices_vec('data/vec/choices_word.train.vec'))

    # pick the answer among 5 choices
    print "Generating answer..."
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


