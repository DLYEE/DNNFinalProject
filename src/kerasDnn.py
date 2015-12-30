import numpy as np
import IO
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

def construct_model():
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 400-dimensional vectors.
    model.add(Dense(64, input_dim=400, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(400, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(400, init='uniform'))
    # model.add(Activation('softmax'))

def train():
    print "Training begins..."

    # trainKeyOrder = IO.read_qid('data/final_project_pack/question.train')
    x_train = IO.read_question_vec('data/vec/question_word.train.vec')
    y_train = IO.read_answer_vec('data/vec/answer_word.train.vec')

    sgd = SGD(lr=1E-3, decay=0, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd)

    model.fit(x_train, y_train, nb_epoch=3, batch_size=100, show_accuracy = True)

    print "Training ends."

def test():
    print "Testing begins..."

    def similarity(sent_vec1, sent_vec2):
        return np.dot(sen_vec1, sent_vec2)
        testKeyOrder = IO.read_qid('data/final_project_pack/question.train')

    x_test = IO.read_question_vec('data/vec/question_word.test.vec')

    predict = model.predict(x_test, batch_size=100)

    choices = IO.read_choices_vec('data/vec/choices_word.train.vec')

    # pick the answer among 5 choices
    answer = []
    for index in range(len(testKeyOrder)):
        cos_similarity = 0
        answer_of_a_question = -1
        for i in range(5):
            if similarity(choices[index][i], predict[index]) > cos_similarity:
                cos_similarity = similarity(choices[index][i], predict[index])
                answer_of_a_question = i
        answer.append(answer_of_a_question)

    # write file
    IO.write_file('predict.csv', answer, testKeyOrder)

    print "Testing ends."


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


