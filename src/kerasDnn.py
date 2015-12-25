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
    model.add(Dense(64, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(100, init='uniform'))
    # model.add(Activation('softmax'))

###
def sent2vec(sentence):
    淞楓write it!!!
###

###
def similarity(sent_vec1, sent_vec2):
    淞楓write it!!!
###


###
### main code
###
model = Sequential()
construct_model()

###
### train
x_train_dict, trainKeyOrder = IO.read_questions('data/final_project_pack/question.train')
y_train_dict = IO.read_answers('data/final_project_pack/answer.train')

# convert type 'dict' into type'list'
x_train = []
y_train = []
for key in trainKeyOrder:
    x_train.append(x_train_dict[key][1])
    y_train.append(y_train_dict[key])

# convert 'str' into 'feature vector'
x_train_vec = sent2vec(x_train)
y_train_vec = sent2vec(y_train)

sgd = SGD(lr=1E-3, decay=0, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(x_train_vec, y_train_vec, nb_epoch=3, batch_size=16, show_accuracy = True)
### end train
###

###
###test
x_test_dict, testKeyOrder = IO.read_questions('data/final_project_pack/question.train')

# convert type 'dict' into type'list'
x_test = []
for key in testKeyOrder:
    x_test.append(x_test_dict[key][1])

# convert 'str' into 'feature vector'
x_test_vec = sent2vec(x_test)

predict = model.predict(x_test, batch_size=16)

# convert into type 'list' and 'feature vector'
choices_dict = IO.read_choices('data/final_project_pack/choices.train')
choices = []
for key in testKeyOrder:
    choices_of_a_question = []
    for i in range(5):
        choices_of_a_question.append(sent2vec(choices_dict[key][i]))
    choices.append(choices_of_a_question)

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
###end test
###

###
###end main
###


