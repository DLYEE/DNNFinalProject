import IO
import logging
import sys
import os
sys.path.insert(0, '/Users/apple/sentence2vec')
import word2vec
from word2vec import Word2Vec, Sent2Vec, LineSentence

def make_question_word_embed_file(file, dict, keyOrder):
    #read from question dictionary
    #every question per line in the order of keyOrder
    with open(file, 'w') as word_file:
        for key in keyOrder:
            word_file.write(dict[key][1] + '\n')

def make_answer_word_embed_file(file, dict, keyOrder):
    #read from answer dictionary
    #every answer per line in the order of keyOrder
    with open(file, 'w') as word_file:
        for key in keyOrder:
            word_file.write(dict[key] + '\n')

def make_choices_word_embed_file(file, dict, keyOrder):
    #read from choices dictionary
    #every choice per line in the order of (A) (B) (C) (D) (E) in keyOrder
    with open(file, 'w') as word_file:
        for key in keyOrder:
            for index in dict[key]:
                word_file.write(index + '\n')

def make_total_word_embed_file(fwordembed, fqtrain, fctrain, fqtest, fctest):
    #to train word embedding model, we need a file of all words
    #so we merge all files together
    with open(fwordembed, 'w') as word_file:
        for line in open(fqtrain, 'r+'):
            word_file.write(line)
        for line in open(fctrain, 'r+'):
            word_file.write(line)
        for line in open(fqtest, 'r+'):
            word_file.write(line)
        for line in open(fctest, 'r+'):
            word_file.write(line)

def train_word_model(fword, vecsize, mincount):
    #make model
    word_model = Word2Vec(LineSentence(fword), size=vecsize, window=5, sg=0, min_count = mincount, workers=8)
    #save model
    word_model.save(fword + '.model')
    #save word vectors
    word_model.save_word2vec_format(fword + '.vec')

def train_sent_vec(fword, fsent):
    #use trained module to train sentence vectors
    sent_model = Sent2Vec(LineSentence(fsent), model_file=fword + '.model')
    #save sentence vectors
    sent_model.save_sent2vec_format(fsent + '.vec')


question_train_file = 'data/final_project_pack/question.train'
answer_train_file = 'data/final_project_pack/answer.train'
choices_train_file = 'data/final_project_pack/choices.train'
question_test_file = 'data/final_project_pack/question.test'
choices_test_file = 'data/final_project_pack/choices.test'

#read from files to dict
x_train_dict, trainKeyOrder = IO.read_question(question_train_file)
y_train_dict = IO.read_answer(answer_train_file)
c_train_dict = IO.read_choices(choices_train_file)
x_test_dict, testKeyOrder = IO.read_question(question_test_file)
c_test_dict = IO.read_choices(choices_test_file)

question_train_word_file = 'data/vec/question_word.train'
answer_train_word_file = 'data/vec/answer_word.train'
choices_train_word_file = 'data/vec/choices_word.train'
question_test_word_file = 'data/vec/question_word.test'
choices_test_word_file = 'data/vec/choices_word.test'

#make files for word embedding
make_question_word_embed_file(question_train_word_file, x_train_dict, trainKeyOrder)
make_answer_word_embed_file(answer_train_word_file, y_train_dict, trainKeyOrder)
make_choices_word_embed_file(choices_train_word_file, c_train_dict, trainKeyOrder)
make_question_word_embed_file(question_test_word_file, x_test_dict, testKeyOrder)
make_choices_word_embed_file(choices_test_word_file, c_test_dict, testKeyOrder)

word_embed_file = 'data/vec/word_embed.txt'
make_total_word_embed_file(word_embed_file, question_train_word_file, choices_train_word_file, question_test_word_file, choices_test_word_file )


#some system output
logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

#train a word model from word_embed_file
train_word_model(word_embed_file, vecsize=100, mincount=0)
#train sentence vectors from previously train word model
train_sent_vec(word_embed_file, question_train_word_file)
train_sent_vec(word_embed_file, answer_train_word_file)
train_sent_vec(word_embed_file, choices_train_word_file)
train_sent_vec(word_embed_file, question_test_word_file)
train_sent_vec(word_embed_file, choices_test_word_file)

#some system output
program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
