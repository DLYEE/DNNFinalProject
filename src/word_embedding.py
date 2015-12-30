import IO
import logging
import sys
import os
sys.path.insert(0, '/Users/apple/sentence2vec')
import word2vec
from word2vec import Word2Vec, Sent2Vec, LineSentence

#make files for word embedding
# if not open('question_word.train','r+'):
x_train_dict, trainKeyOrder = IO.read_questions('data/final_project_pack/question.train')
with open('data/vec/question_word.train','w') as word_file:
    for key in trainKeyOrder:
        word_file.write(x_train_dict[key][1] + '\n')

# if not open('answer_word.train','r+'):
y_train_dict = IO.read_answers('data/final_project_pack/answer.train')
with open('data/vec/answer_word.train','w') as word_file:
    for key in trainKeyOrder:
        word_file.write(y_train_dict[key] + '\n')

# if not open('choices_word.train','r+'):
c_train_dict = IO.read_choices('data/final_project_pack/choices.train')
with open('data/vec/choices_word.train','w') as word_file:
    for key in trainKeyOrder:
        for index in c_train_dict[key]:
            word_file.write(index + '\n')

# if not open('question_word.test','r+'):
x_test_dict, testKeyOrder = IO.read_questions('data/final_project_pack/question.test')
with open('data/vec/question_word.test','w') as word_file:
    for key in testKeyOrder:
        word_file.write(x_test_dict[key][1] + '\n')

# if not open('choices_word.test','r+'):
choices_dict = IO.read_choices('data/final_project_pack/choices.test')
with open('data/vec/choices_word.test','w') as word_file:
    for key in testKeyOrder:
        for index in choices_dict[key]:
            word_file.write(index + '\n')

#make a file of total words to train word-embedding model
with open('data/vec/word_embed.txt','w') as word_file:
    for line in open('data/vec/question_word.train','r+'):
        word_file.write(line)
    for line in open('data/vec/choices_word.train','r+'):
        word_file.write(line)
    for line in open('data/vec/question_word.test','r+'):
        word_file.write(line)
    for line in open('data/vec/choices_word.test','r+'):
        word_file.write(line)

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

input_file = 'data/vec/word_embed.txt'
#make model
word_model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=0, workers=8)
#save model
word_model.save(input_file + '.model')
#save word vectors
word_model.save_word2vec_format(input_file + '.vec')

sent_file = 'data/vec/question_word.train'
#use trained module to train sentence vectors
sent_model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
#save sentence vectors of question.train
sent_model.save_sent2vec_format(sent_file + '.vec')

sent_file = 'data/vec/answer_word.train'
sent_model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
sent_model.save_sent2vec_format(sent_file + '.vec')

sent_file = 'data/vec/choices_word.train'
sent_model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
sent_model.save_sent2vec_format(sent_file + '.vec')

sent_file = 'data/vec/question_word.test'
sent_model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
sent_model.save_sent2vec_format(sent_file + '.vec')

sent_file = 'data/vec/choices_word.test'
sent_model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
sent_model.save_sent2vec_format(sent_file + '.vec')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
