import shlex
from sklearn.ensemble import RandomForestClassifier

def x_feature_generate(question_file, choice_file):

def y_feature_generate(answer_file):
    y = []
    with file as open(answer_file,'r+'):
        for line in file:
            ans = shlex.split(line, posiz=False)
            if ans[2] == 'A':
                y.append(0)
            elif ans[2] == 'B':
                y.append(1)
            elif ans[2] == 'C':
                y.append(2)
            elif ans[2] == 'D':
                y.append(3)
            elif ans[2] == 'E':
                y.append(4)
    return y

X = []
Y = []
clf = RandomForestClassifier(n_estimators = 10)
clf = clf.fit(X,Y)
Yt = clf.predict(Xt)
