
answer_list = []
with open('data/final_project_pack/answer.train_sol', 'r+') as answer_file:
    for line in answer_file:
        if line[0].isdigit():
            answer_list.append(line[-2])
predict_list = []
with open('predict.csv', 'r+') as predict_file:
    for line in predict_file:
        if line[0].isdigit():
            predict_list.append(line[-2])

count = 0
for i in range(len(answer_list)):
    if answer_list[i] == predict_list[i]:
        count = count + 1

print count
print len(answer_list)
print float(count)/float(len(answer_list))
