import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

df = pd.read_csv("document.csv")

data = df.values.tolist()

data, test = train_test_split(data, test_size = 0.2)

X = [data[i][0] for i in range(len(data))]
y = [data[i][1] for i in range(len(data))]

X_pos = [X[i] for i in range(len(X)) if y[i] == 'pos']
X_neg = [X[i] for i in range(len(X)) if y[i] == 'neg']

newX_pos, newX_neg = [], []

for i in X_pos:
    for j in i.split(' '):
        newX_pos.append(j)

for i in X_neg:
    for j in  i.split(' '):
        newX_neg.append(j)


X_test, y_test = [test[i][0] for i in range(len(test))], [test[i][1] for i in range(len(test))]

preds = []
print(X_test, '\n', y_test, '\n')
p_yes, p_no = len(newX_pos)/(len(newX_neg) + len(newX_neg)), len(newX_neg)/(len(newX_pos) + len(newX_neg))


for pred in X_test:
    pred = pred.split(' ')

    for i in pred:
        if i in newX_pos:
            p_yes *= newX_pos.count(i)/len(newX_pos)
        if i in newX_neg:
            p_no *= newX_neg.count(i)/len(newX_neg)
    
    #print(p_yes, p_no)
    if p_yes > p_no:
        preds.append('pos')
    else:
        preds.append('neg')

print(preds)

print("Accuracy: ", accuracy_score(y_test, preds))
print("Recall: ", recall_score(y_test, preds, average = 'weighted'))
print("Precision: ", precision_score(y_test, preds, average = 'weighted'))
print("Confusion Matrix: \n", confusion_matrix(y_test, preds))