"""Write a program to implement the naïve Bayesian classifier to classify the following English text. 
I love this sandwich, pos This is an amazing place,pos I feel very good about these cheese, This is my best work,pos What an awesome view, pos I do not like this restaurant,neg I am tired of this stuff,neg I can't deal with this,neg He is my sworn enemy, neg My boss is horrible,neg This is an awesome place, pos I do not like the taste of this juice, neg I love to dance,pos I am sick and tired of this place,neg What a great holiday, pos That is a bad locality to stay,neg We will have good fun tomorrow, pos I went to my enemy's house today,neg
Predict 1. Total Instances of Dataset 2.Accuracy  3. Recall 4.  Precision 5. Confusion Matrix"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

msg = pd.read_csv("document.csv", names=["message", "label"])

print("Total Instances of Dataset: ", msg.shape[0])

msg["labelnum"] = msg.label.map({"pos": 1, "neg": 0})
X = msg.message
y = msg.labelnum

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
count_v = CountVectorizer()
Xtrain_dm = count_v.fit_transform(Xtrain)
Xtest_dm = count_v.transform(Xtest)
df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
clf = MultinomialNB()
clf.fit(Xtrain_dm, ytrain)
pred = clf.predict(Xtest_dm)
print("Accuracy Metrics:")
print("Accuracy: ", accuracy_score(ytest, pred))
print("Recall: ", recall_score(ytest, pred))
print("Precision: ", precision_score(ytest, pred))
print("Confusion Matrix: \n", confusion_matrix(ytest, pred))