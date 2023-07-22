import pandas as pd
import numpy as np

df = pd.read_csv("enjoysport.csv")
a = df.values.tolist()

yes, no = [], []

for i in a:
    if i[-1] == "Yes":
        yes.append(i)
    else:
        no.append(i)

#set a new instance to predicted about for enjoy sport
pred = ["Rainy", 'Cool', 'High', 'True']

#calculate the probability of yes and no
prob_yes = len(yes) / len(a)
prob_no = len(no) / len(a)

pred_prob_yes = [0] * len(pred)
pred_prob_no = [0] * len(pred)

for i in yes:
    for j in i[:-1]:
        if j == pred[i.index(j)]:
            pred_prob_yes[i.index(j)] += 1/len(yes)

for i in no:
    for j in i[:-1]:
        if j == pred[i.index(j)]:
            pred_prob_no[i.index(j)] += 1/len(no)

print("The probability of yes is ", pred_prob_yes)
print("The probability of no is ", pred_prob_no)

#for all 0s in pred_prob_yes and pred_prob_no, replace with 1
pred_prob_yes = [i if i != 0 else 1 for i in pred_prob_yes]
pred_prob_no = [i if i != 0 else 1 for i in pred_prob_no]

yes, no = np.prod(pred_prob_yes) * prob_yes, np.prod(pred_prob_no) * prob_no

print("The probability of yes is ",  yes/(yes + no))
print("The probability of no is ", no/(yes + no))