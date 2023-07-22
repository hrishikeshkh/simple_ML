import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

df = pd.read_csv()

values = df.columns.values[:-1]

le = LabelEncoder()

for i in values:
    df[i] = le.fit_transform(df[i])

y = df.iloc[:, -1]

dsc = tree.DecisionTreeClassifier(criterion='entropy')

dsc.fit(X,y)

tree.plot_tree(dsc)

