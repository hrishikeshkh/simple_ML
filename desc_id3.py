import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn import tree
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('playtennis.csv')

# Convert categorical data to numerical using LabelEncoder
le = LabelEncoder()
for column in data.columns:
    data[column] = le.fit_transform(data[column])

# Separate features and target variable
X = data.drop(columns=['Play'])
y = data['Play']

# Build the decision tree classifier using ID3 algorithm
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Sample new data for classification (change values accordingly)
new_sample = [[1, 2, 1, 0]]

# Classify the new sample
predicted_class = clf.predict(new_sample)
print("Predicted class:", le.inverse_transform(predicted_class)[0])

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, feature_names=data.columns[:-1].values.tolist(), class_names=['No', 'Yes'], filled=True)
plt.show()