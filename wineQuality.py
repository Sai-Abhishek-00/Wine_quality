import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, tree

data_path = "winequality-red.csv"
df = pd.read_csv(data_path, sep=";")

print(df)  # data as read by pandas
print(df.index)  # index values
print(df.columns)  # column headers

df.loc[df.quality > 7, ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol', 'quality']]  # highest quality indices

x = df.drop('quality', axis=1)  # features - all cols excluding quality
y = df.quality  # label - quality

# data split 80-20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print(x_train)  # training data before preprocessing
x_train_scaled = preprocessing.scale(x_train)
print(x_train_scaled)  # training data after scaling : noramlized data

#classify using decision trees
dtclf = tree.DecisionTreeClassifier()
dtclf.fit(x_train, y_train)
confidence = dtclf.score(x_test, y_test)
print("Confidence: ", confidence)

y_pred = dtclf.predict(x_test)
y_pred_list = np.array(y_pred).tolist()
for i in range(0,5):
    print(y_pred_list[i])
print(y_test.head())
