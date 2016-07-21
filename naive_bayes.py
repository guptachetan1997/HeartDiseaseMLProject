import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd  

#importing dataset and converting to datasframe
data = pd.read_csv('heart.csv', header=None)
# data = np.genfromtxt(open("heart.csv","rb"),delimiter=",")

df = pd.DataFrame(data) #data frame

#extracting columns x and y
x = df.iloc[:, 0:13]
y = df.iloc[:, 13]
y = y-1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#plotting the data
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[3],x[4], c=y)
ax1.set_title("Original Data")

model = MultinomialNB()

#10-fold cross validation
scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
print ("10-Fold Accuracy : ", scores.mean()*100)

#Testing Accuracy
model.fit(x_train,y_train)

predicts = model.predict(x)
ax2 = fig.add_subplot(1,2,2)
ax2.scatter(x[3],x[4], c=predicts)
ax2.set_title("Naive Bayes")

cm = metrics.confusion_matrix(y, predicts)
print (cm/len(x))
print (metrics.classification_report(y, predicts))

plt.show()