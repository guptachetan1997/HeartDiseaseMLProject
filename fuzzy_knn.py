import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd  

#importing dataset and converting to datasframe
data = pd.read_csv('heart.csv', header=None)
# data = np.genfromtxt(open("heart.csv","rb"),delimiter=",")

df = pd.DataFrame(data) #data frame

#extracting columns x and y
x = df.iloc[:, 0:5]
x = x.drop(x.columns[1:3], axis=1)
x = pd.DataFrame(scale(x))

y = df.iloc[:, 13]
y = y-1

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#plotting the data
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[1],x[2], c=y)
ax1.set_title("Original Data")

model = KNeighborsClassifier(n_neighbors=5, weights='distance')

#10-fold cross validation
scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
# print scores
print ("10-Fold Accuracy : ", scores.mean()*100)

#creation of the confusion matrix
model.fit(x_train,y_train)
print ("Testing Accuracy : ",model.score(x_test, y_test)*100)
predicted = model.predict(x)

ax2 = fig.add_subplot(1,2,2)
ax2.scatter(x[1],x[2], c=predicted)
ax2.set_title("Fuzzy KNearestNeighbours")

cm = metrics.confusion_matrix(y, predicted)
print (cm/len(y))
print (metrics.classification_report(y, predicted))


plt.show()