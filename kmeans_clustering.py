import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize,scale
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

def count(predicts):
	c = 0
	for pre in predicts:
		if pre == True:
			c+=1
	return c

#importing dataset and converting to datasframe
data = pd.read_csv('heart.csv', header=None)
df = pd.DataFrame(data) #data frame

#extracting columns x and y
x = df.iloc[:, 0:5]
x = x.drop(x.columns[1:3], axis=1)
x = pd.DataFrame(scale(x))

y = df.iloc[:, 13]
y = y-1

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

#plotting the data
fig = plt.figure()

ax1 = fig.add_subplot(1,2,1)
ax1.scatter(x[1],x[2], c=y)
ax1.set_title("Original Data")

clusters = 2

model = KMeans(init='k-means++', n_clusters=clusters, n_init=10,random_state=100)

scores = cross_val_score(model, x, y, scoring='accuracy', cv=10)
print ("10-Fold Accuracy : ", scores.mean()*100)

model.fit(x)

predicts = model.predict(x)
print ("Accuracy(Total) = ", count(predicts == np.array(y))/(len(y)*1.0) *100)
centroids = model.cluster_centers_

# print centroids

ax1.scatter(centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

ax2 = fig.add_subplot(1,2,2)
ax2.set_title("KMeans Clustering")
ax2.scatter(x[1],x[2], c=predicts)
ax2.scatter(centroids[:, 1], centroids[:, 2],
            marker='x', s=169, linewidths=3,
            color='b', zorder=10)

#metrics
cm = metrics.confusion_matrix(y, predicts)
print (cm/len(y))
print (metrics.classification_report(y, predicts))

plt.show()
