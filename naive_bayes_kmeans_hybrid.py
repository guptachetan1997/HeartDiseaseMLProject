import pandas as pd  
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

#importing dataset and converting to datasframe
data = pd.read_csv('heart.csv', header=None)
df = pd.DataFrame(data) #data frame

#extracting columns x and y separately for kmeans and naive bayes classifiers
x_kmeans = df.iloc[:, 0:5]
x_kmeans = x_kmeans.drop(x_kmeans.columns[1:3], axis=1)
x_kmeans = pd.DataFrame(scale(x_kmeans))

x_naive = df.iloc[:, 0:13]

y = df.iloc[:, 13]
y = y-1

train_test_split = 390 

y_train = pd.Series(y.iloc[:train_test_split])
y_test = pd.Series(y.iloc[train_test_split:])

x_train_kmeans = x_kmeans.iloc[:train_test_split, :]
x_test_kmeans = x_kmeans.iloc[train_test_split:, :]

x_train_naive = x_naive.iloc[:train_test_split, :]
x_test_naive = x_naive.iloc[train_test_split:, :]


#Kmeans model for the processed data
clusters = 5
model_kmeans = KMeans(init='k-means++', n_clusters=clusters, n_init=10,random_state=10000)
model_kmeans.fit(x_train_kmeans)
kmean_predictions = model_kmeans.predict(x_train_kmeans)

#building datset according to clusters
x0 = pd.DataFrame()
y0 = pd.Series()
x1 = pd.DataFrame()
y1 = pd.Series()
x2 = pd.DataFrame()
y2 = pd.Series()
x3 = pd.DataFrame()
y3 = pd.Series()
x4 = pd.DataFrame()
y4 = pd.Series()


for kmean_prediction,i in zip(kmean_predictions, range(len(x_train_kmeans))):
	row_x =  x_train_naive.iloc[i, :]
	row_y = pd.Series(y_train.iloc[i])
	if kmean_prediction == 0:
		x0 = x0.append(row_x, ignore_index=True)
		y0 = y0.append(row_y)
	elif kmean_prediction == 1 :
		x1 = x1.append(row_x, ignore_index=True)
		y1 = y1.append(row_y)
	elif kmean_prediction == 2 :
		x2 = x2.append(row_x, ignore_index=True)
		y2 = y2.append(row_y)
	elif kmean_prediction == 3 :
		x3 = x3.append(row_x, ignore_index=True)
		y3 = y3.append(row_y)
	elif kmean_prediction == 4 :
		x4 = x4.append(row_x, ignore_index=True)
		y4 = y4.append(row_y)

#applying naive bayes classifier
clstr0 = MultinomialNB()
clstr1 = MultinomialNB()
clstr2 = MultinomialNB()
clstr3 = MultinomialNB()
clstr4 = MultinomialNB()

clstr0.fit(x0, y0)
clstr1.fit(x1, y1)
clstr2.fit(x2, y2)
clstr3.fit(x3, y3)
clstr4.fit(x4, y4)

#calculating predictions for the testing based on the hybrid algorithm
predicts = []
c=0
for i in range(len(x_test_kmeans)):
	prediction = model_kmeans.predict(x_test_kmeans.iloc[i, :].reshape(1,-1))
	if prediction == 0:
		pred_naive = clstr0.predict(x_test_naive.iloc[i, :].reshape(1,-1))
	elif prediction == 1:
		pred_naive = clstr1.predict(x_test_naive.iloc[i, :].reshape(1,-1))
	elif prediction == 2:
		pred_naive = clstr2.predict(x_test_naive.iloc[i, :].reshape(1,-1))
	elif prediction == 3:
		pred_naive = clstr3.predict(x_test_naive.iloc[i, :].reshape(1,-1))
	elif prediction == 4:
		pred_naive = clstr4.predict(x_test_naive.iloc[i, :].reshape(1,-1))
	predicts.append(pred_naive)
	if pred_naive == y_test.iloc[i]:
		c+=1

print ("Test set accuracy : ",  ((c*100.0)/len(x_test_kmeans)))

#metrics
predicts = np.array(predicts)
cm = metrics.confusion_matrix(y_test, predicts)
print (cm/len(y_test))
print (metrics.classification_report(y_test, predicts))
