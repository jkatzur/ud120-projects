#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
###plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


from time import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

best = {'score': 0, 'model': '', 'params': {}}

bayes = GaussianNB()
bayes.fit(features_train, labels_train)
score = bayes.score(features_test, labels_test)
if (score > best['score']):
	best['score'] = score
	best['model'] = 'bayes'
	best['params'] = {}


for cs in (1.0, 10.0, 100.0, 1000.0, 10000.0):
	svm = SVC(kernel = "rbf", C=10000.0)
	svm.fit(features_train, labels_train)
	score = svm.score(features_test, labels_test)
	if (score > best['score']):
		best['score'] = score
		best['model'] = 'svc'
		best['params'] = {'c': cs}


for alg in ("auto", "ball_tree", "kd_tree", "brute"):
	for n in (1, 2, 3, 4, 5, 10, 50, 100, 250):
		neighbors = KNeighborsClassifier(n_neighbors = n, algorithm = alg)
		neighbors.fit(features_train, labels_train)
		score = neighbors.score(features_test, labels_test)
		if (score > best['score']):
			best['score'] = score
			best['model'] = 'knn'
			best['params'] = {'alg': alg, 'neighbors': n}

boost = AdaBoostClassifier(n_estimators = 250)
boost.fit(features_train, labels_train)
score = boost.score(features_test, labels_test)
if (score > best['score']):
			best['score'] = score
			best['model'] = 'boost'
			best['params'] = {}

for n in (1, 5, 10, 25, 100):
	forest = RandomForestClassifier(n_estimators = n)
	forest.fit(features_train, labels_train)
	score = forest.score(features_test, labels_test)
	if (score > best['score']):
				best['score'] = score
				best['model'] = 'forest'
				best['params'] = {'n': n}

print "Best Score is %r with model type %s" %(best['score'], best['model'])
print best['params']



try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
