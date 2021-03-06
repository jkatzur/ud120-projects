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

highest = 0

print "\n\n\n"
print "************************************NEW RUN***************************************"
print "-------------------------------------------"
print "Naive Bayes"
bayes = GaussianNB()
t0 = time()
bayes.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
print "Accuracy is %r and time is %r" %(bayes.score(features_test, labels_test), round(time()-t0, 3))

print "-------------------------------------------"
print "SVM"
svm = SVC(kernel = "rbf", C=10000.0)
t0 = time()
svm.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
print "Accuracy is %r and time is %r" %(svm.score(features_test, labels_test), round(time()-t0, 3))

for alg in ("auto", "ball_tree", "kd_tree", "brute"):
	for neighbors in (1, 2, 3, 4, 5, 10, 50, 100, 250):
		print "-------------------------------------------"
		print "KNN, Alg %s, run with %i" %(alg, neighbors)
		neighbors = KNeighborsClassifier(n_neighbors = neighbors, algorithm = alg)
		t0 = time()
		neighbors.fit(features_train, labels_train)
		print "		training time:", round(time()-t0, 3), "s"
		t0 = time()
		print "Accuracy is %r and time is %r" %(neighbors.score(features_test, labels_test), round(time()-t0, 3))

print "-------------------------------------------"
print "Ada Boost"
boost = AdaBoostClassifier()
t0 = time()
boost.fit(features_train, labels_train)
print "		training time:", round(time()-t0, 3), "s"
t0 = time()
print "Accuracy is %r and time is %r" %(boost.score(features_test, labels_test), round(time()-t0, 3))


print "-------------------------------------------"
print "Random Forest"
forest = RandomForestClassifier(n_estimators=10)
t0 = time()
forest.fit(features_train, labels_train)
print "		training time:", round(time()-t0, 3), "s"
t0 = time()
print "Accuracy is %r and time is %r" %(forest.score(features_test, labels_test), round(time()-t0, 3))

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
